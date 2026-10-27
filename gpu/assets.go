package gpu

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

var (
	lock        sync.Mutex
	payloadsDir = ""
	// payloadsTmpDir is the private temporary directory this process created to
	// hold extracted payloads. It is empty when the runners come from
	// OLLAMA_RUNNERS_DIR or are shipped alongside the executable (Windows).
	payloadsTmpDir = ""
)

func PayloadsDir() (string, error) {
	lock.Lock()
	defer lock.Unlock()
	if payloadsTmpDir != "" {
		// Tmp cleaners may remove our directory while we keep running, and any
		// local user could then recreate the same path. Never reuse a temp
		// directory that is no longer the private one we created.
		if err := checkOwnedPath(payloadsTmpDir, true); err != nil {
			slog.Warn("payload directory is missing or no longer trusted, creating a new one", "dir", payloadsTmpDir, "error", err)
			payloadsDir, payloadsTmpDir = "", ""
		}
	}
	if payloadsDir == "" {
		runnersDir := os.Getenv("OLLAMA_RUNNERS_DIR")
		// On Windows we do not carry the payloads inside the main executable
		if runtime.GOOS == "windows" && runnersDir == "" {
			appExe, err := os.Executable()
			if err != nil {
				slog.Error("failed to lookup executable path", "error", err)
				return "", err
			}

			cwd, err := os.Getwd()
			if err != nil {
				slog.Error("failed to lookup working directory", "error", err)
				return "", err
			}

			var paths []string
			for _, root := range []string{filepath.Dir(appExe), cwd} {
				paths = append(paths,
					filepath.Join(root),
					filepath.Join(root, "windows-"+runtime.GOARCH),
					filepath.Join(root, "dist", "windows-"+runtime.GOARCH),
				)
			}

			// Try a few variations to improve developer experience when building from source in the local tree
			for _, p := range paths {
				candidate := filepath.Join(p, "ollama_runners")
				_, err := os.Stat(candidate)
				if err == nil {
					runnersDir = candidate
					break
				}
			}
			if runnersDir == "" {
				err = fmt.Errorf("unable to locate llm runner directory.  Set OLLAMA_RUNNERS_DIR to the location of 'ollama_runners'")
				slog.Error("incomplete distribution", "error", err)
				return "", err
			}
		}
		if runnersDir != "" {
			payloadsDir = runnersDir
			return payloadsDir, nil
		}

		// The remainder only applies on non-windows where we still carry payloads in the main executable
		parentDir := os.Getenv("OLLAMA_TMPDIR")
		if parentDir != "" {
			if err := os.MkdirAll(parentDir, 0o755); err != nil {
				return "", fmt.Errorf("failed to generate tmp dir %s: %w", parentDir, err)
			}
		}
		cleanupTmpDirs(parentDir)

		// Always create a fresh, randomly named directory only we can write to
		tmpDir, err := os.MkdirTemp(parentDir, "ollama")
		if err != nil {
			return "", fmt.Errorf("failed to generate tmp dir: %w", err)
		}

		// Track our pid so we can clean up orphaned tmpdirs
		pidFilePath := filepath.Join(tmpDir, "ollama.pid")
		if err := os.WriteFile(pidFilePath, []byte(fmt.Sprint(os.Getpid())), 0o600); err != nil {
			return "", err
		}

		// We create a distinct subdirectory for payloads within the tmpdir
		// This will typically look like /tmp/ollama3208993108/runners on linux
		payloadsTmpDir = tmpDir
		payloadsDir = filepath.Join(tmpDir, "runners")
	}
	return payloadsDir, nil
}

// ResetPayloadsDir discards the cached temporary payload directory, removing it
// if it is still ours, so the next PayloadsDir call creates a fresh one.
func ResetPayloadsDir() {
	lock.Lock()
	defer lock.Unlock()
	if payloadsTmpDir == "" {
		return
	}
	if checkOwnedPath(payloadsTmpDir, true) == nil {
		if err := os.RemoveAll(payloadsTmpDir); err != nil {
			slog.Debug("unable to remove payload dir", "dir", payloadsTmpDir, "error", err)
		}
	}
	payloadsDir, payloadsTmpDir = "", ""
}

// IsTemporaryPayloadsDir reports whether dir is the temporary payload directory
// managed by this process (as opposed to an operator supplied OLLAMA_RUNNERS_DIR).
func IsTemporaryPayloadsDir(dir string) bool {
	lock.Lock()
	defer lock.Unlock()
	return payloadsTmpDir != "" && dir == payloadsDir
}

// VerifyPayload returns an error unless path is a regular file that is safe to
// load or execute. Inside the temporary payload directory no path component may
// be a symlink, owned by another user, or writable by group or others.
func VerifyPayload(path string) error {
	return verifyPayloadPath(path, false)
}

// VerifyPayloadDir is like VerifyPayload but for a directory.
func VerifyPayloadDir(path string) error {
	return verifyPayloadPath(path, true)
}

func verifyPayloadPath(path string, wantDir bool) error {
	lock.Lock()
	root := payloadsTmpDir
	lock.Unlock()

	if root == "" {
		info, err := os.Stat(path)
		if err != nil {
			return err
		}
		if wantDir && !info.IsDir() {
			return fmt.Errorf("%s is not a directory", path)
		}
		if !wantDir && !info.Mode().IsRegular() {
			return fmt.Errorf("%s is not a regular file", path)
		}
		return nil
	}

	rel, err := filepath.Rel(root, filepath.Clean(path))
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return fmt.Errorf("%s is outside the payload directory %s", path, root)
	}
	if err := checkOwnedPath(root, true); err != nil {
		return err
	}
	cur := root
	parts := strings.Split(rel, string(filepath.Separator))
	for i, part := range parts {
		cur = filepath.Join(cur, part)
		if err := checkOwnedPath(cur, wantDir || i < len(parts)-1); err != nil {
			return err
		}
	}
	return nil
}

// checkOwnedPath verifies path is not a symlink, has the expected type, is owned
// by the current user and is not writable by group or others.
func checkOwnedPath(path string, wantDir bool) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	mode := info.Mode()
	switch {
	case mode&os.ModeSymlink != 0:
		return fmt.Errorf("%s is a symlink", path)
	case wantDir && !mode.IsDir():
		return fmt.Errorf("%s is not a directory", path)
	case !wantDir && !mode.IsRegular():
		return fmt.Errorf("%s is not a regular file", path)
	}
	return checkOwner(path, info)
}

// Best effort to clean up prior tmpdirs
func cleanupTmpDirs(parentDir string) {
	if parentDir == "" {
		parentDir = os.TempDir()
	}
	dirs, err := filepath.Glob(filepath.Join(parentDir, "ollama*"))
	if err != nil {
		return
	}
	for _, d := range dirs {
		// Only consider real directories we own; never follow symlinks or touch other users' files
		if checkOwnedPath(d, true) != nil {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(d, "ollama.pid"))
		if err == nil {
			pid, err := strconv.Atoi(string(raw))
			if err == nil {
				if proc, err := os.FindProcess(int(pid)); err == nil && !errors.Is(proc.Signal(syscall.Signal(0)), os.ErrProcessDone) {
					// Another running ollama, ignore this tmpdir
					continue
				}
			}
		} else {
			slog.Debug("failed to open ollama.pid", "path", d, "error", err)
		}
		err = os.RemoveAll(d)
		if err != nil {
			slog.Debug("unable to cleanup stale tmpdir", "path", d, "error", err)
		}
	}
}

func Cleanup() {
	lock.Lock()
	defer lock.Unlock()
	if payloadsTmpDir == "" {
		return
	}
	// We want to fully clean up the tmpdir parent of the payloads dir, but only if it is still ours
	tmpDir := payloadsTmpDir
	if err := checkOwnedPath(tmpDir, true); err != nil {
		slog.Warn("not cleaning up untrusted payload dir", "dir", tmpDir, "error", err)
		return
	}
	slog.Debug("cleaning up", "dir", tmpDir)
	err := os.RemoveAll(tmpDir)
	if err != nil {
		// If we remove too quickly a library may still be in-use and fail to remove
		time.Sleep(1000 * time.Millisecond)
		err = os.RemoveAll(tmpDir)
		if err != nil {
			slog.Warn("failed to clean up", "dir", tmpDir, "err", err)
		}
	}
}

func UpdatePath(dir string) {
	if runtime.GOOS == "windows" {
		tmpDir := filepath.Dir(dir)
		pathComponents := strings.Split(os.Getenv("PATH"), ";")
		i := 0
		for _, comp := range pathComponents {
			if strings.EqualFold(comp, dir) {
				return
			}
			// Remove any other prior paths to our temp dir
			if !strings.HasPrefix(strings.ToLower(comp), strings.ToLower(tmpDir)) {
				pathComponents[i] = comp
				i++
			}
		}
		newPath := strings.Join(append([]string{dir}, pathComponents...), ";")
		slog.Info("updating", "PATH", newPath)
		os.Setenv("PATH", newPath)
	}
	// linux and darwin rely on rpath
}
