//go:build !windows

package gpu

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupPayloadsTest(t *testing.T) string {
	t.Helper()
	parent := t.TempDir()
	t.Setenv("OLLAMA_RUNNERS_DIR", "")
	t.Setenv("OLLAMA_TMPDIR", parent)
	lock.Lock()
	payloadsDir, payloadsTmpDir = "", ""
	lock.Unlock()
	t.Cleanup(func() {
		lock.Lock()
		payloadsDir, payloadsTmpDir = "", ""
		lock.Unlock()
	})
	return parent
}

func plantRunner(t *testing.T, runners string, mode os.FileMode) string {
	t.Helper()
	variantDir := filepath.Join(runners, "cpu_avx2")
	require.NoError(t, os.MkdirAll(variantDir, 0o777))
	server := filepath.Join(variantDir, "ollama_llama_server")
	require.NoError(t, os.WriteFile(server, []byte("#!/bin/sh\ntouch /tmp/pwned\n"), 0o755))
	require.NoError(t, os.Chmod(variantDir, mode))
	return server
}

func TestPayloadsDirIsPrivateUnderTmpDir(t *testing.T) {
	parent := setupPayloadsTest(t)

	dir, err := PayloadsDir()
	require.NoError(t, err)
	assert.Equal(t, "runners", filepath.Base(dir))

	tmpDir := filepath.Dir(dir)
	assert.Equal(t, parent, filepath.Dir(tmpDir))
	info, err := os.Lstat(tmpDir)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o700), info.Mode().Perm())
	assert.True(t, IsTemporaryPayloadsDir(dir))

	again, err := PayloadsDir()
	require.NoError(t, err)
	assert.Equal(t, dir, again)
}

func TestPayloadsDirNotReusedAfterRemoval(t *testing.T) {
	setupPayloadsTest(t)

	dir, err := PayloadsDir()
	require.NoError(t, err)
	tmpDir := filepath.Dir(dir)

	// A tmp cleaner removes the directory and another user recreates the same path
	require.NoError(t, os.RemoveAll(tmpDir))
	require.NoError(t, os.Mkdir(tmpDir, 0o777))
	require.NoError(t, os.Chmod(tmpDir, 0o777))
	planted := plantRunner(t, filepath.Join(tmpDir, "runners"), 0o777)

	assert.Error(t, VerifyPayload(planted))

	fresh, err := PayloadsDir()
	require.NoError(t, err)
	assert.NotEqual(t, dir, fresh)
	assert.False(t, IsTemporaryPayloadsDir(dir))
	assert.Error(t, VerifyPayload(planted), "planted runner outside the fresh payload dir must be rejected")
}

func TestPayloadsDirNotReusedWhenRemoved(t *testing.T) {
	setupPayloadsTest(t)

	dir, err := PayloadsDir()
	require.NoError(t, err)
	require.NoError(t, os.RemoveAll(filepath.Dir(dir)))

	fresh, err := PayloadsDir()
	require.NoError(t, err)
	assert.NotEqual(t, dir, fresh)
	_, err = os.Stat(filepath.Dir(fresh))
	assert.NoError(t, err)
}

func TestPayloadsDirNotReusedWhenReplacedBySymlink(t *testing.T) {
	setupPayloadsTest(t)

	dir, err := PayloadsDir()
	require.NoError(t, err)
	tmpDir := filepath.Dir(dir)

	attacker := t.TempDir()
	plantRunner(t, filepath.Join(attacker, "runners"), 0o755)
	require.NoError(t, os.RemoveAll(tmpDir))
	require.NoError(t, os.Symlink(attacker, tmpDir))

	fresh, err := PayloadsDir()
	require.NoError(t, err)
	assert.NotEqual(t, dir, fresh)
}

func TestVerifyPayload(t *testing.T) {
	setupPayloadsTest(t)

	dir, err := PayloadsDir()
	require.NoError(t, err)

	variantDir := filepath.Join(dir, "cpu")
	require.NoError(t, os.MkdirAll(variantDir, 0o755))
	server := filepath.Join(variantDir, "ollama_llama_server")
	require.NoError(t, os.WriteFile(server, []byte("ok"), 0o755))
	assert.NoError(t, VerifyPayload(server))
	assert.NoError(t, VerifyPayloadDir(variantDir))

	// missing
	assert.ErrorIs(t, VerifyPayload(filepath.Join(variantDir, "missing")), os.ErrNotExist)

	// group/other writable binary
	require.NoError(t, os.Chmod(server, 0o777))
	assert.Error(t, VerifyPayload(server))
	require.NoError(t, os.Chmod(server, 0o755))

	// group/other writable directory in the chain
	require.NoError(t, os.Chmod(variantDir, 0o777))
	assert.Error(t, VerifyPayload(server))
	require.NoError(t, os.Chmod(variantDir, 0o755))

	// symlinked binary
	target := filepath.Join(t.TempDir(), "evil")
	require.NoError(t, os.WriteFile(target, []byte("evil"), 0o755))
	link := filepath.Join(variantDir, "link")
	require.NoError(t, os.Symlink(target, link))
	assert.Error(t, VerifyPayload(link))

	// symlinked directory
	linkDir := filepath.Join(dir, "cpu_avx2")
	require.NoError(t, os.Symlink(filepath.Dir(target), linkDir))
	assert.Error(t, VerifyPayload(filepath.Join(linkDir, "evil")))

	// outside the payload dir
	assert.Error(t, VerifyPayload(target))
	assert.Error(t, VerifyPayload(filepath.Join(dir, "..", "..", "ollama_llama_server")))
}

func TestResetPayloadsDir(t *testing.T) {
	setupPayloadsTest(t)

	dir, err := PayloadsDir()
	require.NoError(t, err)

	ResetPayloadsDir()
	_, err = os.Stat(filepath.Dir(dir))
	assert.ErrorIs(t, err, os.ErrNotExist)

	fresh, err := PayloadsDir()
	require.NoError(t, err)
	assert.NotEqual(t, dir, fresh)
}

func TestRunnersDirIsNotTemporary(t *testing.T) {
	setupPayloadsTest(t)
	runners := t.TempDir()
	t.Setenv("OLLAMA_RUNNERS_DIR", runners)

	dir, err := PayloadsDir()
	require.NoError(t, err)
	assert.Equal(t, runners, dir)
	assert.False(t, IsTemporaryPayloadsDir(dir))

	ResetPayloadsDir()
	_, err = os.Stat(runners)
	assert.NoError(t, err, "operator supplied runners dir must never be removed")
}

func TestCleanupTmpDirsSkipsSymlinks(t *testing.T) {
	parent := setupPayloadsTest(t)

	victim := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(victim, "keep"), []byte("x"), 0o600))
	require.NoError(t, os.Symlink(victim, filepath.Join(parent, "ollama-link")))

	stale := filepath.Join(parent, "ollama-stale")
	require.NoError(t, os.Mkdir(stale, 0o700))

	cleanupTmpDirs(parent)

	_, err := os.Stat(filepath.Join(victim, "keep"))
	assert.NoError(t, err)
	_, err = os.Lstat(stale)
	assert.ErrorIs(t, err, os.ErrNotExist)
}
