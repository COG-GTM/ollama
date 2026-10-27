package lifecycle

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/version"
)

var (
	UpdateCheckURLBase  = "https://ollama.com/api/update"
	UpdateDownloaded    = false
	UpdateCheckInterval = 60 * 60 * time.Second
)

// TODO - maybe move up to the API package?
type UpdateResponse struct {
	UpdateURL     string `json:"url"`
	UpdateVersion string `json:"version"`
	// UpdateSHA256 is the optional hex encoded SHA-256 of the installer.
	UpdateSHA256 string `json:"sha256,omitempty"`
}

const maxUpdateRedirects = 10

// updateHTTPClient refuses to follow redirects away from https so the
// installer is never fetched over plain http.
var updateHTTPClient = &http.Client{
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= maxUpdateRedirects {
			return fmt.Errorf("stopped after %d redirects", maxUpdateRedirects)
		}
		return validateUpdateURL(req.URL)
	},
}

func validateUpdateURL(u *url.URL) error {
	if u == nil || !strings.EqualFold(u.Scheme, "https") || u.Host == "" {
		return fmt.Errorf("refusing non-https update URL %q", u)
	}
	return nil
}

func parseUpdateURL(raw string) (*url.URL, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("invalid update URL: %w", err)
	}
	if err := validateUpdateURL(u); err != nil {
		return nil, err
	}
	return u, nil
}

// stagePathComponent rejects server supplied etags and filenames that are
// not a single, plain path element.
func stagePathComponent(name string) (string, error) {
	if name == "" || name == "." || name == ".." || strings.ContainsAny(name, `/\:`) || name != filepath.Base(name) {
		return "", fmt.Errorf("invalid update file name %q", name)
	}
	return name, nil
}

func verifyUpdateDigest(payload []byte, expected string) error {
	if expected == "" {
		return nil
	}
	want, err := hex.DecodeString(strings.TrimSpace(expected))
	if err != nil || len(want) != sha256.Size {
		return fmt.Errorf("invalid installer sha256 %q", expected)
	}
	got := sha256.Sum256(payload)
	if !bytes.Equal(got[:], want) {
		return fmt.Errorf("installer sha256 mismatch: got %x, expected %x", got, want)
	}
	return nil
}

func IsNewReleaseAvailable(ctx context.Context) (bool, UpdateResponse) {
	var updateResp UpdateResponse

	requestURL, err := url.Parse(UpdateCheckURLBase)
	if err != nil {
		return false, updateResp
	}

	query := requestURL.Query()
	query.Add("os", runtime.GOOS)
	query.Add("arch", runtime.GOARCH)
	query.Add("version", version.Version)
	query.Add("ts", fmt.Sprintf("%d", time.Now().Unix()))

	nonce, err := auth.NewNonce(rand.Reader, 16)
	if err != nil {
		return false, updateResp
	}

	query.Add("nonce", nonce)
	requestURL.RawQuery = query.Encode()

	data := []byte(fmt.Sprintf("%s,%s", http.MethodGet, requestURL.RequestURI()))
	signature, err := auth.Sign(ctx, data)
	if err != nil {
		return false, updateResp
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
	if err != nil {
		slog.Warn(fmt.Sprintf("failed to check for update: %s", err))
		return false, updateResp
	}
	req.Header.Set("Authorization", signature)
	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s) Go/%s", version.Version, runtime.GOARCH, runtime.GOOS, runtime.Version()))

	slog.Debug("checking for available update", "requestURL", requestURL)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		slog.Warn(fmt.Sprintf("failed to check for update: %s", err))
		return false, updateResp
	}
	defer resp.Body.Close()

	if resp.StatusCode == 204 {
		slog.Debug("check update response 204 (current version is up to date)")
		return false, updateResp
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		slog.Warn(fmt.Sprintf("failed to read body response: %s", err))
	}

	if resp.StatusCode != 200 {
		slog.Info(fmt.Sprintf("check update error %d - %.96s", resp.StatusCode, string(body)))
		return false, updateResp
	}
	err = json.Unmarshal(body, &updateResp)
	if err != nil {
		slog.Warn(fmt.Sprintf("malformed response checking for update: %s", err))
		return false, updateResp
	}
	if _, err := parseUpdateURL(updateResp.UpdateURL); err != nil {
		slog.Warn(fmt.Sprintf("ignoring update: %s", err))
		return false, updateResp
	}
	// Extract the version string from the URL in the github release artifact path
	updateResp.UpdateVersion = path.Base(path.Dir(updateResp.UpdateURL))

	slog.Info("New update available at " + updateResp.UpdateURL)
	return true, updateResp
}

func DownloadNewRelease(ctx context.Context, updateResp UpdateResponse) error {
	updateURL, err := parseUpdateURL(updateResp.UpdateURL)
	if err != nil {
		return err
	}

	// Do a head first to check etag info
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, updateURL.String(), nil)
	if err != nil {
		return err
	}

	resp, err := updateHTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("error checking update: %w", err)
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("unexpected status attempting to download update %d", resp.StatusCode)
	}
	resp.Body.Close()
	etag := strings.Trim(resp.Header.Get("etag"), "\"")
	if etag == "" {
		slog.Debug("no etag detected, falling back to filename based dedup")
		etag = "_"
	}
	if etag, err = stagePathComponent(etag); err != nil {
		return err
	}
	filename := Installer
	_, params, err := mime.ParseMediaType(resp.Header.Get("content-disposition"))
	if err == nil {
		filename = params["filename"]
	}
	if filename, err = stagePathComponent(filename); err != nil {
		return err
	}

	stageFilename := filepath.Join(UpdateStageDir, etag, filename)

	// Check to see if we already have it downloaded
	_, err = os.Stat(stageFilename)
	if err == nil {
		err := verifyStagedUpdate(stageFilename, updateResp.UpdateSHA256)
		if err == nil {
			slog.Info("update already downloaded")
			UpdateDownloaded = true
			return nil
		}
		slog.Warn(fmt.Sprintf("discarding previously downloaded update %s: %s", stageFilename, err))
	}

	cleanupOldDownloads()

	req.Method = http.MethodGet
	resp, err = updateHTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("error checking update: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("unexpected status attempting to download update %d", resp.StatusCode)
	}
	etag = strings.Trim(resp.Header.Get("etag"), "\"")
	if etag == "" {
		slog.Debug("no etag detected, falling back to filename based dedup") // TODO probably can get rid of this redundant log
		etag = "_"
	}
	if etag, err = stagePathComponent(etag); err != nil {
		return err
	}

	stageFilename = filepath.Join(UpdateStageDir, etag, filename)

	_, err = os.Stat(filepath.Dir(stageFilename))
	if errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(filepath.Dir(stageFilename), 0o755); err != nil {
			return fmt.Errorf("create ollama dir %s: %v", filepath.Dir(stageFilename), err)
		}
	}

	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read body response: %w", err)
	}
	if err := verifyUpdateDigest(payload, updateResp.UpdateSHA256); err != nil {
		return err
	}

	// Write to a temporary name so an unverified installer is never left
	// where DoUpgrade looks for it.
	pendingFilename := stageFilename + ".pending"
	if err := writeStagedPayload(pendingFilename, payload); err != nil {
		os.Remove(pendingFilename) //nolint:errcheck
		return err
	}
	if err := verifyInstaller(pendingFilename); err != nil {
		os.Remove(pendingFilename) //nolint:errcheck
		return fmt.Errorf("downloaded update failed verification: %w", err)
	}
	if err := os.Rename(pendingFilename, stageFilename); err != nil {
		os.Remove(pendingFilename) //nolint:errcheck
		return fmt.Errorf("stage update %s: %w", stageFilename, err)
	}
	slog.Info("new update downloaded and verified " + stageFilename)

	UpdateDownloaded = true
	return nil
}

func writeStagedPayload(filename string, payload []byte) error {
	fp, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		return fmt.Errorf("write payload %s: %w", filename, err)
	}
	if n, err := fp.Write(payload); err != nil || n != len(payload) {
		fp.Close()
		return fmt.Errorf("write payload %s: %d vs %d -- %w", filename, n, len(payload), err)
	}
	if err := fp.Close(); err != nil {
		return fmt.Errorf("write payload %s: %w", filename, err)
	}
	return nil
}

func verifyStagedUpdate(filename, expectedSHA256 string) error {
	if expectedSHA256 != "" {
		payload, err := os.ReadFile(filename)
		if err != nil {
			return err
		}
		if err := verifyUpdateDigest(payload, expectedSHA256); err != nil {
			return err
		}
	}
	return verifyInstaller(filename)
}

func cleanupOldDownloads() {
	files, err := os.ReadDir(UpdateStageDir)
	if err != nil && errors.Is(err, os.ErrNotExist) {
		// Expected behavior on first run
		return
	} else if err != nil {
		slog.Warn(fmt.Sprintf("failed to list stage dir: %s", err))
		return
	}
	for _, file := range files {
		fullname := filepath.Join(UpdateStageDir, file.Name())
		slog.Debug("cleaning up old download: " + fullname)
		err = os.RemoveAll(fullname)
		if err != nil {
			slog.Warn(fmt.Sprintf("failed to cleanup stale update download %s", err))
		}
	}
}

func StartBackgroundUpdaterChecker(ctx context.Context, cb func(string) error) {
	go func() {
		// Don't blast an update message immediately after startup
		// time.Sleep(30 * time.Second)
		time.Sleep(3 * time.Second)

		for {
			available, resp := IsNewReleaseAvailable(ctx)
			if available {
				err := DownloadNewRelease(ctx, resp)
				if err != nil {
					slog.Error(fmt.Sprintf("failed to download new release: %s", err))
				} else {
					err = cb(resp.UpdateVersion)
					if err != nil {
						slog.Warn(fmt.Sprintf("failed to register update available with tray: %s", err))
					}
				}
			}
			select {
			case <-ctx.Done():
				slog.Debug("stopping background update checker")
				return
			default:
				time.Sleep(UpdateCheckInterval)
			}
		}
	}()
}
