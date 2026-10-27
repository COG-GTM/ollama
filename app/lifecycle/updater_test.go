package lifecycle

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
)

func TestValidateUpdateURL(t *testing.T) {
	cases := map[string]bool{
		"https://github.com/ollama/ollama/releases/download/v0.1.33/OllamaSetup.exe": true,
		"HTTPS://ollama.com/download/OllamaSetup.exe":                                true,
		"http://ollama.com/download/OllamaSetup.exe":                                 false,
		"file:///C:/Users/alice/evil.exe":                                            false,
		"ftp://ollama.com/OllamaSetup.exe":                                           false,
		"https:///OllamaSetup.exe":                                                   false,
		"OllamaSetup.exe":                                                            false,
		"":                                                                           false,
	}
	for raw, ok := range cases {
		_, err := parseUpdateURL(raw)
		if ok && err != nil {
			t.Errorf("%q: unexpected error %v", raw, err)
		}
		if !ok && err == nil {
			t.Errorf("%q: expected error", raw)
		}
	}
}

func TestStagePathComponent(t *testing.T) {
	for _, name := range []string{"OllamaSetup.exe", "_", "abc123-def"} {
		if _, err := stagePathComponent(name); err != nil {
			t.Errorf("%q: unexpected error %v", name, err)
		}
	}
	for _, name := range []string{"", ".", "..", "../evil.exe", `..\evil.exe`, "a/b", `C:evil.exe`, `\\host\share\x.exe`} {
		if _, err := stagePathComponent(name); err == nil {
			t.Errorf("%q: expected error", name)
		}
	}
}

func TestVerifyUpdateDigest(t *testing.T) {
	payload := []byte("installer")
	sum := sha256.Sum256(payload)
	good := hex.EncodeToString(sum[:])

	if err := verifyUpdateDigest(payload, ""); err != nil {
		t.Errorf("empty digest: %v", err)
	}
	if err := verifyUpdateDigest(payload, good); err != nil {
		t.Errorf("matching digest: %v", err)
	}
	if err := verifyUpdateDigest(payload, strings.ToUpper(good)); err != nil {
		t.Errorf("upper case digest: %v", err)
	}
	if err := verifyUpdateDigest([]byte("tampered"), good); err == nil {
		t.Error("expected mismatch error")
	}
	if err := verifyUpdateDigest(payload, "abcd"); err == nil {
		t.Error("expected invalid digest error")
	}
}

type updateServer struct {
	tls      *httptest.Server
	payload  []byte
	header   http.Header
	redirect string
}

func newUpdateServer(t *testing.T, payload []byte) *updateServer {
	t.Helper()
	s := &updateServer{payload: payload, header: http.Header{}}
	s.header.Set("ETag", `"abc123"`)
	s.tls = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.redirect != "" {
			http.Redirect(w, r, s.redirect, http.StatusFound)
			return
		}
		for k, v := range s.header {
			w.Header()[k] = v
		}
		w.Write(s.payload) //nolint:errcheck
	}))
	t.Cleanup(s.tls.Close)

	origTransport := updateHTTPClient.Transport
	updateHTTPClient.Transport = s.tls.Client().Transport
	t.Cleanup(func() { updateHTTPClient.Transport = origTransport })

	origStage := UpdateStageDir
	UpdateStageDir = t.TempDir()
	t.Cleanup(func() { UpdateStageDir = origStage })

	UpdateDownloaded = false
	t.Cleanup(func() { UpdateDownloaded = false })
	return s
}

func (s *updateServer) url() string {
	return s.tls.URL + "/download/v9.9.9/OllamaSetup.exe"
}

func stagedInstallers(t *testing.T) []string {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(UpdateStageDir, "*", "*"))
	if err != nil {
		t.Fatal(err)
	}
	return files
}

func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func TestDownloadNewReleaseRejectsHTTPURL(t *testing.T) {
	var hits atomic.Int32
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hits.Add(1) }))
	defer plain.Close()
	newUpdateServer(t, nil)

	err := DownloadNewRelease(context.Background(), UpdateResponse{UpdateURL: plain.URL + "/OllamaSetup.exe"})
	if err == nil || !strings.Contains(err.Error(), "non-https") {
		t.Fatalf("expected non-https error, got %v", err)
	}
	if hits.Load() != 0 {
		t.Fatalf("plain http server was contacted %d times", hits.Load())
	}
}

func TestDownloadNewReleaseRejectsRedirectToHTTP(t *testing.T) {
	var hits atomic.Int32
	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Write([]byte("evil")) //nolint:errcheck
	}))
	defer plain.Close()
	s := newUpdateServer(t, nil)
	s.redirect = plain.URL + "/OllamaSetup.exe"

	err := DownloadNewRelease(context.Background(), UpdateResponse{UpdateURL: s.url()})
	if err == nil || !strings.Contains(err.Error(), "non-https") {
		t.Fatalf("expected non-https redirect error, got %v", err)
	}
	if hits.Load() != 0 {
		t.Fatalf("plain http server was contacted %d times", hits.Load())
	}
	if files := stagedInstallers(t); len(files) != 0 {
		t.Fatalf("unexpected staged files %v", files)
	}
}

func TestDownloadNewReleaseRejectsDigestMismatch(t *testing.T) {
	s := newUpdateServer(t, []byte("tampered installer"))

	err := DownloadNewRelease(context.Background(), UpdateResponse{UpdateURL: s.url(), UpdateSHA256: digest([]byte("real installer"))})
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("expected sha256 mismatch, got %v", err)
	}
	if files := stagedInstallers(t); len(files) != 0 {
		t.Fatalf("unexpected staged files %v", files)
	}
	if UpdateDownloaded {
		t.Fatal("UpdateDownloaded set for rejected download")
	}
}

func TestDownloadNewReleaseRejectsUnsafeNames(t *testing.T) {
	for name, set := range map[string]func(h http.Header){
		"filename": func(h http.Header) { h.Set("Content-Disposition", `attachment; filename="../../evil.exe"`) },
		"etag":     func(h http.Header) { h.Set("ETag", `"..\\..\\evil"`) },
	} {
		t.Run(name, func(t *testing.T) {
			s := newUpdateServer(t, []byte("installer"))
			set(s.header)
			err := DownloadNewRelease(context.Background(), UpdateResponse{UpdateURL: s.url()})
			if err == nil || !strings.Contains(err.Error(), "invalid update file name") {
				t.Fatalf("expected invalid name error, got %v", err)
			}
			parent := filepath.Dir(UpdateStageDir)
			if _, err := os.Stat(filepath.Join(parent, "evil.exe")); err == nil {
				t.Fatal("installer written outside the stage directory")
			}
		})
	}
}

func TestDownloadNewReleaseStagesVerifiedPayload(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test payload is not Authenticode signed")
	}
	payload := []byte("installer")
	s := newUpdateServer(t, payload)

	if err := DownloadNewRelease(context.Background(), UpdateResponse{UpdateURL: s.url(), UpdateSHA256: digest(payload)}); err != nil {
		t.Fatal(err)
	}
	files := stagedInstallers(t)
	if len(files) != 1 || filepath.Base(files[0]) != Installer {
		t.Fatalf("unexpected staged files %v", files)
	}
	if !UpdateDownloaded {
		t.Fatal("UpdateDownloaded not set")
	}
}

func TestUpdateHTTPClientRedirectLimit(t *testing.T) {
	via := make([]*http.Request, maxUpdateRedirects)
	u, _ := url.Parse("https://ollama.com/x")
	if err := updateHTTPClient.CheckRedirect(&http.Request{URL: u}, via); err == nil {
		t.Fatal("expected redirect limit error")
	}
	if err := updateHTTPClient.CheckRedirect(&http.Request{URL: u}, via[:1]); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
}
