package lifecycle

import (
	"context"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

type updateServer struct {
	headEtag, getEtag, disposition string
	gets                           atomic.Int32
}

func (s *updateServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	etag := s.headEtag
	if r.Method == http.MethodGet {
		s.gets.Add(1)
		etag = s.getEtag
	}
	if etag != "" {
		w.Header().Set("ETag", `"`+etag+`"`)
	}
	if s.disposition != "" {
		w.Header().Set("Content-Disposition", s.disposition)
	}
	if r.Method == http.MethodGet {
		w.Write([]byte("installer payload")) //nolint:errcheck
	}
}

func setupUpdateTest(t *testing.T, h http.Handler) (root string, srv *httptest.Server) {
	t.Helper()
	root = t.TempDir()

	oldStage, oldClient, oldDownloaded := UpdateStageDir, updateHTTPClient, UpdateDownloaded
	t.Cleanup(func() {
		UpdateStageDir, updateHTTPClient, UpdateDownloaded = oldStage, oldClient, oldDownloaded
	})
	UpdateStageDir = filepath.Join(root, "a", "b", "updates")

	srv = httptest.NewTLSServer(h)
	t.Cleanup(srv.Close)
	updateHTTPClient = &http.Client{Transport: srv.Client().Transport, CheckRedirect: requireHTTPSRedirect}
	return root, srv
}

func filesUnder(t *testing.T, root string) []string {
	t.Helper()
	var files []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			files = append(files, p)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return files
}

func TestDownloadNewReleaseStaysInStageDir(t *testing.T) {
	cases := map[string]*updateServer{
		"backslash traversal": &updateServer{
			headEtag:    `..\..\x`,
			getEtag:     `..\..\x`,
			disposition: `attachment; filename="..\..\y.exe"`,
		},
		"slash traversal": &updateServer{
			headEtag:    "../../x",
			getEtag:     "../../x",
			disposition: `attachment; filename="../../y.exe"`,
		},
		"absolute filename": &updateServer{
			headEtag:    "abc",
			getEtag:     "abc",
			disposition: `attachment; filename="/etc/y.exe"`,
		},
		"get etag differs from head": &updateServer{
			headEtag: "safe",
			getEtag:  "../../../evil",
		},
		"no etag": &updateServer{
			disposition: `attachment; filename="..\..\..\y.exe"`,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			root, srv := setupUpdateTest(t, tc)

			if err := DownloadNewRelease(context.Background(), UpdateResponse{UpdateURL: srv.URL + "/download/OllamaSetup.exe"}); err != nil {
				t.Fatal(err)
			}

			want, err := stageInstallerPath(tc.headEtag)
			if err != nil {
				t.Fatal(err)
			}
			files := filesUnder(t, root)
			if len(files) != 1 || files[0] != want {
				t.Fatalf("expected only %s to be written, got %v", want, files)
			}
			if filepath.Dir(filepath.Dir(want)) != UpdateStageDir || filepath.Base(want) != Installer {
				t.Fatalf("installer staged at unexpected location %s", want)
			}
			if got, _ := os.ReadFile(want); string(got) != "installer payload" {
				t.Fatalf("unexpected payload %q", got)
			}
			if !UpdateDownloaded {
				t.Fatal("expected UpdateDownloaded to be set")
			}
		})
	}
}

func TestDownloadNewReleaseSkipsExistingDownload(t *testing.T) {
	s := &updateServer{headEtag: "v1", getEtag: "v1"}
	_, srv := setupUpdateTest(t, s)
	resp := UpdateResponse{UpdateURL: srv.URL + "/download/OllamaSetup.exe"}

	for range 2 {
		if err := DownloadNewRelease(context.Background(), resp); err != nil {
			t.Fatal(err)
		}
	}
	if n := s.gets.Load(); n != 1 {
		t.Fatalf("expected 1 GET, got %d", n)
	}
}

func TestDownloadNewReleaseRequiresHTTPS(t *testing.T) {
	s := &updateServer{headEtag: "v1", getEtag: "v1"}
	root, _ := setupUpdateTest(t, s)
	plain := httptest.NewServer(s)
	defer plain.Close()

	err := DownloadNewRelease(context.Background(), UpdateResponse{UpdateURL: plain.URL + "/download/OllamaSetup.exe"})
	if err == nil || !strings.Contains(err.Error(), "non-https") {
		t.Fatalf("expected non-https error, got %v", err)
	}
	if n := s.gets.Load(); n != 0 {
		t.Fatalf("expected no download, got %d GETs", n)
	}
	if files := filesUnder(t, root); len(files) != 0 {
		t.Fatalf("expected no files, got %v", files)
	}
}

func TestDownloadNewReleaseRejectsHTTPRedirect(t *testing.T) {
	s := &updateServer{headEtag: "v1", getEtag: "v1"}
	plain := httptest.NewServer(s)
	defer plain.Close()
	root, srv := setupUpdateTest(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, plain.URL+"/download/OllamaSetup.exe", http.StatusFound)
	}))

	err := DownloadNewRelease(context.Background(), UpdateResponse{UpdateURL: srv.URL + "/download/OllamaSetup.exe"})
	if err == nil || !strings.Contains(err.Error(), "non-https") {
		t.Fatalf("expected non-https redirect error, got %v", err)
	}
	if n := s.gets.Load(); n != 0 {
		t.Fatalf("expected no download, got %d GETs", n)
	}
	if files := filesUnder(t, root); len(files) != 0 {
		t.Fatalf("expected no files, got %v", files)
	}
}
