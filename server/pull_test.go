package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPullModelRejectsInvalidDigests(t *testing.T) {
	modelsDir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", modelsDir)

	target := filepath.Join(t.TempDir(), "traversal-test")
	rel, err := filepath.Rel(filepath.Join(modelsDir, "blobs"), target)
	if err != nil {
		t.Fatal(err)
	}

	validDigest := "sha256:" + strings.Repeat("0", 64)

	cases := []struct {
		name     string
		manifest ManifestV2
	}{
		{
			name: "layer",
			manifest: ManifestV2{
				SchemaVersion: 2,
				Config:        &Layer{Digest: validDigest},
				Layers:        []*Layer{{Digest: rel}},
			},
		},
		{
			name: "config",
			manifest: ManifestV2{
				SchemaVersion: 2,
				Config:        &Layer{Digest: rel},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var blobRequested bool
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case strings.Contains(r.URL.Path, "/manifests/"):
					w.Header().Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
					_ = json.NewEncoder(w).Encode(tc.manifest)
				default:
					blobRequested = true
					_, _ = w.Write([]byte("malicious"))
				}
			}))
			defer srv.Close()

			u, err := url.Parse(srv.URL)
			if err != nil {
				t.Fatal(err)
			}

			name := "http://" + u.Host + "/x/y:latest"
			err = PullModel(context.Background(), name, &registryOptions{Insecure: true}, func(api.ProgressResponse) {})
			if !errors.Is(err, ErrInvalidDigestFormat) {
				t.Fatalf("expected invalid digest error, got %v", err)
			}

			if blobRequested {
				t.Error("blob was requested from registry")
			}

			if _, err := os.Stat(target); !errors.Is(err, os.ErrNotExist) {
				t.Errorf("traversal target should not exist: %v", err)
			}
		})
	}
}

func TestVerifyFileRejectsDirectory(t *testing.T) {
	dir := t.TempDir()
	err := verifyFile(dir, "sha256:"+strings.Repeat("0", 64))
	if !errors.Is(err, errDigestMismatch) {
		t.Fatalf("expected digest mismatch, got %v", err)
	}
}
