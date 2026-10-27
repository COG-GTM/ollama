package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func sha256Digest(b []byte) string {
	return fmt.Sprintf("sha256:%x", sha256.Sum256(b))
}

func TestGetBlobsPath(t *testing.T) {
	tmp := t.TempDir()
	models := filepath.Join(tmp, "models")
	t.Setenv("OLLAMA_MODELS", models)

	hex := strings.Repeat("a", 64)
	blobs := filepath.Join(models, "blobs")

	valid := []struct {
		digest string
		want   string
	}{
		{"", blobs},
		{"sha256:" + hex, filepath.Join(blobs, "sha256-"+hex)},
		{"sha256-" + hex, filepath.Join(blobs, "sha256-"+hex)},
	}

	for _, tt := range valid {
		got, err := GetBlobsPath(tt.digest)
		if err != nil {
			t.Fatalf("GetBlobsPath(%q): unexpected error: %v", tt.digest, err)
		}
		if got != tt.want {
			t.Errorf("GetBlobsPath(%q) = %q, want %q", tt.digest, got, tt.want)
		}
	}

	invalid := []string{
		"../../traversal-test",
		"../../../../etc/ld.so.preload",
		"sha256:../../traversal-test",
		"sha256-../../traversal-test",
		"sha256:" + hex + "/../../../traversal-test",
		"sha256:" + hex[:63],
		"sha256:" + hex + "a",
		"sha256:" + strings.ToUpper(hex),
		"sha512:" + hex,
		"sha256:" + hex[:62] + "/x",
		"sha256",
	}

	for _, digest := range invalid {
		got, err := GetBlobsPath(digest)
		if !errors.Is(err, errInvalidDigestFormat) {
			t.Errorf("GetBlobsPath(%q) = %q, %v; want errInvalidDigestFormat", digest, got, err)
		}
	}

	if _, err := os.Stat(filepath.Join(tmp, "traversal-test")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("traversal target should not exist, stat err = %v", err)
	}
}

func TestValidateManifest(t *testing.T) {
	good := sha256Digest([]byte("good"))

	cases := []struct {
		name string
		m    *ManifestV2
		ok   bool
	}{
		{"valid", &ManifestV2{Config: &Layer{Digest: good}, Layers: []*Layer{{Digest: good}}}, true},
		{"nil manifest", nil, false},
		{"missing config", &ManifestV2{Layers: []*Layer{{Digest: good}}}, false},
		{"traversal config", &ManifestV2{Config: &Layer{Digest: "../../x"}, Layers: []*Layer{{Digest: good}}}, false},
		{"traversal layer", &ManifestV2{Config: &Layer{Digest: good}, Layers: []*Layer{{Digest: good}, {Digest: "../../../../etc/ld.so.preload"}}}, false},
		{"dash form layer", &ManifestV2{Config: &Layer{Digest: good}, Layers: []*Layer{{Digest: strings.Replace(good, ":", "-", 1)}}}, false},
		{"nil layer", &ManifestV2{Config: &Layer{Digest: good}, Layers: []*Layer{nil}}, false},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			err := validateManifest(tt.m)
			if tt.ok && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !tt.ok && err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestGetSHA256DigestDirectory(t *testing.T) {
	dir := t.TempDir()
	f, err := os.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if _, _, err := GetSHA256Digest(f); err == nil {
		t.Fatal("expected error hashing a directory")
	}
}

func TestVerifyBlobDirectory(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", filepath.Join(t.TempDir(), "models"))

	digest := sha256Digest([]byte("dir"))
	fp, err := GetBlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(fp, 0o755); err != nil {
		t.Fatal(err)
	}

	if err := verifyBlob(digest); err == nil {
		t.Fatal("expected error verifying a directory blob")
	}
}

type fakeRegistry struct {
	manifest   *ManifestV2
	blobs      map[string][]byte
	blobHits   atomic.Int32
	serverURL  string
	httpServer *httptest.Server
}

func newFakeRegistry(t *testing.T, manifest *ManifestV2, blobs map[string][]byte) *fakeRegistry {
	t.Helper()

	r := &fakeRegistry{manifest: manifest, blobs: blobs}
	r.httpServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		switch {
		case req.URL.Path == "/v2/x/y/manifests/latest":
			w.Header().Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
			_ = json.NewEncoder(w).Encode(r.manifest)
		case strings.HasPrefix(req.URL.Path, "/v2/x/y/blobs/"):
			r.blobHits.Add(1)
			content, ok := r.blobs[strings.TrimPrefix(req.URL.Path, "/v2/x/y/blobs/")]
			if !ok {
				http.NotFound(w, req)
				return
			}
			http.ServeContent(w, req, "", time.Time{}, bytes.NewReader(content))
		default:
			http.NotFound(w, req)
		}
	}))
	t.Cleanup(r.httpServer.Close)

	r.serverURL = strings.TrimPrefix(r.httpServer.URL, "http://")
	return r
}

func (r *fakeRegistry) pull(t *testing.T) error {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	return PullModel(ctx, "http://"+r.serverURL+"/x/y:latest", &registryOptions{Insecure: true}, func(api.ProgressResponse) {})
}

func TestPullModelRejectsTraversalDigests(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("OLLAMA_MODELS", filepath.Join(tmp, "models"))
	t.Setenv("OLLAMA_NOPRUNE", "1")

	victim := filepath.Join(tmp, "victim")
	if err := os.WriteFile(victim, []byte("do not delete"), 0o644); err != nil {
		t.Fatal(err)
	}

	config := []byte(`{}`)
	payload := []byte("attacker controlled content")

	cases := map[string]*ManifestV2{
		"layer creates file": {
			Config: &Layer{Digest: sha256Digest(config), Size: int64(len(config))},
			Layers: []*Layer{{Digest: "../../traversal-test", Size: int64(len(payload))}},
		},
		"layer deletes existing file": {
			Config: &Layer{Digest: sha256Digest(config), Size: int64(len(config))},
			Layers: []*Layer{{Digest: "../../victim", Size: int64(len(payload))}},
		},
		"config traversal": {
			Config: &Layer{Digest: "sha256:../../../traversal-test", Size: int64(len(payload))},
		},
		"later layer traversal": {
			Config: &Layer{Digest: sha256Digest(config), Size: int64(len(config))},
			Layers: []*Layer{
				{Digest: sha256Digest(payload), Size: int64(len(payload))},
				{Digest: "../../traversal-test", Size: int64(len(payload))},
			},
		},
		"directory digest": {
			Config: &Layer{Digest: sha256Digest(config), Size: int64(len(config))},
			Layers: []*Layer{{Digest: "../../", Size: int64(len(payload))}},
		},
	}

	for name, manifest := range cases {
		t.Run(name, func(t *testing.T) {
			reg := newFakeRegistry(t, manifest, map[string][]byte{
				sha256Digest(config):  config,
				sha256Digest(payload): payload,
			})

			err := reg.pull(t)
			if !errors.Is(err, errInvalidDigestFormat) && (err == nil || !strings.Contains(err.Error(), errInvalidDigestFormat.Error())) {
				t.Fatalf("expected invalid digest error, got %v", err)
			}

			if hits := reg.blobHits.Load(); hits != 0 {
				t.Errorf("expected no blob requests before validation, got %d", hits)
			}

			if _, err := os.Stat(filepath.Join(tmp, "traversal-test")); !errors.Is(err, os.ErrNotExist) {
				t.Errorf("traversal target should not exist, stat err = %v", err)
			}

			if b, err := os.ReadFile(victim); err != nil || string(b) != "do not delete" {
				t.Errorf("victim file was modified: %q, %v", b, err)
			}
		})
	}
}

func TestPullModelVerifiesBlobBeforeRename(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", filepath.Join(t.TempDir(), "models"))
	t.Setenv("OLLAMA_NOPRUNE", "1")

	config := []byte(`{}`)
	expected := []byte("expected-content")
	served := []byte("attacker-content")

	digest := sha256Digest(expected)
	reg := newFakeRegistry(t, &ManifestV2{
		Config: &Layer{Digest: sha256Digest(config), Size: int64(len(config))},
		Layers: []*Layer{{Digest: digest, Size: int64(len(expected))}},
	}, map[string][]byte{
		sha256Digest(config): config,
		digest:               served,
	})

	if err := reg.pull(t); !errors.Is(err, errDigestMismatch) && (err == nil || !strings.Contains(err.Error(), errDigestMismatch.Error())) {
		t.Fatalf("expected digest mismatch, got %v", err)
	}

	fp, err := GetBlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}

	for _, p := range []string{fp, fp + "-partial"} {
		if _, err := os.Stat(p); !errors.Is(err, os.ErrNotExist) {
			t.Errorf("%s should not exist after digest mismatch, stat err = %v", p, err)
		}
	}
}

func TestPullModelSuccess(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", filepath.Join(t.TempDir(), "models"))
	t.Setenv("OLLAMA_NOPRUNE", "1")

	config := []byte(`{"model_format":"gguf"}`)
	layer := []byte("model weights")

	reg := newFakeRegistry(t, &ManifestV2{
		SchemaVersion: 2,
		Config:        &Layer{Digest: sha256Digest(config), Size: int64(len(config))},
		Layers:        []*Layer{{Digest: sha256Digest(layer), Size: int64(len(layer))}},
	}, map[string][]byte{
		sha256Digest(config): config,
		sha256Digest(layer):  layer,
	})

	if err := reg.pull(t); err != nil {
		t.Fatalf("pull failed: %v", err)
	}

	for digest, want := range map[string][]byte{sha256Digest(config): config, sha256Digest(layer): layer} {
		fp, err := GetBlobsPath(digest)
		if err != nil {
			t.Fatal(err)
		}
		got, err := os.ReadFile(fp)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("blob %s = %q, want %q", digest, got, want)
		}
	}
}

func TestPruneLayersRemovesPartialDownloads(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", filepath.Join(t.TempDir(), "models"))

	dir, err := GetBlobsPath("")
	if err != nil {
		t.Fatal(err)
	}

	unused := strings.Replace(sha256Digest([]byte("unused")), ":", "-", 1)
	partial := strings.Replace(sha256Digest([]byte("partial")), ":", "-", 1) + "-partial"
	removed := []string{unused, partial, partial + "-0", partial + "-12"}
	kept := []string{"unrelated", unused + "-partial-x", "sha256-abc-partial"}

	for _, name := range append(append([]string{}, removed...), kept...) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	if err := PruneLayers(); err != nil {
		t.Fatal(err)
	}

	for _, name := range removed {
		if _, err := os.Stat(filepath.Join(dir, name)); !errors.Is(err, os.ErrNotExist) {
			t.Errorf("%s should have been pruned, stat err = %v", name, err)
		}
	}

	for _, name := range kept {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Errorf("%s should not have been pruned: %v", name, err)
		}
	}
}
