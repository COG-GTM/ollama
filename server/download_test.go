package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

const testDigest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func TestGetValueTruncated(t *testing.T) {
	cases := map[string]string{
		`realm=`:  "",
		`realm="`: "",
		`realm=x`: "",
		`realm="https://registry.example/token",service="ollama"`: "https://registry.example/token",
	}

	for header, want := range cases {
		if got := getValue(header, "realm"); got != want {
			t.Errorf("getValue(%q) = %q, want %q", header, got, want)
		}
	}

	if c := parseRegistryChallenge("Bearer realm="); c.Realm != "" {
		t.Errorf("expected empty realm, got %q", c.Realm)
	}
}

func TestValidateManifest(t *testing.T) {
	layer := &Layer{Digest: testDigest}
	cases := map[string]*ManifestV2{
		"nil":          nil,
		"no config":    {Layers: []*Layer{layer}},
		"no layers":    {Config: layer},
		"null layer":   {Config: layer, Layers: []*Layer{nil}},
		"short digest": {Config: layer, Layers: []*Layer{{Digest: "sha256:abc"}}},
		"traversal":    {Config: &Layer{Digest: "../../../etc/passwd"}, Layers: []*Layer{layer}},
	}

	for name, m := range cases {
		t.Run(name, func(t *testing.T) {
			if err := validateManifest(m); !errors.Is(err, errInvalidManifest) {
				t.Fatalf("expected errInvalidManifest, got %v", err)
			}
		})
	}

	if err := validateManifest(&ManifestV2{Config: layer, Layers: []*Layer{layer}}); err != nil {
		t.Fatalf("expected valid manifest, got %v", err)
	}
}

func TestBlobDownloadPrepareRejectsBadContentLength(t *testing.T) {
	cases := map[string]string{
		"missing": "",
		"zero":    "0",
		"huge":    "9223372036854775807",
	}

	for name, contentLength := range cases {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if contentLength != "" {
					w.Header().Set("Content-Length", contentLength)
				}
				w.WriteHeader(http.StatusOK)
			}))
			defer srv.Close()

			u, _ := url.Parse(srv.URL + "/v2/library/m/blobs/" + testDigest)
			b := &blobDownload{Name: filepath.Join(t.TempDir(), "blob"), Digest: testDigest}

			err := b.Prepare(context.Background(), u, &registryOptions{Insecure: true})
			if !errors.Is(err, errInvalidBlobResponse) {
				t.Fatalf("expected errInvalidBlobResponse, got %v", err)
			}

			if len(b.Parts) != 0 {
				t.Fatalf("expected no parts, got %d", len(b.Parts))
			}
		})
	}
}

func TestBlobDownloadPrepareDiscardsInvalidParts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1024")
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	name := filepath.Join(t.TempDir(), "blob")
	tampered := &blobDownloadPart{N: 0, Offset: 0, Size: math.MaxInt64}
	b := &blobDownload{Name: name, Digest: testDigest}
	tampered.blobDownload = b
	if err := b.writePart(tampered.Name(), tampered); err != nil {
		t.Fatal(err)
	}

	u, _ := url.Parse(srv.URL + "/v2/library/m/blobs/" + testDigest)
	b = &blobDownload{Name: name, Digest: testDigest}
	if err := b.Prepare(context.Background(), u, &registryOptions{Insecure: true}); err != nil {
		t.Fatal(err)
	}

	if b.Total != 1024 || len(b.Parts) != 1 || b.Parts[0].Size != 1024 {
		t.Fatalf("expected a fresh 1024 byte part, got total %d, parts %+v", b.Total, b.Parts)
	}
}

func TestBlobDownloadChunkLimitsBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusPartialContent)
		w.Write(bytes.Repeat([]byte("A"), 1<<20))
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	b := &blobDownload{Name: filepath.Join(t.TempDir(), "blob"), Digest: testDigest}
	part := &blobDownloadPart{blobDownload: b, Size: 16}

	var buf bytes.Buffer
	if err := b.downloadChunk(context.Background(), u, &buf, part, &registryOptions{Insecure: true}); err != nil {
		t.Fatal(err)
	}

	if buf.Len() != 16 || part.Completed != 16 {
		t.Fatalf("expected 16 bytes written, got %d (completed %d)", buf.Len(), part.Completed)
	}
}

func TestBlobDownloadChunkRejectsIgnoredRange(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write(bytes.Repeat([]byte("A"), 64))
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	b := &blobDownload{Name: filepath.Join(t.TempDir(), "blob"), Digest: testDigest}
	part := &blobDownloadPart{blobDownload: b, N: 1, Offset: 16, Size: 16}

	var buf bytes.Buffer
	err := b.downloadChunk(context.Background(), u, &buf, part, &registryOptions{Insecure: true})
	if !errors.Is(err, errInvalidBlobResponse) {
		t.Fatalf("expected errInvalidBlobResponse, got %v", err)
	}

	if buf.Len() != 0 {
		t.Fatalf("expected nothing written, got %d bytes", buf.Len())
	}
}

func TestPullModelMalformedRegistry(t *testing.T) {
	validManifest := func(digest string) string {
		return fmt.Sprintf(`{"schemaVersion":2,"config":{"digest":%q,"size":1},"layers":[{"digest":%q,"size":1}]}`, testDigest, digest)
	}

	cases := []struct {
		name     string
		manifest string
		headers  map[string]string
		status   int
	}{
		{name: "no config", manifest: fmt.Sprintf(`{"schemaVersion":2,"layers":[{"digest":%q}]}`, testDigest)},
		{name: "null manifest", manifest: `null`},
		{name: "short digest", manifest: validManifest("sha256:abc")},
		{name: "traversal digest", manifest: validManifest("sha256:../../../../tmp/x")},
		{name: "no content length", manifest: validManifest(testDigest)},
		{name: "huge content length", manifest: validManifest(testDigest), headers: map[string]string{"Content-Length": "9223372036854775807"}},
		{name: "empty realm", status: http.StatusUnauthorized, headers: map[string]string{"WWW-Authenticate": "Bearer realm="}},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			t.Setenv("OLLAMA_NOPRUNE", "1")

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tt.status == 0 && strings.Contains(r.URL.Path, "/manifests/") {
					w.Write([]byte(tt.manifest))
					return
				}

				for k, v := range tt.headers {
					w.Header().Set(k, v)
				}

				if tt.status != 0 {
					w.WriteHeader(tt.status)
					return
				}

				w.WriteHeader(http.StatusOK)
			}))
			defer srv.Close()

			name := strings.TrimPrefix(srv.URL, "http://") + "/library/m:latest"
			fn := func(api.ProgressResponse) {}

			var err error
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("PullModel panicked: %v", r)
					}
				}()
				err = PullModel(context.Background(), name, &registryOptions{Insecure: true}, fn)
			}()

			if err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}

func TestPullModelHandlerMalformedManifest(t *testing.T) {
	// a manifest without config previously dereferenced nil inside the
	// handler goroutine and killed the process
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_NOPRUNE", "1")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"schemaVersion":2}`))
	}))
	defer srv.Close()

	s := &Server{}
	router := s.GenerateRoutes()

	body, _ := json.Marshal(api.PullRequest{Model: strings.TrimPrefix(srv.URL, "http://") + "/library/m:latest", Insecure: true})
	req := httptest.NewRequest(http.MethodPost, "/api/pull", bytes.NewReader(body))
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if !strings.Contains(w.Body.String(), `"error"`) {
		t.Fatalf("expected error in response, got %s", w.Body.String())
	}
}
