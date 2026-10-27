package server

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"
)

func partialBlobs(t *testing.T) []string {
	t.Helper()

	blobs, err := GetBlobsPath("")
	if err != nil {
		t.Fatal(err)
	}

	matches, err := filepath.Glob(filepath.Join(blobs, "sha256-*-partial"))
	if err != nil {
		t.Fatal(err)
	}

	return matches
}

func postBlob(t *testing.T, digest string, body []byte) int {
	t.Helper()

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{}
	r.POST("/api/blobs/:digest", s.CreateBlobHandler)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/blobs/"+digest, bytes.NewReader(body))
	r.ServeHTTP(w, req)
	return w.Code
}

func TestCreateBlobDigestMismatchRemovesTemp(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	digest := fmt.Sprintf("sha256:%064x", 0)
	if code := postBlob(t, digest, bytes.Repeat([]byte("a"), 1<<20)); code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, code)
	}

	if partials := partialBlobs(t); len(partials) != 0 {
		t.Fatalf("expected no partial blobs, found %v", partials)
	}

	blob, err := GetBlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(blob); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("expected no blob for mismatched digest, got %v", err)
	}
}

func TestCreateBlobCommitsMatchingDigest(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	body := []byte("hello blob")
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(body))
	if code := postBlob(t, digest, body); code != http.StatusCreated {
		t.Fatalf("expected status %d, got %d", http.StatusCreated, code)
	}

	if partials := partialBlobs(t); len(partials) != 0 {
		t.Fatalf("expected no partial blobs, found %v", partials)
	}

	blob, err := GetBlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(blob)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, body) {
		t.Fatalf("expected blob contents %q, got %q", body, got)
	}
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) {
	return 0, errors.New("read failed")
}

func TestNewLayerReadErrorRemovesTemp(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	if _, err := NewLayer(failingReader{}, ""); err == nil {
		t.Fatal("expected error")
	}

	if partials := partialBlobs(t); len(partials) != 0 {
		t.Fatalf("expected no partial blobs, found %v", partials)
	}
}

func TestLayerDiscard(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	layer, err := NewLayer(bytes.NewReader([]byte("data")), "")
	if err != nil {
		t.Fatal(err)
	}

	if partials := partialBlobs(t); len(partials) != 1 {
		t.Fatalf("expected one partial blob, found %v", partials)
	}

	if err := layer.Discard(); err != nil {
		t.Fatal(err)
	}
	if partials := partialBlobs(t); len(partials) != 0 {
		t.Fatalf("expected no partial blobs, found %v", partials)
	}

	if err := layer.Discard(); err != nil {
		t.Fatalf("expected second Discard to be a no-op, got %v", err)
	}
}
