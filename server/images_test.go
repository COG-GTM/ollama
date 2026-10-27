package server

import (
	"archive/zip"
	"bytes"
	"compress/flate"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

type zipEntry struct {
	name string
	mode fs.FileMode
	body string
}

func buildZip(t *testing.T, entries ...zipEntry) []byte {
	t.Helper()

	var buf bytes.Buffer
	w := zip.NewWriter(&buf)
	for _, e := range entries {
		h := &zip.FileHeader{Name: e.name, Method: zip.Deflate}
		if e.mode != 0 {
			h.SetMode(e.mode)
		}

		f, err := w.CreateHeader(h)
		if err != nil {
			t.Fatal(err)
		}

		if _, err := f.Write([]byte(e.body)); err != nil {
			t.Fatal(err)
		}
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	return buf.Bytes()
}

func openZip(t *testing.T, b []byte) *zip.Reader {
	t.Helper()

	r, err := zip.NewReader(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}

	return r
}

func TestExtractZipRejectsUnsafeEntries(t *testing.T) {
	cases := map[string]zipEntry{
		"parent traversal":    {name: "../../../escaped.txt", body: "pwned"},
		"nested traversal":    {name: "a/../../escaped.txt", body: "pwned"},
		"absolute path":       {name: "/tmp/escaped.txt", body: "pwned"},
		"backslash traversal": {name: `..\escaped.txt`, body: "pwned"},
		"dot dot only":        {name: "..", body: "pwned"},
		"symlink":             {name: "link", mode: fs.ModeSymlink | 0o777, body: "/etc/passwd"},
	}

	for name, entry := range cases {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			dir := filepath.Join(root, "extract")
			if err := os.Mkdir(dir, 0o755); err != nil {
				t.Fatal(err)
			}

			r := openZip(t, buildZip(t, zipEntry{name: "config.json", body: "{}"}, entry))
			err := extractZip(r, dir)
			if !errors.Is(err, errUnsafeZipEntry) {
				t.Fatalf("expected errUnsafeZipEntry, got %v", err)
			}

			if _, err := os.Stat(filepath.Join(root, "escaped.txt")); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("entry escaped extraction dir: %v", err)
			}

			if _, err := os.Lstat(filepath.Join(dir, "link")); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("symlink entry was written: %v", err)
			}
		})
	}
}

func TestExtractZipIgnoresArchiveModes(t *testing.T) {
	dir := t.TempDir()
	r := openZip(t, buildZip(t,
		zipEntry{name: "sub/", mode: fs.ModeDir | 0o777},
		zipEntry{name: "sub/deeper/model.safetensors", mode: 0o4777, body: "weights"},
		zipEntry{name: "config.json", mode: 0o777, body: "{}"},
	))

	if err := extractZip(r, dir); err != nil {
		t.Fatal(err)
	}

	for name, want := range map[string]string{
		"config.json":                  "{}",
		"sub/deeper/model.safetensors": "weights",
	} {
		p := filepath.Join(dir, filepath.FromSlash(name))
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}

		if string(b) != want {
			t.Errorf("%s: got %q, want %q", name, b, want)
		}

		fi, err := os.Stat(p)
		if err != nil {
			t.Fatal(err)
		}

		if fi.Mode()&(fs.ModeSetuid|0o111) != 0 {
			t.Errorf("%s: archive mode bits applied: %v", name, fi.Mode())
		}
	}
}

func TestExtractZipRejectsSizeMismatch(t *testing.T) {
	var compressed bytes.Buffer
	fw, err := flate.NewWriter(&compressed, flate.BestCompression)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := fw.Write(bytes.Repeat([]byte("A"), 1<<20)); err != nil {
		t.Fatal(err)
	}

	if err := fw.Close(); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	w := zip.NewWriter(&buf)
	f, err := w.CreateRaw(&zip.FileHeader{
		Name:               "config.json",
		Method:             zip.Deflate,
		CompressedSize64:   uint64(compressed.Len()),
		UncompressedSize64: 4,
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := f.Write(compressed.Bytes()); err != nil {
		t.Fatal(err)
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	if err := extractZip(openZip(t, buf.Bytes()), dir); err == nil {
		t.Fatal("expected error for entry larger than its header size")
	}

	if fi, err := os.Stat(filepath.Join(dir, "config.json")); err == nil && fi.Size() > 5 {
		t.Fatalf("wrote %d bytes past declared size", fi.Size())
	}
}

func TestCreateFromUploadedZipSlipBlob(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	tmp := t.TempDir()
	t.Setenv("TMPDIR", tmp)

	srv := httptest.NewServer((&Server{}).GenerateRoutes())
	t.Cleanup(srv.Close)

	blob := buildZip(t, zipEntry{name: "../zipslip-test", mode: 0o755, body: "pwned"})
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(blob))

	resp, err := http.Post(srv.URL+"/api/blobs/"+digest, "application/octet-stream", bytes.NewReader(blob))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("blob upload: status %d", resp.StatusCode)
	}

	body, err := json.Marshal(api.CreateRequest{Name: "zipslip", Modelfile: "FROM @" + digest})
	if err != nil {
		t.Fatal(err)
	}

	resp, err = http.Post(srv.URL+"/api/create", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var sawError bool
	dec := json.NewDecoder(resp.Body)
	for {
		var msg map[string]any
		if err := dec.Decode(&msg); err != nil {
			break
		}

		if e, ok := msg["error"].(string); ok {
			sawError = true
			if !strings.Contains(e, errUnsafeZipEntry.Error()) {
				t.Errorf("unexpected error: %s", e)
			}
		}
	}

	if !sawError {
		t.Fatal("expected /api/create to fail on unsafe zip entry")
	}

	if _, err := os.Stat(filepath.Join(tmp, "zipslip-test")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("zip entry escaped temp dir: %v", err)
	}
}
