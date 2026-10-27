package server

import (
	"archive/zip"
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func newTestZip(t *testing.T, entries map[string]string) *zip.Reader {
	t.Helper()

	var buf bytes.Buffer
	w := zip.NewWriter(&buf)
	for name, content := range entries {
		hdr := &zip.FileHeader{Name: name, Method: zip.Store}
		hdr.SetMode(0o777)
		f, err := w.CreateHeader(hdr)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := zip.NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestExtractZipRejectsUnsafeEntries(t *testing.T) {
	cases := []string{
		"../escape",
		"../../../../tmp/zipslip-test",
		"a/../../escape",
		"/abs/path",
		"..",
		`..\escape`,
	}

	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			parent := t.TempDir()
			dir := filepath.Join(parent, "extract")
			if err := os.Mkdir(dir, 0o755); err != nil {
				t.Fatal(err)
			}

			err := extractZip(newTestZip(t, map[string]string{name: "pwned"}), dir)
			if !errors.Is(err, errUnsafeZipEntry) {
				t.Fatalf("expected errUnsafeZipEntry, got %v", err)
			}

			if _, err := os.Stat(filepath.Join(parent, "escape")); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("file escaped extraction dir: %v", err)
			}
		})
	}
}

func TestExtractZip(t *testing.T) {
	dir := t.TempDir()
	r := newTestZip(t, map[string]string{
		"config.json":         "{}",
		"sub/":                "",
		"sub/tokenizer.model": "tok",
	})

	if err := extractZip(r, dir); err != nil {
		t.Fatal(err)
	}

	for name, want := range map[string]string{"config.json": "{}", "sub/tokenizer.model": "tok"} {
		p := filepath.Join(dir, filepath.FromSlash(name))
		got, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != want {
			t.Errorf("%s: got %q, want %q", name, got, want)
		}

		fi, err := os.Stat(p)
		if err != nil {
			t.Fatal(err)
		}
		if fi.Mode().Perm()&0o111 != 0 {
			t.Errorf("%s: archive mode bits should be ignored, got %v", name, fi.Mode())
		}
	}
}
