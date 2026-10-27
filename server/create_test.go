package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// createTestBlob copies the file at path into the blob store and returns its digest.
func createTestBlob(t *testing.T, path string) string {
	t.Helper()

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	layer, err := NewLayer(f, "")
	if err != nil {
		t.Fatal(err)
	}

	if _, err := layer.Commit(); err != nil {
		t.Fatal(err)
	}

	return layer.Digest
}

func postCreate(t *testing.T, srv *httptest.Server, req api.CreateRequest) (int, string) {
	t.Helper()

	stream := false
	req.Stream = &stream
	bts, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := srv.Client().Post(srv.URL+"/api/create", "application/json", bytes.NewReader(bts))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var body struct {
		Error string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}

	return resp.StatusCode, body.Error
}

func TestCreateModelDoesNotProbeServerPaths(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	srv := httptest.NewServer((&Server{}).GenerateRoutes())
	t.Cleanup(srv.Close)

	dir := t.TempDir()
	// a minimal GGUF v3 file: magic, version, 0 tensors, 0 kv pairs
	gguf := append([]byte("GGUF"), 3, 0, 0, 0)
	gguf = append(gguf, make([]byte, 16)...)

	readable := filepath.Join(dir, "readable")
	if err := os.WriteFile(readable, gguf, 0o644); err != nil {
		t.Fatal(err)
	}

	unreadable := filepath.Join(dir, "unreadable")
	if err := os.WriteFile(unreadable, gguf, 0o000); err != nil {
		t.Fatal(err)
	}

	paths := []string{
		filepath.Join(dir, "missing"),
		readable,
		unreadable,
		dir,
		"/etc/shadow",
		"/etc/passwd",
	}

	t.Run("path", func(t *testing.T) {
		for _, p := range paths {
			code, msg := postCreate(t, srv, api.CreateRequest{Name: "x", Path: p})
			if code != http.StatusBadRequest || msg != "path is not supported, send the Modelfile contents in modelfile" {
				t.Errorf("path %q: got %d %q", p, code, msg)
			}

			if strings.Contains(msg, p) {
				t.Errorf("path %q echoed in error %q", p, msg)
			}
		}
	})

	for _, instruction := range []string{"FROM", "ADAPTER"} {
		t.Run(instruction, func(t *testing.T) {
			args := append(paths, "../../../etc/passwd", "file:///etc/passwd")
			if instruction == "ADAPTER" {
				args = append(args, "~/.ssh/id_rsa", "~")
			}

			var want string
			for _, p := range args {
				var modelfile string
				if instruction == "ADAPTER" {
					modelfile = "FROM @" + createTestBlob(t, readable) + "\nADAPTER " + p
				} else {
					modelfile = "FROM " + p
				}

				code, msg := postCreate(t, srv, api.CreateRequest{Name: "x", Modelfile: modelfile})
				if code != http.StatusInternalServerError {
					t.Errorf("%s %q: got status %d", instruction, p, code)
				}

				// the error must not depend on whether the path exists or is readable
				got := strings.ReplaceAll(msg, p, "<arg>")
				if want == "" {
					want = got
				} else if got != want {
					t.Errorf("%s %q: error %q differs from %q", instruction, p, got, want)
				}

				for _, leak := range []string{"no such file", "permission denied", "is a directory", "gguf"} {
					if strings.Contains(strings.ToLower(msg), leak) {
						t.Errorf("%s %q: error %q leaks file state", instruction, p, msg)
					}
				}
			}
		})
	}
}

func TestBlobPath(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	blobs, err := GetBlobsPath("")
	if err != nil {
		t.Fatal(err)
	}

	digest := "sha256:" + strings.Repeat("a", 64)
	want := filepath.Join(blobs, "sha256-"+strings.Repeat("a", 64))

	cases := []struct {
		arg     string
		want    string
		isBlob  bool
		wantErr bool
	}{
		{arg: "@" + digest, want: want, isBlob: true},
		{arg: want, want: want, isBlob: true},
		{arg: "llama3", isBlob: false},
		{arg: "/etc/passwd", isBlob: false},
		{arg: "./model.gguf", isBlob: false},
		{arg: "~/model.gguf", isBlob: false},
		{arg: filepath.Join(blobs, "..", "manifests"), isBlob: false},
		{arg: "@../../etc/passwd", isBlob: true, wantErr: true},
		{arg: "@sha256:1234", isBlob: true, wantErr: true},
		{arg: filepath.Join(blobs, "not-a-digest"), isBlob: true, wantErr: true},
	}

	for _, tt := range cases {
		got, isBlob, err := blobPath(tt.arg)
		if (err != nil) != tt.wantErr || isBlob != tt.isBlob || got != tt.want {
			t.Errorf("blobPath(%q) = %q, %v, %v; want %q, %v, err=%v", tt.arg, got, isBlob, err, tt.want, tt.isBlob, tt.wantErr)
		}
	}
}

func TestModelPathValidateRejectsTraversal(t *testing.T) {
	for _, name := range []string{"/etc/passwd", "../../etc:passwd", "a/../b", "./x", "a//b", `a\b`} {
		if err := ParseModelPath(name).Validate(); err == nil {
			t.Errorf("Validate(%q) = nil, want error", name)
		}
	}

	for _, name := range []string{"llama3", "llama3:8b", "ns/repo:tag", "example.com:5000/ns/repo:tag"} {
		if err := ParseModelPath(name).Validate(); err != nil {
			t.Errorf("Validate(%q) = %v, want nil", name, err)
		}
	}
}

func TestBlobHandlersRejectInvalidDigest(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	srv := httptest.NewServer((&Server{}).GenerateRoutes())
	t.Cleanup(srv.Close)

	for _, method := range []string{http.MethodHead, http.MethodPost} {
		req, err := http.NewRequest(method, srv.URL+"/api/blobs/..", nil)
		if err != nil {
			t.Fatal(err)
		}

		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()

		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("%s /api/blobs/..: got %d, want 400", method, resp.StatusCode)
		}
	}
}
