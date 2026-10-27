package server

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/pem"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"golang.org/x/crypto/ssh"
)

const realmMarker = "INTERNAL-SECRET-MARKER"

func mustParseURL(t *testing.T, s string) *url.URL {
	t.Helper()
	u, err := url.Parse(s)
	if err != nil {
		t.Fatal(err)
	}
	return u
}

func setupSigningKey(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)

	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	block, err := ssh.MarshalPrivateKey(priv, "")
	if err != nil {
		t.Fatal(err)
	}

	if err := os.MkdirAll(filepath.Join(home, ".ollama"), 0o755); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(home, ".ollama", "id_ed25519"), pem.EncodeToMemory(block), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestValidateRealm(t *testing.T) {
	t.Setenv("OLLAMA_REGISTRY_AUTH_HOSTS", "auth.example.com,tokens.example.com:8443")

	cases := []struct {
		realm, registry string
		ok              bool
	}{
		{"https://ollama.com:443/api/auth/token", "https://registry.ollama.ai/v2/library/llama3/manifests/latest", true},
		{"https://ollama.com/api/auth/token", "https://registry.ollama.ai/v2/x/y/manifests/latest", true},
		{"https://registry.example.com/token", "https://registry.example.com/v2/x/y/manifests/latest", true},
		{"https://REGISTRY.example.com:443/token", "https://registry.example.com/v2/x/y/manifests/latest", true},
		{"http://registry.example.com:5000/token", "http://registry.example.com:5000/v2/x/y/manifests/latest", true},
		{"https://registry.example.com/token", "http://registry.example.com/v2/x/y/manifests/latest", true},
		{"https://auth.example.com/token", "https://registry.example.com/v2/x/y/manifests/latest", true},
		{"https://tokens.example.com:8443/token", "https://registry.example.com/v2/x/y/manifests/latest", true},

		{"http://10.0.0.5:8500/v1/kv/config", "http://mallory.example/v2/x/y/manifests/latest", false},
		{"https://10.0.0.5:8500/v1/kv/config", "https://mallory.example/v2/x/y/manifests/latest", false},
		{"http://169.254.169.254/latest/meta-data/", "https://mallory.example/v2/x/y/manifests/latest", false},
		{"https://ollama.com/api/auth/token", "https://mallory.example/v2/x/y/manifests/latest", false},
		{"http://ollama.com/api/auth/token", "https://registry.ollama.ai/v2/x/y/manifests/latest", false},
		{"http://registry.example.com/token", "https://registry.example.com/v2/x/y/manifests/latest", false},
		{"https://registry.example.com:8500/token", "https://registry.example.com/v2/x/y/manifests/latest", false},
		{"https://tokens.example.com/token", "https://registry.example.com/v2/x/y/manifests/latest", false},
		{"https://user:pass@registry.example.com/token", "https://registry.example.com/v2/x/y/manifests/latest", false},
		{"file:///etc/passwd", "https://registry.example.com/v2/x/y/manifests/latest", false},
		{"gopher://registry.example.com/token", "https://registry.example.com/v2/x/y/manifests/latest", false},
		{"/token", "https://registry.example.com/v2/x/y/manifests/latest", false},
		{"", "https://registry.example.com/v2/x/y/manifests/latest", false},
	}

	for _, tt := range cases {
		t.Run(tt.realm+" <- "+tt.registry, func(t *testing.T) {
			err := validateRealm(mustParseURL(t, tt.realm), mustParseURL(t, tt.registry))
			if tt.ok && err != nil {
				t.Fatalf("expected realm to be allowed, got %v", err)
			}
			if !tt.ok && !errors.Is(err, errUntrustedRealm) {
				t.Fatalf("expected errUntrustedRealm, got %v", err)
			}
		})
	}
}

func TestPullRefusesUntrustedRealm(t *testing.T) {
	setupSigningKey(t)

	var internalHits atomic.Int32
	internal := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		internalHits.Add(1)
		http.Error(w, realmMarker, http.StatusForbidden)
	}))
	defer internal.Close()

	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("WWW-Authenticate", fmt.Sprintf(`Bearer realm="%s/anything",service="x",scope="repository:x/y:pull"`, internal.URL))
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer registry.Close()

	requestURL := mustParseURL(t, registry.URL+"/v2/x/y/manifests/latest")
	_, err := makeRequestWithRetry(context.Background(), http.MethodGet, requestURL, nil, nil, &registryOptions{Insecure: true})
	if !errors.Is(err, errUntrustedRealm) {
		t.Fatalf("expected errUntrustedRealm, got %v", err)
	}

	if strings.Contains(err.Error(), realmMarker) {
		t.Fatalf("error leaks realm response body: %v", err)
	}

	if n := internalHits.Load(); n != 0 {
		t.Fatalf("server requested untrusted realm %d times", n)
	}
}

func TestAuthErrorBodyNotReturned(t *testing.T) {
	setupSigningKey(t)

	var tokenHits atomic.Int32
	var registry *httptest.Server
	registry = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/token" {
			tokenHits.Add(1)
			http.Error(w, realmMarker, http.StatusForbidden)
			return
		}
		w.Header().Set("WWW-Authenticate", fmt.Sprintf(`Bearer realm="%s/token",service="x",scope="repository:x/y:pull"`, registry.URL))
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer registry.Close()

	requestURL := mustParseURL(t, registry.URL+"/v2/x/y/manifests/latest")
	_, err := makeRequestWithRetry(context.Background(), http.MethodGet, requestURL, nil, nil, &registryOptions{Insecure: true})
	if err == nil {
		t.Fatal("expected error")
	}

	if tokenHits.Load() != 1 {
		t.Fatalf("expected same-host realm to be requested once, got %d", tokenHits.Load())
	}

	if strings.Contains(err.Error(), realmMarker) {
		t.Fatalf("error leaks token response body: %v", err)
	}

	if !strings.Contains(err.Error(), "403 Forbidden") {
		t.Fatalf("expected status in error, got %v", err)
	}
}

func TestAuthRedirectToOtherHostRefused(t *testing.T) {
	setupSigningKey(t)

	var internalHits atomic.Int32
	internal := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		internalHits.Add(1)
		http.Error(w, realmMarker, http.StatusForbidden)
	}))
	defer internal.Close()

	var registry *httptest.Server
	registry = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/token" {
			http.Redirect(w, r, internal.URL+"/anything", http.StatusFound)
			return
		}
		w.Header().Set("WWW-Authenticate", fmt.Sprintf(`Bearer realm="%s/token"`, registry.URL))
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer registry.Close()

	requestURL := mustParseURL(t, registry.URL+"/v2/x/y/manifests/latest")
	_, err := makeRequestWithRetry(context.Background(), http.MethodGet, requestURL, nil, nil, &registryOptions{Insecure: true})
	if !errors.Is(err, errUntrustedRealm) {
		t.Fatalf("expected errUntrustedRealm, got %v", err)
	}

	if internalHits.Load() != 0 {
		t.Fatalf("token request followed cross-host redirect")
	}
}

func TestRedirectedChallengeUsesRespondingHost(t *testing.T) {
	setupSigningKey(t)

	var tokenHits atomic.Int32
	var storage *httptest.Server
	storage = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/token":
			tokenHits.Add(1)
			fmt.Fprint(w, `{"token":"storage-token"}`)
		case r.Header.Get("Authorization") == "Bearer storage-token":
			w.WriteHeader(http.StatusOK)
		default:
			w.Header().Set("WWW-Authenticate", fmt.Sprintf(`Bearer realm="%s/token"`, storage.URL))
			w.WriteHeader(http.StatusUnauthorized)
		}
	}))
	defer storage.Close()

	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, storage.URL+"/blob", http.StatusTemporaryRedirect)
	}))
	defer registry.Close()

	requestURL := mustParseURL(t, registry.URL+"/v2/x/y/blobs/sha256:abc")
	resp, err := makeRequestWithRetry(context.Background(), http.MethodGet, requestURL, nil, nil, &registryOptions{Insecure: true})
	if err != nil {
		t.Fatalf("expected redirected challenge to succeed, got %v", err)
	}
	resp.Body.Close()

	if tokenHits.Load() != 1 {
		t.Fatalf("expected one token request, got %d", tokenHits.Load())
	}
}

func TestRedirectedUploadUnauthorizedDoesNotPanic(t *testing.T) {
	setupSigningKey(t)

	var tokenHits atomic.Int32
	var storage *httptest.Server
	storage = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/token" {
			tokenHits.Add(1)
			fmt.Fprint(w, `{"token":"t"}`)
			return
		}
		w.Header().Set("WWW-Authenticate", fmt.Sprintf(`Bearer realm="%s/token"`, storage.URL))
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer storage.Close()

	f, err := os.CreateTemp(t.TempDir(), "blob")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.WriteString("data"); err != nil {
		t.Fatal(err)
	}

	b := &blobUpload{Layer: &Layer{Digest: "sha256:abc"}, file: f}
	part := &blobUploadPart{Size: 4}
	err = b.uploadPart(context.Background(), http.MethodPut, mustParseURL(t, storage.URL+"/upload"), part, nil)
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("expected 401 error, got %v", err)
	}

	if tokenHits.Load() != 0 {
		t.Fatalf("redirected upload requested a token")
	}
}

func TestRedactURL(t *testing.T) {
	u := mustParseURL(t, "https://user:pass@storage.example/blob/sha256?X-Amz-Signature=secret&X-Amz-Credential=key#frag")
	if got, want := redactURL(u), "https://storage.example/blob/sha256"; got != want {
		t.Fatalf("redactURL = %q, want %q", got, want)
	}
}

func TestRegistryErrorBody(t *testing.T) {
	cases := []struct {
		name        string
		contentType string
		body        string
		want        string
		notWant     string
	}{
		{"plain text", "text/plain", realmMarker, "registry responded with 403 Forbidden", realmMarker},
		{"html", "text/html", "<html>" + realmMarker + "</html>", "registry responded with 403 Forbidden", realmMarker},
		{"unrelated json", "application/json", `{"secret":"` + realmMarker + `"}`, "registry responded with 403 Forbidden", realmMarker},
		{"distribution errors", "application/json", `{"errors":[{"code":"DENIED","message":"requested access to the resource is denied"}]}`, "403: requested access to the resource is denied", ""},
		{"distribution code only", "application/json", `{"errors":[{"code":"DENIED"}]}`, "403: DENIED", ""},
		{"ollama error", "application/json", `{"error":"model not found"}`, "403: model not found", ""},
		{"control chars", "application/json", `{"error":"line1\nline2\u001b[31m"}`, "403: line1 line2 [31m", "\n"},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", tt.contentType)
				w.WriteHeader(http.StatusForbidden)
				fmt.Fprint(w, tt.body)
			}))
			defer srv.Close()

			requestURL := mustParseURL(t, srv.URL+"/v2/x/y/manifests/latest")
			_, err := makeRequestWithRetry(context.Background(), http.MethodGet, requestURL, nil, nil, &registryOptions{})
			if err == nil {
				t.Fatal("expected error")
			}

			if err.Error() != tt.want {
				t.Fatalf("got %q, want %q", err.Error(), tt.want)
			}

			if tt.notWant != "" && strings.Contains(err.Error(), tt.notWant) {
				t.Fatalf("error leaks %q: %v", tt.notWant, err)
			}
		})
	}

	long := strings.Repeat("a", 1000)
	msg := sanitizeRegistryMessage(long)
	if len(msg) != maxRegistryErrorLength+len("...") {
		t.Fatalf("expected message to be truncated, got length %d", len(msg))
	}
}

func TestIsLocalHost(t *testing.T) {
	local := []string{"localhost", "LOCALHOST.", "foo.localhost", "127.0.0.1", "::1", "10.0.0.5", "172.16.0.1", "192.168.1.1", "169.254.169.254", "fe80::1", "fd00::1", "0.0.0.0", "::ffff:127.0.0.1"}
	for _, h := range local {
		if !isLocalHost(h) {
			t.Errorf("expected %q to be local", h)
		}
	}

	remote := []string{"registry.ollama.ai", "ollama.com", "8.8.8.8", "2606:4700::1", "example.com"}
	for _, h := range remote {
		if isLocalHost(h) {
			t.Errorf("expected %q to be remote", h)
		}
	}
}
