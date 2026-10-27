package server

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"testing"
)

func mustParseURL(t *testing.T, s string) *url.URL {
	t.Helper()
	u, err := url.Parse(s)
	if err != nil {
		t.Fatal(err)
	}
	return u
}

func TestValidateRealm(t *testing.T) {
	cases := []struct {
		name     string
		realm    string
		registry string
		insecure bool
		ok       bool
	}{
		{"same host", "https://registry.example.com/token", "https://registry.example.com/v2/x/y/manifests/latest", false, true},
		{"same host case and default port", "https://Registry.Example.com:443/token", "https://registry.example.com/v2/", false, true},
		{"default registry delegates to ollama.com", "https://ollama.com/token", "https://registry.ollama.ai/v2/library/llama3/manifests/latest", false, true},
		{"default registry same host", "https://registry.ollama.ai/token", "https://registry.ollama.ai/v2/", false, true},
		{"foreign registry points at ollama.com", "https://ollama.com/token", "https://mallory.example/v2/x/y/manifests/latest", false, false},
		{"default registry points at foreign realm", "https://mallory.example/token", "https://registry.ollama.ai/v2/", false, false},
		{"subdomain is not the same host", "https://auth.registry.example.com/token", "https://registry.example.com/v2/", false, false},
		{"different port", "https://registry.example.com:8443/token", "https://registry.example.com/v2/", false, false},
		{"http realm rejected", "http://registry.example.com/token", "https://registry.example.com/v2/", false, false},
		{"http realm allowed for insecure registry", "http://localhost:5000/token", "http://localhost:5000/v2/", true, true},
		{"insecure does not relax host check", "http://ollama.com/token", "http://localhost:5000/v2/", true, false},
		{"userinfo rejected", "https://user:pass@registry.example.com/token", "https://registry.example.com/v2/", false, false},
		{"non-http scheme rejected", "file:///etc/passwd", "https://registry.example.com/v2/", false, false},
		{"empty realm rejected", "", "https://registry.example.com/v2/", false, false},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			err := validateRealm(mustParseURL(t, tt.realm), mustParseURL(t, tt.registry), tt.insecure)
			if tt.ok && err != nil {
				t.Fatalf("expected realm to be accepted, got %v", err)
			}
			if !tt.ok {
				if err == nil {
					t.Fatal("expected realm to be rejected")
				}
				if !errors.Is(err, errUntrustedRealm) {
					t.Fatalf("expected errUntrustedRealm, got %v", err)
				}
			}
		})
	}
}

func TestGetAuthorizationTokenRejectsMismatchedRealm(t *testing.T) {
	challenge := parseRegistryChallenge(`Bearer realm="https://ollama.com/token",service="ollama.com",scope="repository:acme/model:pull,push"`)
	if challenge.Realm != "https://ollama.com/token" {
		t.Fatalf("unexpected realm %q", challenge.Realm)
	}

	_, err := getAuthorizationToken(context.Background(), challenge, mustParseURL(t, "https://mallory.example/v2/x/y/manifests/latest"), false)
	if !errors.Is(err, errUntrustedRealm) {
		t.Fatalf("expected errUntrustedRealm, got %v", err)
	}
}

type recordingServer struct {
	*httptest.Server
	mu    sync.Mutex
	auths []string
}

func (s *recordingServer) record(r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.auths = append(s.auths, r.Header.Get("Authorization"))
}

func (s *recordingServer) authHeaders() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.auths...)
}

func testJWT(sub string) string {
	enc := base64.RawURLEncoding.EncodeToString
	return enc([]byte(`{"alg":"none"}`)) + "." + enc([]byte(fmt.Sprintf(`{"sub":%q}`, sub))) + ".sig"
}

func TestMakeRequestWithRetryForeignRealm(t *testing.T) {
	tokenServer := &recordingServer{}
	tokenServer.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tokenServer.record(r)
		fmt.Fprintf(w, `{"token":%q}`, testJWT("alice"))
	}))
	defer tokenServer.Close()

	registry := &recordingServer{}
	registry.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		registry.record(r)
		w.Header().Set("WWW-Authenticate", fmt.Sprintf(`Bearer realm="%s/token",service="ollama.com",scope="repository:acme/model:pull,push"`, tokenServer.URL))
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer registry.Close()

	regOpts := &registryOptions{Insecure: true}
	_, err := makeRequestWithRetry(context.Background(), http.MethodGet, mustParseURL(t, registry.URL+"/v2/x/y/manifests/latest"), nil, nil, regOpts)
	if !errors.Is(err, errUntrustedRealm) {
		t.Fatalf("expected errUntrustedRealm, got %v", err)
	}

	if got := tokenServer.authHeaders(); len(got) != 0 {
		t.Fatalf("token server must not be contacted, got %d requests", len(got))
	}

	for _, a := range registry.authHeaders() {
		if a != "" {
			t.Fatalf("registry received Authorization header %q", a)
		}
	}

	if regOpts.Token != "" {
		t.Fatalf("token must not be stored, got %q", regOpts.Token)
	}
}

func TestMakeRequestWithRetrySameHostRealm(t *testing.T) {
	token := testJWT("bob")

	var mu sync.Mutex
	var tokenAuth []string
	registry := &recordingServer{}
	registry.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/token" {
			mu.Lock()
			tokenAuth = append(tokenAuth, r.Header.Get("Authorization"))
			mu.Unlock()
			fmt.Fprintf(w, `{"token":%q}`, token)
			return
		}

		registry.record(r)
		if r.Header.Get("Authorization") != "Bearer "+token {
			w.Header().Set("WWW-Authenticate", fmt.Sprintf(`Bearer realm="http://%s/token",service="test",scope="repository:x/y:pull"`, r.Host))
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer registry.Close()

	regOpts := &registryOptions{Insecure: true}
	resp, err := makeRequestWithRetry(context.Background(), http.MethodGet, mustParseURL(t, registry.URL+"/v2/x/y/manifests/latest"), nil, nil, regOpts)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	mu.Lock()
	defer mu.Unlock()
	if len(tokenAuth) != 1 {
		t.Fatalf("expected one token request, got %d", len(tokenAuth))
	}
	if tokenAuth[0] != "" {
		t.Fatalf("token request to a non-Ollama registry must not be signed, got %q", tokenAuth[0])
	}

	if regOpts.TokenHost != canonicalHost(mustParseURL(t, registry.URL)) {
		t.Fatalf("token bound to %q, want %q", regOpts.TokenHost, registry.Listener.Addr().String())
	}
}

func TestMakeRequestTokenBoundToHost(t *testing.T) {
	newServer := func() *recordingServer {
		s := &recordingServer{}
		s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			s.record(r)
		}))
		return s
	}

	issuer := newServer()
	defer issuer.Close()
	other := newServer()
	defer other.Close()

	regOpts := &registryOptions{Token: "secret", TokenHost: canonicalHost(mustParseURL(t, issuer.URL))}

	for _, s := range []*recordingServer{issuer, other} {
		resp, err := makeRequest(context.Background(), http.MethodGet, mustParseURL(t, s.URL+"/v2/"), nil, nil, regOpts)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}

	if got := issuer.authHeaders(); len(got) != 1 || got[0] != "Bearer secret" {
		t.Fatalf("issuer should receive the token, got %q", got)
	}
	if got := other.authHeaders(); len(got) != 1 || got[0] != "" {
		t.Fatalf("other host must not receive the token, got %q", got)
	}

	unbound := &registryOptions{Token: "secret"}
	resp, err := makeRequest(context.Background(), http.MethodGet, mustParseURL(t, other.URL+"/v2/"), nil, nil, unbound)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if got := other.authHeaders(); got[len(got)-1] != "" {
		t.Fatalf("unbound token must not be sent, got %q", got[len(got)-1])
	}
}
