package server

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
)

type registryChallenge struct {
	Realm   string
	Service string
	Scope   string
}

func (r registryChallenge) URL() (*url.URL, error) {
	redirectURL, err := url.Parse(r.Realm)
	if err != nil {
		return nil, err
	}

	values := redirectURL.Query()
	values.Add("service", r.Service)
	for _, s := range strings.Split(r.Scope, " ") {
		values.Add("scope", s)
	}

	values.Add("ts", strconv.FormatInt(time.Now().Unix(), 10))

	nonce, err := auth.NewNonce(rand.Reader, 16)
	if err != nil {
		return nil, err
	}

	values.Add("nonce", nonce)

	redirectURL.RawQuery = values.Encode()
	return redirectURL, nil
}

var errUntrustedRealm = errors.New("registry returned an untrusted authentication realm")

// defaultRealmHosts are token hosts that may issue tokens for a registry other than themselves.
var defaultRealmHosts = map[string][]string{
	DefaultRegistry: {"ollama.com"},
}

// maxAuthResponseSize bounds how much of a token response is read.
const maxAuthResponseSize = 1 << 20

var authClient = &http.Client{
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 10 {
			return errors.New("stopped after 10 redirects")
		}

		if !sameOrigin(req.URL, via[0].URL) {
			return errUntrustedRealm
		}

		return nil
	},
}

func effectivePort(u *url.URL) string {
	if port := u.Port(); port != "" {
		return port
	}

	switch u.Scheme {
	case "http":
		return "80"
	case "https":
		return "443"
	}

	return ""
}

func sameOrigin(a, b *url.URL) bool {
	return a.Scheme == b.Scheme && strings.EqualFold(a.Hostname(), b.Hostname()) && effectivePort(a) == effectivePort(b)
}

// sameHost reports whether realm points at the registry's own host. An upgrade
// from the registry's default http port to the default https port counts as the
// same host.
func sameHost(realm, registry *url.URL) bool {
	if !strings.EqualFold(realm.Hostname(), registry.Hostname()) {
		return false
	}

	realmPort, registryPort := effectivePort(realm), effectivePort(registry)
	if realmPort == registryPort {
		return true
	}

	return realm.Port() == "" && registry.Port() == "" && realm.Scheme == "https" && registry.Scheme == "http"
}

// trustedRealmHosts returns the hosts, other than the registry itself, that are
// allowed to serve tokens for registry. OLLAMA_REGISTRY_AUTH_HOSTS adds
// comma-separated host or host:port entries.
func trustedRealmHosts(registry *url.URL) []string {
	hosts := append([]string{}, defaultRealmHosts[strings.ToLower(registry.Hostname())]...)
	for _, h := range strings.Split(os.Getenv("OLLAMA_REGISTRY_AUTH_HOSTS"), ",") {
		if h = strings.TrimSpace(h); h != "" {
			hosts = append(hosts, h)
		}
	}

	return hosts
}

// validateRealm ensures the token realm advertised by a registry challenge
// cannot be used to make the server request arbitrary URLs. The realm must be
// the registry's own host, or an explicitly trusted https token host.
func validateRealm(realm, registry *url.URL) error {
	if realm.Opaque != "" || realm.User != nil || realm.Hostname() == "" {
		return errUntrustedRealm
	}

	switch realm.Scheme {
	case "https":
	case "http":
		// plain http is only acceptable when talking to an insecure registry on the same host
		if registry.Scheme != "http" || !sameHost(realm, registry) {
			return errUntrustedRealm
		}
	default:
		return errUntrustedRealm
	}

	if sameHost(realm, registry) {
		return nil
	}

	for _, h := range trustedRealmHosts(registry) {
		host, port, err := net.SplitHostPort(h)
		if err != nil {
			host, port = h, "443"
		}

		if strings.EqualFold(realm.Hostname(), host) && effectivePort(realm) == port {
			return nil
		}
	}

	return errUntrustedRealm
}

// getAuthorizationToken requests a bearer token from the realm in challenge.
// registryURL is the registry request that produced the challenge.
func getAuthorizationToken(ctx context.Context, challenge registryChallenge, registryURL *url.URL) (string, error) {
	redirectURL, err := challenge.URL()
	if err != nil {
		return "", errUntrustedRealm
	}

	if err := validateRealm(redirectURL, registryURL); err != nil {
		slog.Warn("refusing registry authentication realm", "registry", registryURL.Host, "realm", redirectURL.Redacted())
		return "", err
	}

	sha256sum := sha256.Sum256(nil)
	data := []byte(fmt.Sprintf("%s,%s,%s", http.MethodGet, redirectURL.String(), base64.StdEncoding.EncodeToString([]byte(hex.EncodeToString(sha256sum[:])))))

	headers := make(http.Header)
	signature, err := auth.Sign(ctx, data)
	if err != nil {
		return "", err
	}

	headers.Add("Authorization", signature)

	response, err := makeRequestWithClient(ctx, authClient, http.MethodGet, redirectURL, headers, nil, nil)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()

	body, err := io.ReadAll(io.LimitReader(response.Body, maxAuthResponseSize))
	if err != nil {
		return "", fmt.Errorf("registry authentication failed: %s", statusText(response.StatusCode))
	}

	if response.StatusCode >= http.StatusBadRequest {
		slog.Debug("registry authentication failed", "status", response.StatusCode, "body", truncateForLog(body))
		return "", fmt.Errorf("registry authentication failed: %s", statusText(response.StatusCode))
	}

	var token api.TokenResponse
	if err := json.Unmarshal(body, &token); err != nil {
		slog.Debug("invalid registry token response", "error", err)
		return "", errors.New("registry authentication failed: invalid token response")
	}

	return token.Token, nil
}
