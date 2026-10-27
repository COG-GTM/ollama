package server

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

type ModelPath struct {
	ProtocolScheme string
	Registry       string
	Namespace      string
	Repository     string
	Tag            string
}

const (
	DefaultRegistry       = "registry.ollama.ai"
	DefaultNamespace      = "library"
	DefaultTag            = "latest"
	DefaultProtocolScheme = "https"
)

var (
	ErrInvalidImageFormat = errors.New("invalid image format")
	ErrInvalidProtocol    = errors.New("invalid protocol scheme")
	ErrInsecureProtocol   = errors.New("insecure protocol http")
)

func ParseModelPath(name string) ModelPath {
	mp := ModelPath{
		ProtocolScheme: DefaultProtocolScheme,
		Registry:       DefaultRegistry,
		Namespace:      DefaultNamespace,
		Repository:     "",
		Tag:            DefaultTag,
	}

	before, after, found := strings.Cut(name, "://")
	if found {
		mp.ProtocolScheme = before
		name = after
	}

	name = strings.ReplaceAll(name, string(os.PathSeparator), "/")
	parts := strings.Split(name, "/")
	switch len(parts) {
	case 3:
		mp.Registry = parts[0]
		mp.Namespace = parts[1]
		mp.Repository = parts[2]
	case 2:
		mp.Namespace = parts[0]
		mp.Repository = parts[1]
	case 1:
		mp.Repository = parts[0]
	}

	if repo, tag, found := strings.Cut(mp.Repository, ":"); found {
		mp.Repository = repo
		mp.Tag = tag
	}

	return mp
}

var errModelPathInvalid = errors.New("invalid model path")

func (mp ModelPath) Validate() error {
	if mp.Repository == "" {
		return fmt.Errorf("%w: model repository name is required", errModelPathInvalid)
	}

	if strings.Contains(mp.Tag, ":") {
		return fmt.Errorf("%w: ':' (colon) is not allowed in tag names", errModelPathInvalid)
	}

	for _, part := range []struct{ kind, value string }{
		{"registry", mp.Registry},
		{"namespace", mp.Namespace},
		{"repository", mp.Repository},
		{"tag", mp.Tag},
	} {
		if err := validatePathPart(part.kind, part.value); err != nil {
			return err
		}
	}

	return nil
}

// validatePathPart rejects model path parts that cannot safely be used as a
// single path element under the manifests directory.
func validatePathPart(kind, s string) error {
	switch {
	case s == "":
		return fmt.Errorf("%w: %s name is required", errModelPathInvalid, kind)
	case s == "." || s == "..":
		return fmt.Errorf("%w: %q is not allowed as a %s name", errModelPathInvalid, s, kind)
	case strings.ContainsAny(s, "/\\\x00"):
		return fmt.Errorf("%w: path separators are not allowed in %s names", errModelPathInvalid, kind)
	}

	return nil
}

func (mp ModelPath) GetNamespaceRepository() string {
	return fmt.Sprintf("%s/%s", mp.Namespace, mp.Repository)
}

func (mp ModelPath) GetFullTagname() string {
	return fmt.Sprintf("%s/%s/%s:%s", mp.Registry, mp.Namespace, mp.Repository, mp.Tag)
}

func (mp ModelPath) GetShortTagname() string {
	if mp.Registry == DefaultRegistry {
		if mp.Namespace == DefaultNamespace {
			return fmt.Sprintf("%s:%s", mp.Repository, mp.Tag)
		}
		return fmt.Sprintf("%s/%s:%s", mp.Namespace, mp.Repository, mp.Tag)
	}
	return fmt.Sprintf("%s/%s/%s:%s", mp.Registry, mp.Namespace, mp.Repository, mp.Tag)
}

// modelsDir returns the value of the OLLAMA_MODELS environment variable or the user's home directory if OLLAMA_MODELS is not set.
// The models directory is where Ollama stores its model files and manifests.
func modelsDir() (string, error) {
	if models, exists := os.LookupEnv("OLLAMA_MODELS"); exists {
		return models, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "models"), nil
}

// GetManifestPath returns the path to the manifest file for the given model path, it is up to the caller to create the directory if it does not exist.
func (mp ModelPath) GetManifestPath() (string, error) {
	if err := mp.Validate(); err != nil {
		return "", err
	}

	dir, err := modelsDir()
	if err != nil {
		return "", err
	}

	manifests := filepath.Join(dir, "manifests")
	fp := filepath.Join(manifests, mp.Registry, mp.Namespace, mp.Repository, mp.Tag)
	rel, err := filepath.Rel(manifests, fp)
	if err != nil || !filepath.IsLocal(rel) || rel == "." {
		return "", fmt.Errorf("%w: manifest path escapes the manifests directory", errModelPathInvalid)
	}

	return fp, nil
}

func (mp ModelPath) BaseURL() *url.URL {
	return &url.URL{
		Scheme: mp.ProtocolScheme,
		Host:   mp.Registry,
	}
}

func GetManifestPath() (string, error) {
	dir, err := modelsDir()
	if err != nil {
		return "", err
	}

	path := filepath.Join(dir, "manifests")
	if err := os.MkdirAll(path, 0o755); err != nil {
		return "", err
	}

	return path, nil
}

func GetBlobsPath(digest string) (string, error) {
	dir, err := modelsDir()
	if err != nil {
		return "", err
	}

	digest = strings.ReplaceAll(digest, ":", "-")
	path := filepath.Join(dir, "blobs", digest)
	dirPath := filepath.Dir(path)
	if digest == "" {
		dirPath = path
	}

	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return "", err
	}

	return path, nil
}
