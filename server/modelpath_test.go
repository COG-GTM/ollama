package server

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseModelPath(t *testing.T) {
	tests := []struct {
		name string
		arg  string
		want ModelPath
	}{
		{
			"full path https",
			"https://example.com/ns/repo:tag",
			ModelPath{
				ProtocolScheme: "https",
				Registry:       "example.com",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"full path http",
			"http://example.com/ns/repo:tag",
			ModelPath{
				ProtocolScheme: "http",
				Registry:       "example.com",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no protocol",
			"example.com/ns/repo:tag",
			ModelPath{
				ProtocolScheme: "https",
				Registry:       "example.com",
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no registry",
			"ns/repo:tag",
			ModelPath{
				ProtocolScheme: "https",
				Registry:       DefaultRegistry,
				Namespace:      "ns",
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no namespace",
			"repo:tag",
			ModelPath{
				ProtocolScheme: "https",
				Registry:       DefaultRegistry,
				Namespace:      DefaultNamespace,
				Repository:     "repo",
				Tag:            "tag",
			},
		},
		{
			"no tag",
			"repo",
			ModelPath{
				ProtocolScheme: "https",
				Registry:       DefaultRegistry,
				Namespace:      DefaultNamespace,
				Repository:     "repo",
				Tag:            DefaultTag,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseModelPath(tc.arg)

			if got != tc.want {
				t.Errorf("got: %q want: %q", got, tc.want)
			}
		})
	}
}

func TestGetBlobsPath(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", dir)

	valid := "sha256:" + strings.Repeat("a", 64)

	tests := []struct {
		name    string
		digest  string
		want    string
		wantErr error
	}{
		{"empty", "", filepath.Join(dir, "blobs"), nil},
		{"colon", valid, filepath.Join(dir, "blobs", "sha256-"+strings.Repeat("a", 64)), nil},
		{"dash", "sha256-" + strings.Repeat("a", 64), filepath.Join(dir, "blobs", "sha256-"+strings.Repeat("a", 64)), nil},
		{"traversal", "../../../../tmp/traversal-test", "", ErrInvalidDigestFormat},
		{"traversal with prefix", "sha256:../../../../tmp/traversal-test", "", ErrInvalidDigestFormat},
		{"short", "sha256:1234", "", ErrInvalidDigestFormat},
		{"uppercase", "sha256:" + strings.Repeat("A", 64), "", ErrInvalidDigestFormat},
		{"trailing path", valid + "/x", "", ErrInvalidDigestFormat},
		{"wrong algorithm", "sha512:" + strings.Repeat("a", 64), "", ErrInvalidDigestFormat},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := GetBlobsPath(tc.digest)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("error: got %v want %v", err, tc.wantErr)
			}

			if got != tc.want {
				t.Errorf("path: got %q want %q", got, tc.want)
			}
		})
	}
}
