package server

import (
	"bytes"
	"errors"
	"os"
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

func TestModelPathValidate(t *testing.T) {
	valid := []string{
		"repo",
		"repo:tag",
		"ns/repo:tag",
		"example.com/ns/repo:tag",
		"example.com:5000/ns/repo:tag",
		"https://example.com/ns/repo:tag",
		"llama3:8b-instruct-q4_0",
	}
	for _, name := range valid {
		if err := ParseModelPath(name).Validate(); err != nil {
			t.Errorf("Validate(%q) = %v, want nil", name, err)
		}
	}

	invalid := []string{
		"",
		"repo:",
		"../..:x",
		"../../..:x",
		"..:x",
		"ns/..",
		"ns/..:x",
		"../repo:x",
		"../ns/repo:x",
		"./repo:x",
		"ns/.:x",
		"repo:..",
		"repo:.",
		"repo:a:b",
		"ns\\..\\..:x",
		"repo:x\x00y",
	}
	for _, name := range invalid {
		err := ParseModelPath(name).Validate()
		if !errors.Is(err, errModelPathInvalid) {
			t.Errorf("Validate(%q) = %v, want %v", name, err, errModelPathInvalid)
		}
	}
}

func TestGetManifestPathStaysInManifestsDir(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", dir)
	manifests := filepath.Join(dir, "manifests") + string(os.PathSeparator)

	for _, name := range []string{"repo", "ns/repo:tag", "example.com:5000/ns/repo:tag"} {
		fp, err := ParseModelPath(name).GetManifestPath()
		if err != nil {
			t.Fatalf("GetManifestPath(%q): %v", name, err)
		}
		if !strings.HasPrefix(fp, manifests) {
			t.Errorf("GetManifestPath(%q) = %q, want prefix %q", name, fp, manifests)
		}
	}

	for _, name := range []string{"../..:id_ed25519", "../../..:x", "./..:x", "ns/.:x", "repo:.."} {
		fp, err := ParseModelPath(name).GetManifestPath()
		if !errors.Is(err, errModelPathInvalid) {
			t.Errorf("GetManifestPath(%q) = %q, %v; want %v", name, fp, err, errModelPathInvalid)
		}
	}

	// paths built directly (bypassing ParseModelPath) are also rejected
	mp := ModelPath{Registry: "..", Namespace: "..", Repository: "repo", Tag: "x"}
	if _, err := mp.GetManifestPath(); !errors.Is(err, errModelPathInvalid) {
		t.Errorf("GetManifestPath(%+v) = %v, want %v", mp, err, errModelPathInvalid)
	}
}

func TestManifestPathTraversal(t *testing.T) {
	home := t.TempDir()
	models := filepath.Join(home, ".ollama", "models")
	t.Setenv("OLLAMA_MODELS", models)

	victim := filepath.Join(home, ".ollama", "id_ed25519")
	if err := os.MkdirAll(filepath.Dir(victim), 0o755); err != nil {
		t.Fatal(err)
	}
	original := []byte(`{"config":{"digest":"sha256:0"},"layers":[]}`)
	if err := os.WriteFile(victim, original, 0o600); err != nil {
		t.Fatal(err)
	}

	// manifests/registry.ollama.ai/../../id_ed25519 -> ~/.ollama/models/../id_ed25519
	name := "../..:id_ed25519"

	if err := WriteManifest(name, &Layer{}, nil); !errors.Is(err, errModelPathInvalid) {
		t.Errorf("WriteManifest(%q) = %v, want %v", name, err, errModelPathInvalid)
	}

	if _, _, err := GetManifest(ParseModelPath(name)); !errors.Is(err, errModelPathInvalid) {
		t.Errorf("GetManifest(%q) = %v, want %v", name, err, errModelPathInvalid)
	}

	if err := DeleteModel(name); !errors.Is(err, errModelPathInvalid) {
		t.Errorf("DeleteModel(%q) = %v, want %v", name, err, errModelPathInvalid)
	}

	got, err := os.ReadFile(victim)
	if err != nil {
		t.Fatalf("victim file: %v", err)
	}
	if !bytes.Equal(got, original) {
		t.Errorf("victim file modified: %q", got)
	}
}
