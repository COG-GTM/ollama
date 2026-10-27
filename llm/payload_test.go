//go:build !windows

package llm

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/gpu"
)

func TestWritePayloadReplacesPlantedFile(t *testing.T) {
	dir := t.TempDir()
	dest := filepath.Join(dir, "ollama_llama_server")
	target := filepath.Join(t.TempDir(), "target")
	require.NoError(t, os.WriteFile(target, []byte("victim"), 0o644))
	require.NoError(t, os.Symlink(target, dest))

	require.NoError(t, writePayload(dest, strings.NewReader("real"), true))

	info, err := os.Lstat(dest)
	require.NoError(t, err)
	assert.True(t, info.Mode().IsRegular())
	assert.Equal(t, os.FileMode(0o755), info.Mode().Perm())
	got, err := os.ReadFile(dest)
	require.NoError(t, err)
	assert.Equal(t, "real", string(got))
	victim, err := os.ReadFile(target)
	require.NoError(t, err)
	assert.Equal(t, "victim", string(victim), "symlink target must not be written through")
}

func TestWritePayloadKeepsExistingWithoutOverwrite(t *testing.T) {
	dest := filepath.Join(t.TempDir(), "ollama_llama_server")
	require.NoError(t, os.WriteFile(dest, []byte("custom"), 0o755))

	require.NoError(t, writePayload(dest, strings.NewReader("real"), false))
	got, err := os.ReadFile(dest)
	require.NoError(t, err)
	assert.Equal(t, "custom", string(got))
}

func TestReinitAfterTmpDirReplaced(t *testing.T) {
	t.Setenv("OLLAMA_RUNNERS_DIR", "")
	t.Setenv("OLLAMA_TMPDIR", t.TempDir())
	gpu.ResetPayloadsDir()
	t.Cleanup(gpu.Cleanup)

	require.NoError(t, Init())
	servers := availableServers()
	require.NotEmpty(t, servers)

	var variant, dir string
	for variant, dir = range servers {
		break
	}
	server := serverBinary(dir)
	require.NoError(t, gpu.VerifyPayload(server))
	want, err := os.ReadFile(server)
	require.NoError(t, err)

	// A tmp cleaner removes the directory and another user recreates the same
	// path with their own runner.
	tmpDir := filepath.Dir(filepath.Dir(dir))
	require.NoError(t, os.RemoveAll(tmpDir))
	require.NoError(t, os.MkdirAll(dir, 0o777))
	require.NoError(t, os.Chmod(tmpDir, 0o777))
	planted := []byte("#!/bin/sh\ntouch /tmp/pwned\n")
	require.NoError(t, os.WriteFile(server, planted, 0o755))

	newDir, newServer, err := trustedServer(variant, dir)
	require.NoError(t, err)
	assert.NotEqual(t, dir, newDir)
	assert.NotEqual(t, server, newServer)
	got, err := os.ReadFile(newServer)
	require.NoError(t, err)
	assert.True(t, bytes.Equal(want, got))
}

func TestEnsurePayloadsAfterTmpDirRemoved(t *testing.T) {
	t.Setenv("OLLAMA_RUNNERS_DIR", "")
	t.Setenv("OLLAMA_TMPDIR", t.TempDir())
	gpu.ResetPayloadsDir()
	t.Cleanup(gpu.Cleanup)

	require.NoError(t, Init())
	before := availableServers()
	require.NotEmpty(t, before)

	var dir string
	for _, dir = range before {
		break
	}
	require.NoError(t, os.RemoveAll(filepath.Dir(filepath.Dir(dir))))
	assert.Empty(t, availableServers())

	require.NoError(t, ensurePayloads())
	after := availableServers()
	assert.Equal(t, len(before), len(after))
	for variant, newDir := range after {
		assert.NotEqual(t, before[variant], newDir)
		assert.NoError(t, gpu.VerifyPayload(serverBinary(newDir)))
	}
}
