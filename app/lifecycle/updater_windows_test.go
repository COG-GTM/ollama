package lifecycle

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestVerifyInstallerRejectsUnsignedOnWindows(t *testing.T) {
	// The go test binary is a valid but unsigned PE file.
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if err := verifyInstaller(self); err == nil {
		t.Fatal("expected unsigned executable to be rejected")
	}
	if err := verifyInstaller(writeTestFile(t, []byte("MZ not really"))); err == nil {
		t.Fatal("expected bogus executable to be rejected")
	}
}

func TestDoUpgradeRefusesUnsignedInstaller(t *testing.T) {
	origStage := UpdateStageDir
	UpdateStageDir = t.TempDir()
	t.Cleanup(func() { UpdateStageDir = origStage })

	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(self)
	if err != nil {
		t.Fatal(err)
	}
	staged := filepath.Join(UpdateStageDir, "abc123", Installer)
	if err := os.MkdirAll(filepath.Dir(staged), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(staged, data, 0o755); err != nil {
		t.Fatal(err)
	}

	cancelled := false
	err = DoUpgrade(context.CancelFunc(func() { cancelled = true }), nil)
	if err == nil || !strings.Contains(err.Error(), "failed verification") {
		t.Fatalf("expected verification failure, got %v", err)
	}
	if cancelled {
		t.Fatal("server was shut down for an unverified installer")
	}
	if _, err := os.Stat(staged); !os.IsNotExist(err) {
		t.Fatalf("rejected installer was not removed: %v", err)
	}
}

func TestDownloadNewReleaseRejectsUnsignedOnWindows(t *testing.T) {
	payload := []byte("MZ unsigned installer")
	s := newUpdateServer(t, payload)

	err := DownloadNewRelease(context.Background(), UpdateResponse{UpdateURL: s.url(), UpdateSHA256: digest(payload)})
	if err == nil || !strings.Contains(err.Error(), "failed verification") {
		t.Fatalf("expected verification failure, got %v", err)
	}
	if files := stagedInstallers(t); len(files) != 0 {
		t.Fatalf("unexpected staged files %v", files)
	}
}
