//go:build !windows

package lifecycle

import (
	"context"
	"fmt"
)

func DoUpgrade(cancel context.CancelFunc, done chan int) error {
	return fmt.Errorf("DoUpgrade not yet implemented")
}

// verifyInstaller is a no-op: staged updates are only executed on Windows.
func verifyInstaller(path string) error {
	return nil
}
