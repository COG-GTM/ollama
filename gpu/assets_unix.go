//go:build !windows

package gpu

import (
	"fmt"
	"io/fs"
	"os"
	"syscall"
)

func checkOwner(path string, info fs.FileInfo) error {
	if info.Mode().Perm()&0o022 != 0 {
		return fmt.Errorf("%s is writable by group or others (mode %v)", path, info.Mode().Perm())
	}
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return fmt.Errorf("unable to determine owner of %s", path)
	}
	if int(st.Uid) != os.Geteuid() {
		return fmt.Errorf("%s is owned by uid %d, not the current user (uid %d)", path, st.Uid, os.Geteuid())
	}
	return nil
}
