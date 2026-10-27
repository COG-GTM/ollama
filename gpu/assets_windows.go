package gpu

import "io/fs"

// Payloads are not extracted to a temporary directory on Windows
func checkOwner(path string, info fs.FileInfo) error {
	return nil
}
