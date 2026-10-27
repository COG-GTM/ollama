package lifecycle

import (
	"fmt"
	"syscall"
	"unsafe"
)

var (
	modwintrust        = syscall.NewLazyDLL("wintrust.dll")
	procWinVerifyTrust = modwintrust.NewProc("WinVerifyTrust")

	// WINTRUST_ACTION_GENERIC_VERIFY_V2
	winTrustActionGenericVerifyV2 = syscall.GUID{
		Data1: 0xaac56b,
		Data2: 0xcd44,
		Data3: 0x11d0,
		Data4: [8]byte{0x8c, 0xc2, 0x00, 0xc0, 0x4f, 0xc2, 0x95, 0xee},
	}
)

const (
	wtdUINone            = 2
	wtdRevokeWholeChain  = 1
	wtdChoiceFile        = 1
	wtdStateActionVerify = 1
	wtdStateActionClose  = 2
)

type winTrustFileInfo struct {
	cbStruct       uint32
	pcwszFilePath  *uint16
	hFile          syscall.Handle
	pgKnownSubject *syscall.GUID
}

type winTrustData struct {
	cbStruct            uint32
	pPolicyCallbackData uintptr
	pSIPClientData      uintptr
	dwUIChoice          uint32
	fdwRevocationChecks uint32
	dwUnionChoice       uint32
	pFile               *winTrustFileInfo
	dwStateAction       uint32
	hWVTStateData       syscall.Handle
	pwszURLReference    *uint16
	dwProvFlags         uint32
	dwUIContext         uint32
	pSignatureSettings  uintptr
}

// verifyAuthenticode checks the file's Authenticode signature and certificate
// chain with WinVerifyTrust.
func verifyAuthenticode(path string) error {
	pathPtr, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return err
	}
	fileInfo := winTrustFileInfo{pcwszFilePath: pathPtr}
	fileInfo.cbStruct = uint32(unsafe.Sizeof(fileInfo))
	data := winTrustData{
		dwUIChoice:          wtdUINone,
		fdwRevocationChecks: wtdRevokeWholeChain,
		dwUnionChoice:       wtdChoiceFile,
		pFile:               &fileInfo,
		dwStateAction:       wtdStateActionVerify,
	}
	data.cbStruct = uint32(unsafe.Sizeof(data))

	hwnd := ^uintptr(0) // INVALID_HANDLE_VALUE: no UI
	ret, _, _ := procWinVerifyTrust.Call(hwnd, uintptr(unsafe.Pointer(&winTrustActionGenericVerifyV2)), uintptr(unsafe.Pointer(&data)))

	data.dwStateAction = wtdStateActionClose
	procWinVerifyTrust.Call(hwnd, uintptr(unsafe.Pointer(&winTrustActionGenericVerifyV2)), uintptr(unsafe.Pointer(&data))) //nolint:errcheck

	if ret != 0 {
		return fmt.Errorf("authenticode verification failed: %w", syscall.Errno(uint32(ret)))
	}
	return nil
}

// verifyInstaller refuses installers that are not validly Authenticode signed
// by TrustedInstallerPublisher.
func verifyInstaller(path string) error {
	if err := verifyAuthenticode(path); err != nil {
		return err
	}
	return verifyInstallerPublisher(path)
}
