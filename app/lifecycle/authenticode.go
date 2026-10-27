package lifecycle

import (
	"bytes"
	"crypto/x509"
	"debug/pe"
	"encoding/asn1"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/big"
	"os"
)

// TrustedInstallerPublisher is the subject (Organization or CommonName) of the
// code signing certificate that release installers must be signed with.
var TrustedInstallerPublisher = "Ollama Inc."

const (
	winCertRevision2           = 0x0200
	winCertTypePKCSSignedData  = 0x0002
	winCertificateHeaderLength = 8
	maxAuthenticodeBlobSize    = 1 << 20
)

var (
	oidSignedData = asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 7, 2}

	errNotSigned = errors.New("installer is not Authenticode signed")
)

// verifyInstallerPublisher checks that the primary Authenticode signature
// embedded in the PE file at path was made with a certificate issued to
// TrustedInstallerPublisher. It does not validate the signature or the
// certificate chain; on Windows that is done by WinVerifyTrust.
func verifyInstallerPublisher(path string) error {
	signer, err := authenticodeSigner(path)
	if err != nil {
		return err
	}
	if signer.Subject.CommonName == TrustedInstallerPublisher {
		return nil
	}
	for _, org := range signer.Subject.Organization {
		if org == TrustedInstallerPublisher {
			return nil
		}
	}
	return fmt.Errorf("installer is signed by %q, expected %q", signer.Subject.String(), TrustedInstallerPublisher)
}

// authenticodeSigner returns the signing certificate of the primary
// Authenticode signature embedded in the PE file at path.
func authenticodeSigner(path string) (*x509.Certificate, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	blob, err := peCertificateTable(f)
	if err != nil {
		return nil, err
	}
	return pkcs7Signer(blob)
}

func peCertificateTable(r io.ReaderAt) ([]byte, error) {
	pf, err := pe.NewFile(r)
	if err != nil {
		return nil, fmt.Errorf("installer is not a valid PE file: %w", err)
	}
	defer pf.Close()

	var dirs []pe.DataDirectory
	var numDirs uint32
	switch oh := pf.OptionalHeader.(type) {
	case *pe.OptionalHeader32:
		dirs, numDirs = oh.DataDirectory[:], oh.NumberOfRvaAndSizes
	case *pe.OptionalHeader64:
		dirs, numDirs = oh.DataDirectory[:], oh.NumberOfRvaAndSizes
	default:
		return nil, errors.New("installer has no PE optional header")
	}
	if numDirs < uint32(len(dirs)) {
		dirs = dirs[:numDirs]
	}
	if len(dirs) <= pe.IMAGE_DIRECTORY_ENTRY_SECURITY {
		return nil, errNotSigned
	}
	dir := dirs[pe.IMAGE_DIRECTORY_ENTRY_SECURITY]
	if dir.VirtualAddress == 0 || dir.Size < winCertificateHeaderLength {
		return nil, errNotSigned
	}

	var hdr [winCertificateHeaderLength]byte
	if _, err := r.ReadAt(hdr[:], int64(dir.VirtualAddress)); err != nil {
		return nil, fmt.Errorf("read certificate table: %w", err)
	}
	length := binary.LittleEndian.Uint32(hdr[0:4])
	revision := binary.LittleEndian.Uint16(hdr[4:6])
	certType := binary.LittleEndian.Uint16(hdr[6:8])
	if revision != winCertRevision2 || certType != winCertTypePKCSSignedData {
		return nil, fmt.Errorf("unsupported certificate table entry (revision %#x, type %#x)", revision, certType)
	}
	if length <= winCertificateHeaderLength || length > dir.Size || length > maxAuthenticodeBlobSize {
		return nil, fmt.Errorf("invalid certificate table entry length %d", length)
	}

	blob := make([]byte, length-winCertificateHeaderLength)
	if _, err := r.ReadAt(blob, int64(dir.VirtualAddress)+winCertificateHeaderLength); err != nil {
		return nil, fmt.Errorf("read certificate table: %w", err)
	}
	return blob, nil
}

type pkcs7ContentInfo struct {
	ContentType asn1.ObjectIdentifier
	Content     asn1.RawValue `asn1:"explicit,tag:0"`
}

type pkcs7IssuerAndSerial struct {
	Issuer asn1.RawValue
	Serial *big.Int
}

// pkcs7Signer returns the certificate of the first SignerInfo of a DER
// encoded PKCS#7 SignedData ContentInfo.
func pkcs7Signer(der []byte) (*x509.Certificate, error) {
	var ci pkcs7ContentInfo
	if _, err := asn1.Unmarshal(der, &ci); err != nil {
		return nil, fmt.Errorf("parse signature: %w", err)
	}
	if !ci.ContentType.Equal(oidSignedData) {
		return nil, fmt.Errorf("signature is not PKCS#7 SignedData (%s)", ci.ContentType)
	}

	var sd asn1.RawValue
	if _, err := asn1.Unmarshal(ci.Content.Bytes, &sd); err != nil {
		return nil, fmt.Errorf("parse SignedData: %w", err)
	}
	if sd.Class != asn1.ClassUniversal || sd.Tag != asn1.TagSequence {
		return nil, errors.New("malformed SignedData")
	}

	var certs []*x509.Certificate
	var signerInfos []byte
	rest := sd.Bytes
	for len(rest) > 0 {
		var field asn1.RawValue
		var err error
		if rest, err = asn1.Unmarshal(rest, &field); err != nil {
			return nil, fmt.Errorf("parse SignedData: %w", err)
		}
		switch {
		case field.Class == asn1.ClassContextSpecific && field.Tag == 0:
			if certs, err = x509.ParseCertificates(field.Bytes); err != nil {
				return nil, fmt.Errorf("parse signature certificates: %w", err)
			}
		case field.Class == asn1.ClassUniversal && field.Tag == asn1.TagSet:
			// digestAlgorithms precedes signerInfos; the last SET wins.
			signerInfos = field.Bytes
		}
	}
	if len(signerInfos) == 0 {
		return nil, errors.New("signature has no signers")
	}

	var si asn1.RawValue
	if _, err := asn1.Unmarshal(signerInfos, &si); err != nil {
		return nil, fmt.Errorf("parse SignerInfo: %w", err)
	}
	var version, sid asn1.RawValue
	siRest, err := asn1.Unmarshal(si.Bytes, &version)
	if err != nil {
		return nil, fmt.Errorf("parse SignerInfo: %w", err)
	}
	if _, err := asn1.Unmarshal(siRest, &sid); err != nil {
		return nil, fmt.Errorf("parse SignerInfo: %w", err)
	}

	var match func(*x509.Certificate) bool
	switch {
	case sid.Class == asn1.ClassUniversal && sid.Tag == asn1.TagSequence:
		var ias pkcs7IssuerAndSerial
		if _, err := asn1.Unmarshal(sid.FullBytes, &ias); err != nil {
			return nil, fmt.Errorf("parse signer identifier: %w", err)
		}
		match = func(c *x509.Certificate) bool {
			return bytes.Equal(c.RawIssuer, ias.Issuer.FullBytes) && c.SerialNumber.Cmp(ias.Serial) == 0
		}
	case sid.Class == asn1.ClassContextSpecific && sid.Tag == 0:
		match = func(c *x509.Certificate) bool {
			return len(c.SubjectKeyId) > 0 && bytes.Equal(c.SubjectKeyId, sid.Bytes)
		}
	default:
		return nil, errors.New("unsupported signer identifier")
	}

	for _, c := range certs {
		if match(c) {
			return c, nil
		}
	}
	return nil, errors.New("signing certificate not found in signature")
}
