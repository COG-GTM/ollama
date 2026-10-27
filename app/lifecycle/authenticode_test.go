package lifecycle

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"debug/pe"
	"encoding/asn1"
	"encoding/binary"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func testCert(t *testing.T, org string, serial int64, ski []byte) *x509.Certificate {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(serial),
		Subject:      pkix.Name{CommonName: org + " code signing", Organization: []string{org}},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageCodeSigning},
		SubjectKeyId: ski,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}
	return cert
}

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	b, err := asn1.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func rawSeq(t *testing.T, class, tag int, parts ...[]byte) []byte {
	t.Helper()
	return mustMarshal(t, asn1.RawValue{Class: class, Tag: tag, IsCompound: true, Bytes: bytes.Join(parts, nil)})
}

// testSignedData builds a minimal PKCS#7 SignedData ContentInfo whose single
// SignerInfo identifies signer, either by issuer and serial or by SKI.
func testSignedData(t *testing.T, certs []*x509.Certificate, signer *x509.Certificate, bySKI bool) []byte {
	t.Helper()
	var raws [][]byte
	for _, c := range certs {
		raws = append(raws, c.Raw)
	}
	version := mustMarshal(t, 1)
	sid := mustMarshal(t, pkcs7IssuerAndSerial{Issuer: asn1.RawValue{FullBytes: signer.RawIssuer}, Serial: signer.SerialNumber})
	if bySKI {
		version = mustMarshal(t, 3)
		sid = mustMarshal(t, asn1.RawValue{Class: asn1.ClassContextSpecific, Tag: 0, Bytes: signer.SubjectKeyId})
	}
	sha256Alg := rawSeq(t, asn1.ClassUniversal, asn1.TagSequence, mustMarshal(t, asn1.ObjectIdentifier{2, 16, 840, 1, 101, 3, 4, 2, 1}))
	signerInfo := rawSeq(t, asn1.ClassUniversal, asn1.TagSequence,
		version, sid, sha256Alg,
		rawSeq(t, asn1.ClassUniversal, asn1.TagSequence, mustMarshal(t, asn1.ObjectIdentifier{1, 2, 840, 10045, 4, 3, 2})),
		mustMarshal(t, []byte("signature")),
	)
	signedData := rawSeq(t, asn1.ClassUniversal, asn1.TagSequence,
		mustMarshal(t, 1),
		rawSeq(t, asn1.ClassUniversal, asn1.TagSet, sha256Alg),
		rawSeq(t, asn1.ClassUniversal, asn1.TagSequence, mustMarshal(t, asn1.ObjectIdentifier{1, 3, 6, 1, 4, 1, 311, 2, 1, 4})),
		rawSeq(t, asn1.ClassContextSpecific, 0, raws...),
		rawSeq(t, asn1.ClassUniversal, asn1.TagSet, signerInfo),
	)
	return rawSeq(t, asn1.ClassUniversal, asn1.TagSequence,
		mustMarshal(t, oidSignedData),
		rawSeq(t, asn1.ClassContextSpecific, 0, signedData),
	)
}

// testPE builds a minimal PE32+ image with sig, if non-nil, in its
// certificate table.
func testPE(t *testing.T, sig []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	dos := make([]byte, 0x40)
	copy(dos, "MZ")
	binary.LittleEndian.PutUint32(dos[0x3c:], 0x40)
	buf.Write(dos)
	buf.WriteString("PE\x00\x00")

	oh := pe.OptionalHeader64{Magic: 0x20b, NumberOfRvaAndSizes: 16}
	fh := pe.FileHeader{
		Machine:              pe.IMAGE_FILE_MACHINE_AMD64,
		SizeOfOptionalHeader: uint16(binary.Size(oh)),
		Characteristics:      pe.IMAGE_FILE_EXECUTABLE_IMAGE,
	}
	certOffset := buf.Len() + binary.Size(fh) + binary.Size(oh)
	certOffset += (8 - certOffset%8) % 8
	var cert []byte
	if sig != nil {
		cert = make([]byte, winCertificateHeaderLength, winCertificateHeaderLength+len(sig))
		binary.LittleEndian.PutUint32(cert[0:], uint32(winCertificateHeaderLength+len(sig)))
		binary.LittleEndian.PutUint16(cert[4:], winCertRevision2)
		binary.LittleEndian.PutUint16(cert[6:], winCertTypePKCSSignedData)
		cert = append(cert, sig...)
		oh.DataDirectory[pe.IMAGE_DIRECTORY_ENTRY_SECURITY] = pe.DataDirectory{VirtualAddress: uint32(certOffset), Size: uint32(len(cert))}
	}
	if err := binary.Write(&buf, binary.LittleEndian, fh); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(&buf, binary.LittleEndian, oh); err != nil {
		t.Fatal(err)
	}
	buf.Write(make([]byte, certOffset-buf.Len()))
	buf.Write(cert)
	return buf.Bytes()
}

func writeTestFile(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "OllamaSetup.exe")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestVerifyInstallerPublisher(t *testing.T) {
	ollama := testCert(t, TrustedInstallerPublisher, 1, []byte{1, 2, 3})
	mallory := testCert(t, "Mallory Ltd", 2, []byte{4, 5, 6})

	cases := []struct {
		name    string
		data    []byte
		wantErr string
	}{
		{"trusted publisher", testPE(t, testSignedData(t, []*x509.Certificate{ollama}, ollama, false)), ""},
		{"trusted publisher by SKI", testPE(t, testSignedData(t, []*x509.Certificate{mallory, ollama}, ollama, true)), ""},
		{"other publisher", testPE(t, testSignedData(t, []*x509.Certificate{mallory}, mallory, false)), "expected"},
		{"trusted cert bundled but other signer", testPE(t, testSignedData(t, []*x509.Certificate{ollama, mallory}, mallory, false)), "expected"},
		{"signer cert missing", testPE(t, testSignedData(t, []*x509.Certificate{mallory}, ollama, false)), "not found"},
		{"unsigned PE", testPE(t, nil), "not Authenticode signed"},
		{"garbage signature", testPE(t, []byte("not a pkcs7 blob")), "parse signature"},
		{"not a PE file", []byte("#!/bin/sh\necho pwned\n"), "not a valid PE"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := verifyInstallerPublisher(writeTestFile(t, tc.data))
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected success, got %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestPECertificateTableBounds(t *testing.T) {
	ollama := testCert(t, TrustedInstallerPublisher, 1, nil)
	data := testPE(t, testSignedData(t, []*x509.Certificate{ollama}, ollama, false))

	// Truncate the file in the middle of the certificate table.
	if _, err := authenticodeSigner(writeTestFile(t, data[:len(data)-10])); err == nil {
		t.Fatal("expected error for truncated certificate table")
	}

	// Claim a WIN_CERTIFICATE length larger than the data directory.
	pf, err := pe.NewFile(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	off := pf.OptionalHeader.(*pe.OptionalHeader64).DataDirectory[pe.IMAGE_DIRECTORY_ENTRY_SECURITY].VirtualAddress
	bad := bytes.Clone(data)
	binary.LittleEndian.PutUint32(bad[off:], uint32(len(data)))
	if _, err := authenticodeSigner(writeTestFile(t, bad)); err == nil || !strings.Contains(err.Error(), "length") {
		t.Fatalf("expected length error, got %v", err)
	}
}
