package llm

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"testing"
)

type testTensor struct {
	name  string
	shape []uint64
}

// writeTestGGUF appends a little-endian GGUF v3 file with F32 tensors to buf.
func writeTestGGUF(t *testing.T, buf *bytes.Buffer, arch string, tokens int, tensors []testTensor) {
	t.Helper()
	start := buf.Len()
	w := func(v any) {
		if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	str := func(s string) {
		w(uint64(len(s)))
		buf.WriteString(s)
	}

	w(uint32(FILE_MAGIC_GGUF_LE))
	w(uint32(3))
	w(uint64(len(tensors)))
	w(uint64(3))

	str("general.architecture")
	w(ggufTypeString)
	str(arch)

	str("general.alignment")
	w(ggufTypeUint32)
	w(uint32(32))

	str("tokenizer.ggml.tokens")
	w(ggufTypeArray)
	w(ggufTypeString)
	w(uint64(tokens))
	for i := 0; i < tokens; i++ {
		str(fmt.Sprintf("tok%d", i))
	}

	var offset uint64
	sizes := make([]uint64, len(tensors))
	for i, tt := range tensors {
		str(tt.name)
		w(uint32(len(tt.shape)))
		n := uint64(1)
		for _, d := range tt.shape {
			w(d)
			n *= d
		}
		w(uint32(0)) // F32
		w(offset)
		sizes[i] = n * 4
		offset += sizes[i] + uint64((32-int64(sizes[i])%32)%32)
	}

	pad := func() {
		for (buf.Len()-start)%32 != 0 {
			buf.WriteByte(0)
		}
	}
	pad()
	for _, size := range sizes {
		buf.Write(bytes.Repeat([]byte{1}, int(size)))
		pad()
	}
}

func TestDecodeGGUFConcatenated(t *testing.T) {
	var buf bytes.Buffer
	writeTestGGUF(t, &buf, "llama", 5000, []testTensor{{"a", []uint64{8}}, {"b", []uint64{4, 4}}})
	firstLen := int64(buf.Len())
	writeTestGGUF(t, &buf, "clip", 3, []testTensor{{"c", []uint64{40}}})

	rs := bytes.NewReader(buf.Bytes())

	ggml, size, err := DecodeGGML(rs)
	if err != nil {
		t.Fatal(err)
	}
	if size != firstLen {
		t.Fatalf("size = %d, want %d", size, firstLen)
	}
	if got := ggml.KV().Architecture(); got != "llama" {
		t.Fatalf("architecture = %q", got)
	}
	if got := len(ggml.KV()["tokenizer.ggml.tokens"].([]any)); got != 5000 {
		t.Fatalf("tokens = %d", got)
	}
	if got := ggml.KV().ParameterCount(); got != 24 {
		t.Fatalf("parameter count = %d", got)
	}
	if got := len(ggml.Tensors()); got != 2 {
		t.Fatalf("tensors = %d", got)
	}

	if _, err := rs.Seek(size, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	// the returned offset is absolute within rs
	ggml, end, err := DecodeGGML(rs)
	if err != nil {
		t.Fatal(err)
	}
	if got := ggml.KV().Architecture(); got != "clip" {
		t.Fatalf("second architecture = %q", got)
	}
	if end != int64(buf.Len()) {
		t.Fatalf("second end = %d, want %d", end, buf.Len())
	}

	if _, err := rs.Seek(end, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	if _, _, err := DecodeGGML(rs); !errors.Is(err, io.EOF) {
		t.Fatalf("err = %v, want io.EOF", err)
	}
}
