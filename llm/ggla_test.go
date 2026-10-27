package llm

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"strings"
	"testing"
)

func gglaHeader(t *testing.T, fields ...uint32) *bytes.Buffer {
	t.Helper()

	var b bytes.Buffer
	for _, f := range append([]uint32{FILE_MAGIC_GGLA, 1, 8, 16}, fields...) {
		if err := binary.Write(&b, binary.LittleEndian, f); err != nil {
			t.Fatal(err)
		}
	}

	return &b
}

func TestDecodeGGLARejectsOversizedDims(t *testing.T) {
	// dims, namesize, kind
	b := gglaHeader(t, math.MaxUint32, 4, 0)

	var err error
	allocs := testing.AllocsPerRun(1, func() {
		_, _, err = DecodeGGML(bytes.NewReader(b.Bytes()))
	})
	if err == nil || !strings.Contains(err.Error(), "dimensions") {
		t.Fatalf("expected dimensions error, got %v", err)
	}
	if allocs > 32 {
		t.Fatalf("unexpected allocations: %v", allocs)
	}
}

func TestDecodeGGLARejectsOversizedName(t *testing.T) {
	b := gglaHeader(t, 2, math.MaxUint32, 0)

	_, _, err := DecodeGGML(bytes.NewReader(b.Bytes()))
	if err == nil || !strings.Contains(err.Error(), "name size") {
		t.Fatalf("expected name size error, got %v", err)
	}
}

func TestDecodeGGLATruncatedName(t *testing.T) {
	b := gglaHeader(t, 2, 16, 0, 4, 8)
	b.WriteString("short")

	_, _, err := DecodeGGML(bytes.NewReader(b.Bytes()))
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("expected unexpected EOF, got %v", err)
	}
}

func TestDecodeGGLA(t *testing.T) {
	name := "blk.0.attn_q.weight.loraA"
	// dims, namesize, kind (F32), shape (reversed on disk)
	b := gglaHeader(t, 2, uint32(len(name)), 0, 4, 8)
	b.WriteString(name)
	b.Write(make([]byte, 32-b.Len()%32))
	b.Write(make([]byte, 4*8*4))

	ggml, size, err := DecodeGGML(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if size != int64(b.Len()) {
		t.Fatalf("size = %d, want %d", size, b.Len())
	}

	tensors := ggml.Tensors()
	if len(tensors) != 1 {
		t.Fatalf("got %d tensors, want 1", len(tensors))
	}
	if tensors[0].Name != name {
		t.Fatalf("name = %q, want %q", tensors[0].Name, name)
	}
	if got := tensors[0].Shape; len(got) != 2 || got[0] != 8 || got[1] != 4 {
		t.Fatalf("shape = %v, want [8 4]", got)
	}
}
