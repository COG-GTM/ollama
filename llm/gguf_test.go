package llm

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"testing"
)

type ggufBuilder struct {
	bytes.Buffer
	version uint32
}

func newGGUFBuilder(version uint32, numTensor, numKV uint64) *ggufBuilder {
	b := &ggufBuilder{version: version}
	b.u32(FILE_MAGIC_GGUF_LE)
	b.u32(version)
	if version == 1 {
		b.u32(uint32(numTensor))
		b.u32(uint32(numKV))
	} else {
		b.u64(numTensor)
		b.u64(numKV)
	}
	return b
}

func (b *ggufBuilder) u32(v uint32) { _ = binary.Write(b, binary.LittleEndian, v) }
func (b *ggufBuilder) u64(v uint64) { _ = binary.Write(b, binary.LittleEndian, v) }

func (b *ggufBuilder) str(s string) {
	if b.version == 1 {
		b.u64(uint64(len(s) + 1))
		b.WriteString(s)
		b.WriteByte(0)
		return
	}
	b.u64(uint64(len(s)))
	b.WriteString(s)
}

func (b *ggufBuilder) kvUint32(k string, v uint32) {
	b.str(k)
	b.u32(ggufTypeUint32)
	b.u32(v)
}

func (b *ggufBuilder) tensor(name string, shape ...uint64) {
	b.str(name)
	b.u32(uint32(len(shape)))
	for _, n := range shape {
		b.u64(n)
	}
	b.u32(0) // FP32
	b.u64(0) // offset
}

func decode(t *testing.T, data []byte) error {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("DecodeGGML panicked: %v", r)
		}
	}()

	_, _, err := DecodeGGML(bytes.NewReader(data))
	return err
}

func TestDecodeGGUFMalformed(t *testing.T) {
	cases := []struct {
		name  string
		build func() []byte
	}{
		{
			name: "tensor dims above 4",
			build: func() []byte {
				b := newGGUFBuilder(3, 1, 0)
				b.tensor("t", 1, 1, 1, 1, 1)
				return b.Bytes()
			},
		},
		{
			name: "tensor dims max uint32",
			build: func() []byte {
				b := newGGUFBuilder(3, 1, 0)
				b.str("t")
				b.u32(0xffffffff)
				return b.Bytes()
			},
		},
		{
			name: "v1 zero length key",
			build: func() []byte {
				b := newGGUFBuilder(1, 0, 1)
				b.u64(0)
				return b.Bytes()
			},
		},
		{
			name: "v1 zero length string value",
			build: func() []byte {
				b := newGGUFBuilder(1, 0, 1)
				b.str("general.name")
				b.u32(ggufTypeString)
				b.u64(0)
				return b.Bytes()
			},
		},
		{
			name: "v1 zero length string in array",
			build: func() []byte {
				b := newGGUFBuilder(1, 0, 1)
				b.str("tokenizer.ggml.tokens")
				b.u32(ggufTypeArray)
				b.u32(ggufTypeString)
				b.u32(1)
				b.u64(0)
				return b.Bytes()
			},
		},
		{
			name: "v1 string length above max int64",
			build: func() []byte {
				b := newGGUFBuilder(1, 0, 1)
				b.u64(1 << 63)
				return b.Bytes()
			},
		},
		{
			name: "v3 string length above max int64",
			build: func() []byte {
				b := newGGUFBuilder(3, 0, 1)
				b.u64(1 << 63)
				return b.Bytes()
			},
		},
		{
			name: "zero alignment",
			build: func() []byte {
				b := newGGUFBuilder(3, 0, 1)
				b.kvUint32("general.alignment", 0)
				return b.Bytes()
			},
		},
		{
			name: "zero alignment with tensor",
			build: func() []byte {
				b := newGGUFBuilder(3, 1, 1)
				b.kvUint32("general.alignment", 0)
				b.tensor("t", 4)
				return b.Bytes()
			},
		},
		{
			name: "non power of two alignment",
			build: func() []byte {
				b := newGGUFBuilder(3, 0, 1)
				b.kvUint32("general.alignment", 3)
				return b.Bytes()
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			err := decode(t, tt.build())
			if err == nil || errors.Is(err, io.EOF) {
				t.Fatalf("expected a decode error, got %v", err)
			}
		})
	}
}

func TestDecodeGGUFValid(t *testing.T) {
	for _, version := range []uint32{1, 2, 3} {
		b := newGGUFBuilder(version, 1, 2)
		b.str("general.name")
		b.u32(ggufTypeString)
		b.str("test")
		b.kvUint32("general.alignment", 64)
		b.tensor("t", 2, 4, 4, 2)
		data := b.Bytes()
		data = append(data, make([]byte, 64-len(data)%64)...)
		data = append(data, make([]byte, 2*4*4*2*4)...)

		ggml, size, err := DecodeGGML(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("v%d: unexpected error: %v", version, err)
		}

		if size != int64(len(data)) {
			t.Errorf("v%d: size = %d, want %d", version, size, len(data))
		}

		if got := ggml.KV()["general.name"]; got != "test" {
			t.Errorf("v%d: general.name = %q, want %q", version, got, "test")
		}

		if got := ggml.KV().ParameterCount(); got != 64 {
			t.Errorf("v%d: parameter count = %d, want 64", version, got)
		}
	}
}

func TestDecodeGGLAMalformed(t *testing.T) {
	var b bytes.Buffer
	for _, v := range []uint32{FILE_MAGIC_GGLA, 1, 8, 16, 0xffffffff, 1, 0} {
		_ = binary.Write(&b, binary.LittleEndian, v)
	}

	if err := decode(t, b.Bytes()); err == nil || errors.Is(err, io.EOF) {
		t.Fatalf("expected a decode error, got %v", err)
	}
}
