package llm

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
)

type ggufBuilder struct {
	bytes.Buffer
}

func newGGUFBuilder(version uint32, numTensor, numKV uint64) *ggufBuilder {
	var b ggufBuilder
	b.put([]byte("GGUF"), version)
	if version == 1 {
		b.put(uint32(numTensor), uint32(numKV))
	} else {
		b.put(numTensor, numKV)
	}
	return &b
}

func (b *ggufBuilder) put(vs ...any) *ggufBuilder {
	for _, v := range vs {
		if err := binary.Write(b, binary.LittleEndian, v); err != nil {
			panic(err)
		}
	}
	return b
}

func (b *ggufBuilder) str(s string) *ggufBuilder {
	return b.put(uint64(len(s)), []byte(s))
}

func decodeGGUFBytes(t *testing.T, data []byte) (*GGML, error) {
	t.Helper()
	ggml, _, err := DecodeGGML(bytes.NewReader(data))
	return ggml, err
}

func allocatedDuring(f func()) uint64 {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	f()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

func TestGGUFDecodeTypedArrays(t *testing.T) {
	f, err := os.Create(filepath.Join(t.TempDir(), "model.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	kv := KV{
		"general.architecture":      "gemma",
		"gemma.context_length":      uint32(2048),
		"gemma.embedding_length":    uint32(4),
		"tokenizer.ggml.tokens":     []string{"<s>", "</s>", "hello"},
		"tokenizer.ggml.scores":     []float32{0, 0.5, -1},
		"tokenizer.ggml.token_type": []int32{3, 3, 1},
	}
	if err := NewGGUFV3(binary.LittleEndian).Encode(f, kv, nil); err != nil {
		t.Fatal(err)
	}

	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}

	ggml, _, err := DecodeGGML(f)
	if err != nil {
		t.Fatal(err)
	}

	got := ggml.KV()
	if tokens, ok := got["tokenizer.ggml.tokens"].([]string); !ok || !slices.Equal(tokens, []string{"<s>", "</s>", "hello"}) {
		t.Errorf("tokens = %#v", got["tokenizer.ggml.tokens"])
	}
	if scores, ok := got["tokenizer.ggml.scores"].([]float32); !ok || !slices.Equal(scores, []float32{0, 0.5, -1}) {
		t.Errorf("scores = %#v", got["tokenizer.ggml.scores"])
	}
	if types, ok := got["tokenizer.ggml.token_type"].([]int32); !ok || !slices.Equal(types, []int32{3, 3, 1}) {
		t.Errorf("token_type = %#v", got["tokenizer.ggml.token_type"])
	}
	if got.Architecture() != "gemma" || got.ContextLength() != 2048 {
		t.Errorf("architecture = %q, context = %d", got.Architecture(), got.ContextLength())
	}

	// GraphSize reads the vocabulary size from the typed tokens array.
	if _, full := ggml.GraphSize(2048, 512); full != 4*512*(4+3) {
		t.Errorf("GraphSize full offload = %d, want %d", full, 4*512*(4+3))
	}
}

func TestGGUFDecodeBoolArrayIsCompact(t *testing.T) {
	const n = 1 << 20
	b := newGGUFBuilder(3, 0, 1).str("test.flags").put(ggufTypeArray, ggufTypeBool, uint64(n), make([]byte, n))

	var ggml *GGML
	var err error
	alloc := allocatedDuring(func() { ggml, err = decodeGGUFBytes(t, b.Bytes()) })
	if err != nil {
		t.Fatal(err)
	}

	flags, ok := ggml.KV()["test.flags"].([]bool)
	if !ok || len(flags) != n {
		t.Fatalf("flags = %T len %d", ggml.KV()["test.flags"], len(flags))
	}

	// Boxing each element as `any` costs at least 16 bytes per element plus
	// slice growth; a typed []bool needs one byte plus growth and read buffers.
	if alloc > 8*n {
		t.Errorf("decoding a %d byte bool array allocated %d bytes", n, alloc)
	}
}

func TestGGUFDecodeLimits(t *testing.T) {
	cases := []struct {
		name string
		data []byte
	}{
		{
			name: "kv count",
			data: newGGUFBuilder(3, 0, 1<<40).Bytes(),
		},
		{
			name: "tensor count",
			data: newGGUFBuilder(3, 1<<40, 0).Bytes(),
		},
		{
			name: "array length",
			data: newGGUFBuilder(3, 0, 1).str("a").put(ggufTypeArray, ggufTypeBool, uint64(1)<<40).Bytes(),
		},
		{
			name: "v1 array length",
			data: newGGUFBuilder(1, 0, 1).put(uint64(2), []byte("a\x00"), ggufTypeArray, ggufTypeUint8, uint32(maxGGUFArrayLength+1)).Bytes(),
		},
		{
			name: "string array element length",
			data: newGGUFBuilder(3, 0, 1).str("a").put(ggufTypeArray, ggufTypeString, uint64(1), uint64(1)<<62).Bytes(),
		},
		{
			name: "string value length",
			data: newGGUFBuilder(3, 0, 1).str("a").put(ggufTypeString, uint64(maxGGUFStringLength+1)).Bytes(),
		},
		{
			name: "key length",
			data: newGGUFBuilder(3, 0, 1).put(uint64(1) << 62).Bytes(),
		},
		{
			name: "tensor dims",
			data: newGGUFBuilder(3, 1, 0).str("t").put(uint32(5), [5]uint64{1, 1, 1, 1, 1}, uint32(0), uint64(0)).Bytes(),
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			var err error
			alloc := allocatedDuring(func() { _, err = decodeGGUFBytes(t, tt.data) })
			if !errors.Is(err, ErrGGUFLimitExceeded) {
				t.Fatalf("err = %v, want %v", err, ErrGGUFLimitExceeded)
			}
			if alloc > 1<<20 {
				t.Errorf("allocated %d bytes before rejecting", alloc)
			}
		})
	}
}

func TestGGUFDecodeTruncatedArrayAllocatesIncrementally(t *testing.T) {
	for _, typ := range []uint32{ggufTypeUint64, ggufTypeString} {
		data := newGGUFBuilder(3, 0, 1).str("a").put(ggufTypeArray, typ, uint64(maxGGUFArrayLength), make([]byte, 64)).Bytes()

		alloc := allocatedDuring(func() { decodeGGUFBytes(t, data) })
		if alloc > 4<<20 {
			t.Errorf("type %d: declared %d elements in a %d byte file and allocated %d bytes", typ, maxGGUFArrayLength, len(data), alloc)
		}
	}
}

func TestGGUFDecodeMalformedDoesNotPanic(t *testing.T) {
	cases := map[string][]byte{
		"v1 empty string": newGGUFBuilder(1, 0, 1).put(uint64(0), ggufTypeUint32, uint32(1)).Bytes(),
		"zero alignment":  newGGUFBuilder(3, 0, 1).str("general.alignment").put(ggufTypeUint32, uint32(0)).Bytes(),
	}

	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := decodeGGUFBytes(t, data); err != nil {
				t.Fatal(err)
			}
		})
	}
}
