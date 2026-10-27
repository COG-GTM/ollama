package llm

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

type testTensor struct {
	name  string
	shape []uint64
	kind  uint32
}

// buildGGUF writes a minimal little-endian GGUF v3 file with no KV pairs
// followed by dataLen bytes of tensor data.
func buildGGUF(t *testing.T, tensors []testTensor, dataLen int) []byte {
	t.Helper()

	var b bytes.Buffer
	w := func(v any) {
		if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}

	w(uint32(FILE_MAGIC_GGUF_LE))
	w(uint32(3))
	w(uint64(len(tensors)))
	w(uint64(0))
	for _, tt := range tensors {
		w(uint64(len(tt.name)))
		b.WriteString(tt.name)
		w(uint32(len(tt.shape)))
		for _, n := range tt.shape {
			w(n)
		}
		w(tt.kind)
		w(uint64(0))
	}

	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}

	b.Write(make([]byte, dataLen))
	return b.Bytes()
}

func TestDecodeGGMLValid(t *testing.T) {
	data := buildGGUF(t, []testTensor{{name: "a", shape: []uint64{4, 2}, kind: 0}}, 32)

	ggml, offset, err := DecodeGGML(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}

	if offset != int64(len(data)) {
		t.Fatalf("offset = %d, want %d", offset, len(data))
	}

	if got := ggml.KV().ParameterCount(); got != 8 {
		t.Fatalf("parameter count = %d, want 8", got)
	}
}

func TestDecodeGGMLRejectsMalformedTensors(t *testing.T) {
	cases := []struct {
		name    string
		tensors []testTensor
		dataLen int
	}{
		{
			// F32 size = 4*n wraps to -64, seeking back to byte 0 after the 64-byte header
			name:    "size wraps negative",
			tensors: []testTensor{{name: "a", shape: []uint64{1<<62 - 16}, kind: 0}},
		},
		{
			name:    "parameter product overflows",
			tensors: []testTensor{{name: "a", shape: []uint64{1 << 32, 1 << 32}, kind: 0}},
		},
		{
			name:    "size exceeds int64",
			tensors: []testTensor{{name: "a", shape: []uint64{math.MaxInt64/4 + 1}, kind: 0}},
		},
		{
			name:    "too many dimensions",
			tensors: []testTensor{{name: "a", shape: []uint64{1, 1, 1, 1, 1}, kind: 0}},
		},
		{
			name:    "data past end of file",
			tensors: []testTensor{{name: "a", shape: []uint64{1024}, kind: 0}},
			dataLen: 32,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			data := buildGGUF(t, tt.tensors, tt.dataLen)

			_, offset, err := DecodeGGML(bytes.NewReader(data))
			if !errors.Is(err, ErrInvalidTensor) {
				t.Fatalf("err = %v, offset = %d; want ErrInvalidTensor", err, offset)
			}
		})
	}
}

func TestTensorCheckedSize(t *testing.T) {
	cases := []struct {
		tensor Tensor
		want   int64
	}{
		{Tensor{Kind: 0, Shape: []uint64{4, 2, 1, 1}}, 32},
		{Tensor{Kind: 1, Shape: []uint64{4, 2, 1, 1}}, 16},
		{Tensor{Kind: 2, Shape: []uint64{64, 1, 1, 1}}, 36},
	}

	for _, tt := range cases {
		got, err := tt.tensor.checkedSize()
		if err != nil {
			t.Fatal(err)
		}

		if got != tt.want || uint64(got) != tt.tensor.size() {
			t.Fatalf("checkedSize = %d, size = %d, want %d", got, tt.tensor.size(), tt.want)
		}
	}
}
