package llm

import (
	"bytes"
	"encoding/binary"
	"testing"
	"time"
)

type gglaTestTensor struct {
	kind  uint32
	shape []uint32
	name  string
	data  []byte
}

func encodeGGLA(t *testing.T, tensors ...gglaTestTensor) []byte {
	t.Helper()

	var b bytes.Buffer
	write := func(v any) {
		if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}

	write(uint32(FILE_MAGIC_GGLA))
	write(uint32(1))  // version
	write(uint32(8))  // r
	write(uint32(16)) // alpha

	for _, tt := range tensors {
		write(uint32(len(tt.shape)))
		write(uint32(len(tt.name)))
		write(tt.kind)
		for _, n := range tt.shape {
			write(n)
		}
		b.WriteString(tt.name)
		for b.Len()%32 != 0 {
			b.WriteByte(0)
		}
		b.Write(tt.data)
	}

	return b.Bytes()
}

func decodeGGLAWithTimeout(t *testing.T, data []byte) (*GGML, int64, error) {
	t.Helper()

	type result struct {
		ggml *GGML
		size int64
		err  error
	}

	done := make(chan result, 1)
	go func() {
		ggml, size, err := DecodeGGML(bytes.NewReader(data))
		done <- result{ggml, size, err}
	}()

	select {
	case r := <-done:
		return r.ggml, r.size, r.err
	case <-time.After(5 * time.Second):
		t.Fatal("DecodeGGML did not return")
		return nil, 0, nil
	}
}

func TestGGLADecode(t *testing.T) {
	data := encodeGGLA(t,
		gglaTestTensor{kind: 0, shape: []uint32{2, 2}, name: "a.weight", data: make([]byte, 16)},
		gglaTestTensor{kind: 1, shape: []uint32{4}, name: "b.weight", data: make([]byte, 8)},
	)

	ggml, size, err := decodeGGLAWithTimeout(t, data)
	if err != nil {
		t.Fatal(err)
	}

	if size != int64(len(data)) {
		t.Errorf("size = %d, want %d", size, len(data))
	}

	tensors := ggml.Tensors()
	if len(tensors) != 2 {
		t.Fatalf("got %d tensors, want 2", len(tensors))
	}

	if tensors[0].Name != "a.weight" || tensors[0].Offset != 64 || tensors[0].size() != 16 {
		t.Errorf("unexpected first tensor %+v", tensors[0])
	}

	if tensors[1].Name != "b.weight" || tensors[1].Offset != 128 || tensors[1].size() != 8 {
		t.Errorf("unexpected second tensor %+v", tensors[1])
	}
}

func TestGGLADecodeRejectsMalformed(t *testing.T) {
	cases := map[string][]byte{
		// F32 size wraps to 2^64-48, seeking back to the start of the entry
		"overflow seeks backwards": encodeGGLA(t, gglaTestTensor{kind: 0, shape: []uint32{4294967292, 3221225475}}),
		"overflow wraps to small":  encodeGGLA(t, gglaTestTensor{kind: 0, shape: []uint32{1 << 31, 1 << 31, 1 << 2}, data: make([]byte, 64)}),
		"data past end of file":    encodeGGLA(t, gglaTestTensor{kind: 0, shape: []uint32{1024, 1024}, name: "a", data: make([]byte, 16)}),
		"too many dimensions":      encodeGGLA(t, gglaTestTensor{kind: 0, shape: []uint32{1, 1, 1, 1, 1}, data: make([]byte, 4)}),
	}

	nameTooLong := encodeGGLA(t)
	nameTooLong = binary.LittleEndian.AppendUint32(nameTooLong, 1)          // dims
	nameTooLong = binary.LittleEndian.AppendUint32(nameTooLong, 0xffffffff) // namesize
	nameTooLong = binary.LittleEndian.AppendUint32(nameTooLong, 0)          // kind
	nameTooLong = binary.LittleEndian.AppendUint32(nameTooLong, 1)          // shape
	cases["name past end of file"] = nameTooLong

	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			if _, _, err := decodeGGLAWithTimeout(t, data); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}
