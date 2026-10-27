package llm

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"

	"log/slog"
)

type containerGGUF struct {
	ByteOrder binary.ByteOrder

	Version uint32

	V1 struct {
		NumTensor uint32
		NumKV     uint32
	}

	V2 struct {
		NumTensor uint64
		NumKV     uint64
	}

	V3 struct {
		NumTensor uint64
		NumKV     uint64
	}
}

func (c *containerGGUF) Name() string {
	return "gguf"
}

func (c *containerGGUF) Decode(rs io.ReadSeeker) (model, error) {
	if err := binary.Read(rs, c.ByteOrder, &c.Version); err != nil {
		return nil, err
	}

	var err error
	switch c.Version {
	case 1:
		err = binary.Read(rs, c.ByteOrder, &c.V1)
	case 2:
		err = binary.Read(rs, c.ByteOrder, &c.V2)
	default:
		err = binary.Read(rs, c.ByteOrder, &c.V3)
	}
	if err != nil {
		return nil, err
	}

	model := newGGUF(c)
	if n := model.numKV(); n > maxGGUFKVCount {
		return nil, fmt.Errorf("%w: %d key-values exceeds %d", ErrGGUFLimitExceeded, n, maxGGUFKVCount)
	}

	if n := model.numTensor(); n > maxGGUFTensorCount {
		return nil, fmt.Errorf("%w: %d tensors exceeds %d", ErrGGUFLimitExceeded, n, maxGGUFTensorCount)
	}

	slog.Debug(fmt.Sprintf("model = %#v", model))
	if err := model.Decode(rs); err != nil {
		return nil, err
	}

	return model, nil
}

// Upper bounds on counts and lengths read from a GGUF header. They are far
// above what real models use but keep a crafted file from driving unbounded
// allocation.
const (
	maxGGUFKVCount      = 1 << 16
	maxGGUFTensorCount  = 1 << 20
	maxGGUFTensorDims   = 4
	maxGGUFArrayLength  = 1 << 24
	maxGGUFStringLength = 64 << 20

	// ggufReadChunk bounds how many elements or bytes are allocated ahead of
	// the data actually being present in the file.
	ggufReadChunk = 1 << 16
)

var ErrGGUFLimitExceeded = errors.New("gguf: limit exceeded")

const (
	_ uint32 = iota
	GGUFTokenNormal
	GGUFTokenUnknown
	GGUFTokenControl
	GGUFTokenUserDefined
	GGUFTokenUnused
	GGUFTokenByte
)

const (
	ggufTypeUint8 uint32 = iota
	ggufTypeInt8
	ggufTypeUint16
	ggufTypeInt16
	ggufTypeUint32
	ggufTypeInt32
	ggufTypeFloat32
	ggufTypeBool
	ggufTypeString
	ggufTypeArray
	ggufTypeUint64
	ggufTypeInt64
	ggufTypeFloat64
)

type gguf struct {
	*containerGGUF

	kv      KV
	tensors []*Tensor

	parameters uint64
}

func newGGUF(container *containerGGUF) *gguf {
	return &gguf{
		containerGGUF: container,
		kv:            make(KV),
	}
}

func NewGGUFV3(bo binary.ByteOrder) *gguf {
	return newGGUF(&containerGGUF{ByteOrder: bo, Version: 3})
}

func (llm *gguf) KV() KV {
	return llm.kv
}

func (llm *gguf) Tensors() Tensors {
	return llm.tensors
}

func (llm *gguf) numTensor() uint64 {
	switch llm.Version {
	case 1:
		return uint64(llm.V1.NumTensor)
	case 2:
		return llm.V2.NumTensor
	default:
		return llm.V3.NumTensor
	}
}

func (llm *gguf) numKV() uint64 {
	switch llm.Version {
	case 1:
		return uint64(llm.V1.NumKV)
	case 2:
		return llm.V2.NumKV
	default:
		return llm.V3.NumKV
	}
}

func (llm *gguf) Decode(rs io.ReadSeeker) error {
	// decode key-values
	for i := 0; uint64(i) < llm.numKV(); i++ {
		k, err := readGGUFString(llm, rs)
		if err != nil {
			return err
		}

		t, err := readGGUF[uint32](llm, rs)
		if err != nil {
			return err
		}

		var v any
		switch t {
		case ggufTypeUint8:
			v, err = readGGUF[uint8](llm, rs)
		case ggufTypeInt8:
			v, err = readGGUF[int8](llm, rs)
		case ggufTypeUint16:
			v, err = readGGUF[uint16](llm, rs)
		case ggufTypeInt16:
			v, err = readGGUF[int16](llm, rs)
		case ggufTypeUint32:
			v, err = readGGUF[uint32](llm, rs)
		case ggufTypeInt32:
			v, err = readGGUF[int32](llm, rs)
		case ggufTypeUint64:
			v, err = readGGUF[uint64](llm, rs)
		case ggufTypeInt64:
			v, err = readGGUF[int64](llm, rs)
		case ggufTypeFloat32:
			v, err = readGGUF[float32](llm, rs)
		case ggufTypeFloat64:
			v, err = readGGUF[float64](llm, rs)
		case ggufTypeBool:
			v, err = readGGUF[bool](llm, rs)
		case ggufTypeString:
			v, err = readGGUFString(llm, rs)
		case ggufTypeArray:
			v, err = readGGUFArray(llm, rs)
		default:
			return fmt.Errorf("invalid type: %d", t)
		}

		if err != nil {
			return err
		}

		llm.kv[k] = v
	}

	// decode tensors
	for i := 0; uint64(i) < llm.numTensor(); i++ {
		name, err := readGGUFString(llm, rs)
		if err != nil {
			return err
		}

		// dims is the number of dimensions in the tensor
		dims, err := readGGUF[uint32](llm, rs)
		if err != nil {
			return err
		}

		if dims > maxGGUFTensorDims {
			return fmt.Errorf("%w: tensor %q has %d dimensions, max %d", ErrGGUFLimitExceeded, name, dims, maxGGUFTensorDims)
		}

		shape := [maxGGUFTensorDims]uint64{1, 1, 1, 1}
		for i := 0; uint32(i) < dims; i++ {
			shape[i], err = readGGUF[uint64](llm, rs)
			if err != nil {
				return err
			}
		}

		kind, err := readGGUF[uint32](llm, rs)
		if err != nil {
			return err
		}

		offset, err := readGGUF[uint64](llm, rs)
		if err != nil {
			return err
		}

		tensor := Tensor{
			Name:   name,
			Kind:   kind,
			Offset: offset,
			Shape:  shape[:],
		}

		llm.tensors = append(llm.tensors, &tensor)
		llm.parameters += tensor.parameters()
	}

	// patch KV with parameter count
	llm.kv["general.parameter_count"] = llm.parameters

	alignment, ok := llm.kv["general.alignment"].(uint32)
	if !ok || alignment == 0 {
		alignment = 32
	}

	offset, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	padding := llm.padding(offset, int64(alignment))
	if _, err := rs.Seek(padding, io.SeekCurrent); err != nil {
		return err
	}

	for _, tensor := range llm.tensors {
		if _, err := rs.Seek(int64(tensor.size()), io.SeekCurrent); err != nil {
			return err
		}

		padding := llm.padding(int64(tensor.size()), int64(alignment))
		if _, err := rs.Seek(padding, io.SeekCurrent); err != nil {
			return err
		}
	}

	return nil
}

func readGGUF[T any](llm *gguf, r io.Reader) (T, error) {
	var t T
	err := binary.Read(r, llm.ByteOrder, &t)
	return t, err
}

func writeGGUF[V any](llm *gguf, w io.Writer, t uint32, v V) error {
	if err := binary.Write(w, llm.ByteOrder, t); err != nil {
		return err
	}

	return binary.Write(w, llm.ByteOrder, v)
}

func readGGUFStringData(r io.Reader, length uint64) (string, error) {
	if length > maxGGUFStringLength {
		return "", fmt.Errorf("%w: string length %d exceeds %d", ErrGGUFLimitExceeded, length, maxGGUFStringLength)
	}

	var b bytes.Buffer
	b.Grow(int(min(length, ggufReadChunk)))
	if _, err := io.CopyN(&b, r, int64(length)); err != nil {
		return "", err
	}

	return b.String(), nil
}

func readGGUFV1String(llm *gguf, r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, llm.ByteOrder, &length); err != nil {
		return "", err
	}

	s, err := readGGUFStringData(r, length)
	if err != nil {
		return "", err
	}

	// gguf v1 strings are null-terminated
	return strings.TrimSuffix(s, "\x00"), nil
}

func readGGUFString(llm *gguf, r io.Reader) (string, error) {
	if llm.Version == 1 {
		return readGGUFV1String(llm, r)
	}

	var length uint64
	if err := binary.Read(r, llm.ByteOrder, &length); err != nil {
		return "", err
	}

	return readGGUFStringData(r, length)
}

func writeGGUFString(llm *gguf, w io.Writer, s string) error {
	if err := binary.Write(w, llm.ByteOrder, ggufTypeString); err != nil {
		return err
	}

	if err := binary.Write(w, llm.ByteOrder, uint64(len(s))); err != nil {
		return err
	}

	_, err := io.Copy(w, strings.NewReader(s))
	return err
}

func readGGUFV1Array(llm *gguf, r io.Reader) (any, error) {
	t, err := readGGUF[uint32](llm, r)
	if err != nil {
		return nil, err
	}

	n, err := readGGUF[uint32](llm, r)
	if err != nil {
		return nil, err
	}

	return readGGUFArrayElements(llm, r, t, uint64(n))
}

// readGGUFArray returns the array as a typed slice (e.g. []uint8, []float32,
// []string) rather than []any to avoid boxing every element.
func readGGUFArray(llm *gguf, r io.Reader) (any, error) {
	if llm.Version == 1 {
		return readGGUFV1Array(llm, r)
	}

	t, err := readGGUF[uint32](llm, r)
	if err != nil {
		return nil, err
	}

	n, err := readGGUF[uint64](llm, r)
	if err != nil {
		return nil, err
	}

	return readGGUFArrayElements(llm, r, t, n)
}

func readGGUFArrayElements(llm *gguf, r io.Reader, t uint32, n uint64) (any, error) {
	if n > maxGGUFArrayLength {
		return nil, fmt.Errorf("%w: array length %d exceeds %d", ErrGGUFLimitExceeded, n, maxGGUFArrayLength)
	}

	switch t {
	case ggufTypeUint8:
		return readGGUFArrayData[uint8](llm, r, n)
	case ggufTypeInt8:
		return readGGUFArrayData[int8](llm, r, n)
	case ggufTypeUint16:
		return readGGUFArrayData[uint16](llm, r, n)
	case ggufTypeInt16:
		return readGGUFArrayData[int16](llm, r, n)
	case ggufTypeUint32:
		return readGGUFArrayData[uint32](llm, r, n)
	case ggufTypeInt32:
		return readGGUFArrayData[int32](llm, r, n)
	case ggufTypeUint64:
		return readGGUFArrayData[uint64](llm, r, n)
	case ggufTypeInt64:
		return readGGUFArrayData[int64](llm, r, n)
	case ggufTypeFloat32:
		return readGGUFArrayData[float32](llm, r, n)
	case ggufTypeFloat64:
		return readGGUFArrayData[float64](llm, r, n)
	case ggufTypeBool:
		return readGGUFArrayData[bool](llm, r, n)
	case ggufTypeString:
		a := make([]string, 0, min(n, ggufReadChunk))
		for i := uint64(0); i < n; i++ {
			e, err := readGGUFString(llm, r)
			if err != nil {
				return nil, err
			}

			a = append(a, e)
		}

		return a, nil
	default:
		return nil, fmt.Errorf("invalid array type: %d", t)
	}
}

// readGGUFArrayData reads n fixed-size elements in bounded chunks so memory
// grows only as data is actually read from the file.
func readGGUFArrayData[T uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | float32 | float64 | bool](llm *gguf, r io.Reader, n uint64) ([]T, error) {
	var a []T
	for uint64(len(a)) < n {
		start, k := len(a), int(min(n-uint64(len(a)), ggufReadChunk))
		if cap(a)-start < k {
			a = slices.Grow(a, max(k, start))
		}

		a = a[:start+k]
		if err := binary.Read(r, llm.ByteOrder, a[start:]); err != nil {
			return nil, err
		}
	}

	return a, nil
}

func writeGGUFArray[S ~[]E, E any](llm *gguf, w io.Writer, t uint32, s S) error {
	if err := binary.Write(w, llm.ByteOrder, ggufTypeArray); err != nil {
		return err
	}

	if err := binary.Write(w, llm.ByteOrder, t); err != nil {
		return err
	}

	if err := binary.Write(w, llm.ByteOrder, uint64(len(s))); err != nil {
		return err
	}

	for _, e := range s {
		if err := binary.Write(w, llm.ByteOrder, e); err != nil {
			return err
		}
	}

	return nil
}

var ggufKVOrder = map[string][]string{
	"llama": {
		"general.architecture",
		"general.name",
		"llama.vocab_size",
		"llama.context_length",
		"llama.embedding_length",
		"llama.block_count",
		"llama.feed_forward_length",
		"llama.attention.head_count",
		"llama.attention.head_count_kv",
		"llama.attention.layer_norm_rms_epsilon",
		"llama.rope.freq_base",
		"llama.rope.dimension_count",
		"llama.expert_count",
		"llama.expert_used_count",
		"gemma.context_length",
		"gemma.embedding_length",
		"gemma.block_count",
		"gemma.feed_forward_length",
		"gemma.attention.head_count",
		"gemma.attention.head_count_kv",
		"gemma.attention.layer_norm_rms_epsilon",
		"gemma.attention.key_length",
		"gemma.attention.value_length",
		"general.file_type",
		"tokenizer.ggml.model",
		"tokenizer.ggml.tokens",
		"tokenizer.ggml.scores",
		"tokenizer.ggml.token_type",
		"tokenizer.ggml.bos_token_id",
		"tokenizer.ggml.eos_token_id",
		"tokenizer.ggml.unknown_token_id",
		"tokenizer.ggml.padding_token_id",
		"tokenizer.ggml.add_bos_token",
		"tokenizer.ggml.add_eos_token",
		"tokenizer.chat_template",
	},
}

func (llm *gguf) Encode(ws io.WriteSeeker, kv KV, tensors []Tensor) error {
	switch llm.Version {
	case 3:
		llm.V3.NumTensor = uint64(len(tensors))
		llm.V3.NumKV = uint64(len(kv))
	default:
		return fmt.Errorf("not implemented: ggufv%d", llm.Version)
	}

	if err := binary.Write(ws, llm.ByteOrder, []byte("GGUF")); err != nil {
		return err
	}

	if err := binary.Write(ws, llm.ByteOrder, llm.Version); err != nil {
		return err
	}

	if err := binary.Write(ws, llm.ByteOrder, llm.numTensor()); err != nil {
		return err
	}

	if err := binary.Write(ws, llm.ByteOrder, llm.numKV()); err != nil {
		return err
	}

	kvCheck := make(map[string]bool)
	for k := range kv {
		kvCheck[k] = false
	}

	for _, k := range ggufKVOrder["llama"] {
		v, ok := kv[k]
		if !ok {
			continue
		}
		kvCheck[k] = true

		if err := binary.Write(ws, llm.ByteOrder, uint64(len(k))); err != nil {
			return err
		}

		if err := binary.Write(ws, llm.ByteOrder, []byte(k)); err != nil {
			return err
		}

		var err error
		switch v := v.(type) {
		case uint32:
			err = writeGGUF(llm, ws, ggufTypeUint32, v)
		case float32:
			err = writeGGUF(llm, ws, ggufTypeFloat32, v)
		case bool:
			err = writeGGUF(llm, ws, ggufTypeBool, v)
		case string:
			err = writeGGUFString(llm, ws, v)
		case []int32:
			err = writeGGUFArray(llm, ws, ggufTypeInt32, v)
		case []uint32:
			err = writeGGUFArray(llm, ws, ggufTypeUint32, v)
		case []float32:
			err = writeGGUFArray(llm, ws, ggufTypeFloat32, v)
		case []string:
			if err := binary.Write(ws, llm.ByteOrder, ggufTypeArray); err != nil {
				return err
			}

			if err := binary.Write(ws, llm.ByteOrder, ggufTypeString); err != nil {
				return err
			}

			if err := binary.Write(ws, llm.ByteOrder, uint64(len(v))); err != nil {
				return err
			}

			for _, e := range v {
				if err := binary.Write(ws, llm.ByteOrder, uint64(len(e))); err != nil {
					return err
				}

				if err := binary.Write(ws, llm.ByteOrder, []byte(e)); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("improper type for '%s'", k)
		}
		if err != nil {
			return err
		}
	}

	for k, v := range kvCheck {
		if !v {
			return fmt.Errorf("Didn't know how to write kv %s", k)
		}
	}

	for _, tensor := range tensors {
		if err := binary.Write(ws, llm.ByteOrder, uint64(len(tensor.Name))); err != nil {
			return err
		}

		if err := binary.Write(ws, llm.ByteOrder, []byte(tensor.Name)); err != nil {
			return err
		}

		dims := 0
		for cnt := 0; cnt < len(tensor.Shape); cnt++ {
			if tensor.Shape[cnt] > 0 {
				dims++
			}
		}

		if err := binary.Write(ws, llm.ByteOrder, uint32(dims)); err != nil {
			return err
		}

		for i := 0; i < dims; i++ {
			if err := binary.Write(ws, llm.ByteOrder, uint64(tensor.Shape[dims-1-i])); err != nil {
				return err
			}
		}

		if err := binary.Write(ws, llm.ByteOrder, tensor.Kind); err != nil {
			return err
		}

		if err := binary.Write(ws, llm.ByteOrder, tensor.Offset); err != nil {
			return err
		}
	}

	offset, err := ws.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	var alignment int64 = 32
	padding := llm.padding(offset, alignment)
	if err := binary.Write(ws, llm.ByteOrder, bytes.Repeat([]byte{0}, int(padding))); err != nil {
		return err
	}

	for _, tensor := range tensors {
		if _, err := tensor.WriteTo(ws); err != nil {
			return err
		}

		offset, err := ws.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}

		padding := llm.padding(offset, alignment)
		if err := binary.Write(ws, llm.ByteOrder, bytes.Repeat([]byte{0}, int(padding))); err != nil {
			return err
		}
	}

	return nil
}

func (gguf) padding(offset, align int64) int64 {
	return (align - offset%align) % align
}
