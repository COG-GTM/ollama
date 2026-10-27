package llm

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
)

// gglaMaxDims matches GGML_MAX_DIMS.
const gglaMaxDims = 4

type containerGGLA struct {
	version uint32
}

func (c *containerGGLA) Name() string {
	return "ggla"
}

func (c *containerGGLA) Decode(rs io.ReadSeeker) (model, error) {
	if err := binary.Read(rs, binary.LittleEndian, &c.version); err != nil {
		return nil, err
	}

	switch c.version {
	case 1:
	default:
		return nil, errors.New("invalid version")
	}

	model := newGGLA(c)
	err := model.decode(rs)
	return model, err
}

type ggla struct {
	*containerGGLA

	kv      KV
	tensors []*Tensor
}

func newGGLA(container *containerGGLA) *ggla {
	return &ggla{
		containerGGLA: container,
		kv:            make(KV),
	}
}

func (llm *ggla) KV() KV {
	return llm.kv
}

func (llm *ggla) Tensors() Tensors {
	return llm.tensors
}

func (llm *ggla) decode(rs io.ReadSeeker) error {
	start, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	// readers that cannot seek from the end are bounded only by int64
	end, err := rs.Seek(0, io.SeekEnd)
	if err != nil {
		end = math.MaxInt64
	}

	if _, err := rs.Seek(start, io.SeekStart); err != nil {
		return err
	}

	var r uint32
	if err := binary.Read(rs, binary.LittleEndian, &r); err != nil {
		return err
	}
	llm.kv["r"] = r

	var alpha uint32
	if err := binary.Read(rs, binary.LittleEndian, &alpha); err != nil {
		return err
	}
	llm.kv["alpha"] = alpha

	last := int64(-1)
	for {
		pos, err := rs.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}

		if pos <= last {
			return fmt.Errorf("ggla: tensor entry at offset %d does not advance past %d", pos, last)
		}
		last = pos

		var dims uint32
		if err := binary.Read(rs, binary.LittleEndian, &dims); err != nil {
			return err
		}

		if dims > gglaMaxDims {
			return fmt.Errorf("ggla: tensor has %d dimensions, max %d", dims, gglaMaxDims)
		}

		var namesize uint32
		if err := binary.Read(rs, binary.LittleEndian, &namesize); err != nil {
			return err
		}

		if int64(namesize) > end-pos {
			return fmt.Errorf("ggla: tensor name size %d exceeds file size", namesize)
		}

		var t Tensor
		if err := binary.Read(rs, binary.LittleEndian, &t.Kind); err != nil {
			return err
		}

		t.Shape = make([]uint64, dims)
		for i := 0; uint32(i) < dims; i++ {
			var shape32 uint32
			if err := binary.Read(rs, binary.LittleEndian, &shape32); err != nil {
				return err
			}

			t.Shape[i] = uint64(shape32)
		}

		// ggla tensor shape is reversed
		// ref: https://github.com/ggerganov/llama.cpp/blob/29ae62d2ae163e2b68aa0ad3bf2ab4636de0c957/convert-lora-to-ggml.py#L44
		slices.Reverse(t.Shape)

		name := make([]byte, namesize)
		if err := binary.Read(rs, binary.LittleEndian, &name); err != nil {
			return err
		}

		t.Name = string(name)

		offset, err := rs.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}

		if _, err := rs.Seek((offset+31)&-32, io.SeekStart); err != nil {
			return err
		}

		offset, err = rs.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}

		t.Offset = uint64(offset)

		size, ok := t.checkedSize()
		if !ok || offset > end || size > uint64(end-offset) {
			return fmt.Errorf("ggla: tensor %q data exceeds file size", t.Name)
		}

		if _, err := rs.Seek(int64(size), io.SeekCurrent); err != nil {
			return err
		}

		llm.tensors = append(llm.tensors, &t)
	}
}
