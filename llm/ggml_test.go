package llm

import (
	"bytes"
	"encoding/binary"
	"os"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/gpu"
)

type testModel struct {
	kv      KV
	tensors Tensors
}

func (m testModel) KV() KV           { return m.kv }
func (m testModel) Tensors() Tensors { return m.tensors }

func testGPUs() []gpu.GpuInfo {
	g := gpu.GpuInfo{Library: "cuda"}
	g.TotalMemory = 24 * format.GigaByte
	g.FreeMemory = 12 * format.GigaByte
	return []gpu.GpuInfo{g}
}

func TestLayersMalformedNames(t *testing.T) {
	ts := Tensors{
		{Name: "blk"},
		{Name: "blk.0.attn_q.weight"},
		{Name: ""},
	}

	layers := ts.Layers()
	if _, ok := layers["blk"][""]; !ok {
		t.Errorf("expected bare blk tensor to be kept, got %v", layers)
	}
	if _, ok := layers["blk.0"]["attn_q.weight"]; !ok {
		t.Errorf("expected blk.0 layer, got %v", layers)
	}
}

func TestMalformedMetadataDoesNotPanic(t *testing.T) {
	cases := map[string]testModel{
		"empty": {kv: KV{}},
		"llama without head count or tokens": {
			kv:      KV{"general.architecture": "llama", "llama.block_count": uint32(1), "llama.embedding_length": uint32(4096)},
			tensors: Tensors{{Name: "blk"}, {Name: "blk.0.attn_q.weight", Shape: []uint64{1}}},
		},
		"llama with wrongly typed metadata": {
			kv: KV{
				"general.architecture":       "llama",
				"llama.block_count":          uint32(1),
				"llama.embedding_length":     uint32(4096),
				"llama.attention.head_count": uint32(32),
				"llama.feed_forward_length":  "14336",
				"tokenizer.ggml.tokens":      "not an array",
			},
			tensors: Tensors{
				{Name: "blk.0.ffn_gate_exps.weight", Shape: []uint64{1}},
			},
		},
		"llama with one dimensional ffn_gate.0": {
			kv: KV{
				"general.architecture":       "llama",
				"llama.block_count":          uint32(1),
				"llama.embedding_length":     uint32(4096),
				"llama.attention.head_count": uint32(32),
			},
			tensors: Tensors{
				{Name: "blk.0.ffn_gate.0.weight", Shape: []uint64{1}},
			},
		},
	}

	for name, m := range cases {
		t.Run(name, func(t *testing.T) {
			ggml := &GGML{model: m}
			ggml.GraphSize(2048, 512)
			EstimateGPULayers(testGPUs(), ggml, nil, api.DefaultOptions())
			PredictServerFit(testGPUs(), ggml, nil, nil, api.DefaultOptions())
		})
	}
}

func TestGraphSizeMissingHeadCount(t *testing.T) {
	ggml := &GGML{model: testModel{kv: KV{"general.architecture": "llama", "llama.embedding_length": uint32(4096)}}}
	if partial, full := ggml.GraphSize(2048, 512); partial != 0 || full != 0 {
		t.Errorf("expected zero graph size without head count, got %d %d", partial, full)
	}
}

func TestCraftedGGUFDoesNotPanic(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "crafted.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	// llama model with no attention.head_count, no tokenizer tokens and a tensor named "blk"
	if err := NewGGUFV3(binary.LittleEndian).Encode(f, KV{
		"general.architecture":   "llama",
		"llama.context_length":   uint32(32),
		"llama.embedding_length": uint32(4096),
		"llama.block_count":      uint32(1),
	}, []Tensor{
		{Name: "blk", Kind: 0, Shape: []uint64{1, 1, 1, 1}, WriterTo: &bytes.Reader{}},
	}); err != nil {
		t.Fatal(err)
	}

	ggml, err := LoadModel(f.Name())
	if err != nil {
		t.Fatal(err)
	}

	PredictServerFit(testGPUs(), ggml, nil, nil, api.DefaultOptions())
}
