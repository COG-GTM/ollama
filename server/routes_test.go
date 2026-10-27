package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/version"
)

func Test_Routes(t *testing.T) {
	type testCase struct {
		Name     string
		Method   string
		Path     string
		Setup    func(t *testing.T, req *http.Request)
		Expected func(t *testing.T, resp *http.Response)
	}

	createTestFile := func(t *testing.T, name string) string {
		t.Helper()

		f, err := os.CreateTemp(t.TempDir(), name)
		assert.Nil(t, err)
		defer f.Close()

		err = binary.Write(f, binary.LittleEndian, []byte("GGUF"))
		assert.Nil(t, err)

		err = binary.Write(f, binary.LittleEndian, uint32(3))
		assert.Nil(t, err)

		err = binary.Write(f, binary.LittleEndian, uint64(0))
		assert.Nil(t, err)

		err = binary.Write(f, binary.LittleEndian, uint64(0))
		assert.Nil(t, err)

		return f.Name()
	}

	createTestModel := func(t *testing.T, name string) {
		fname := createTestFile(t, "ollama-model")

		modelfile := strings.NewReader(fmt.Sprintf("FROM %s\nPARAMETER seed 42\nPARAMETER top_p 0.9\nPARAMETER stop foo\nPARAMETER stop bar", fname))
		commands, err := parser.Parse(modelfile)
		assert.Nil(t, err)
		fn := func(resp api.ProgressResponse) {
			t.Logf("Status: %s", resp.Status)
		}
		err = CreateModel(context.TODO(), name, "", "", commands, fn)
		assert.Nil(t, err)
	}

	testCases := []testCase{
		{
			Name:   "Version Handler",
			Method: http.MethodGet,
			Path:   "/api/version",
			Setup: func(t *testing.T, req *http.Request) {
			},
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, contentType, "application/json; charset=utf-8")
				body, err := io.ReadAll(resp.Body)
				assert.Nil(t, err)
				assert.Equal(t, fmt.Sprintf(`{"version":"%s"}`, version.Version), string(body))
			},
		},
		{
			Name:   "Tags Handler (no tags)",
			Method: http.MethodGet,
			Path:   "/api/tags",
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, contentType, "application/json; charset=utf-8")
				body, err := io.ReadAll(resp.Body)
				assert.Nil(t, err)

				var modelList api.ListResponse

				err = json.Unmarshal(body, &modelList)
				assert.Nil(t, err)

				assert.Equal(t, 0, len(modelList.Models))
			},
		},
		{
			Name:   "Tags Handler (yes tags)",
			Method: http.MethodGet,
			Path:   "/api/tags",
			Setup: func(t *testing.T, req *http.Request) {
				createTestModel(t, "test-model")
			},
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, contentType, "application/json; charset=utf-8")
				body, err := io.ReadAll(resp.Body)
				assert.Nil(t, err)

				var modelList api.ListResponse
				err = json.Unmarshal(body, &modelList)
				assert.Nil(t, err)

				assert.Equal(t, 1, len(modelList.Models))
				assert.Equal(t, modelList.Models[0].Name, "test-model:latest")
			},
		},
		{
			Name:   "Create Model Handler",
			Method: http.MethodPost,
			Path:   "/api/create",
			Setup: func(t *testing.T, req *http.Request) {
				f, err := os.CreateTemp(t.TempDir(), "ollama-model")
				assert.Nil(t, err)
				defer f.Close()

				stream := false
				createReq := api.CreateRequest{
					Name:      "t-bone",
					Modelfile: fmt.Sprintf("FROM %s", f.Name()),
					Stream:    &stream,
				}
				jsonData, err := json.Marshal(createReq)
				assert.Nil(t, err)

				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, "application/json", contentType)
				_, err := io.ReadAll(resp.Body)
				assert.Nil(t, err)
				assert.Equal(t, resp.StatusCode, 200)

				model, err := GetModel("t-bone")
				assert.Nil(t, err)
				assert.Equal(t, "t-bone:latest", model.ShortName)
			},
		},
		{
			Name:   "Copy Model Handler",
			Method: http.MethodPost,
			Path:   "/api/copy",
			Setup: func(t *testing.T, req *http.Request) {
				createTestModel(t, "hamshank")
				copyReq := api.CopyRequest{
					Source:      "hamshank",
					Destination: "beefsteak",
				}
				jsonData, err := json.Marshal(copyReq)
				assert.Nil(t, err)

				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				model, err := GetModel("beefsteak")
				assert.Nil(t, err)
				assert.Equal(t, "beefsteak:latest", model.ShortName)
			},
		},
		{
			Name:   "Show Model Handler",
			Method: http.MethodPost,
			Path:   "/api/show",
			Setup: func(t *testing.T, req *http.Request) {
				createTestModel(t, "show-model")
				showReq := api.ShowRequest{Model: "show-model"}
				jsonData, err := json.Marshal(showReq)
				assert.Nil(t, err)
				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, contentType, "application/json; charset=utf-8")
				body, err := io.ReadAll(resp.Body)
				assert.Nil(t, err)

				var showResp api.ShowResponse
				err = json.Unmarshal(body, &showResp)
				assert.Nil(t, err)

				var params []string
				paramsSplit := strings.Split(showResp.Parameters, "\n")
				for _, p := range paramsSplit {
					params = append(params, strings.Join(strings.Fields(p), " "))
				}
				sort.Strings(params)
				expectedParams := []string{
					"seed 42",
					"stop \"bar\"",
					"stop \"foo\"",
					"top_p 0.9",
				}
				assert.Equal(t, expectedParams, params)
			},
		},
	}

	s := &Server{}
	router := s.GenerateRoutes()

	httpSrv := httptest.NewServer(router)
	t.Cleanup(httpSrv.Close)

	workDir, err := os.MkdirTemp("", "ollama-test")
	assert.Nil(t, err)
	defer os.RemoveAll(workDir)
	os.Setenv("OLLAMA_MODELS", workDir)

	for _, tc := range testCases {
		t.Logf("Running Test: [%s]", tc.Name)
		u := httpSrv.URL + tc.Path
		req, err := http.NewRequestWithContext(context.TODO(), tc.Method, u, nil)
		assert.Nil(t, err)

		if tc.Setup != nil {
			tc.Setup(t, req)
		}

		resp, err := httpSrv.Client().Do(req)
		assert.Nil(t, err)
		defer resp.Body.Close()

		if tc.Expected != nil {
			tc.Expected(t, resp)
		}

	}
}

func TestCreateModelRejectsOverflowingTensorSize(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	f, err := os.CreateTemp(t.TempDir(), "ollama-model")
	assert.Nil(t, err)

	// GGUF v3 with one F32 tensor whose size (4*shape) wraps to -64, which
	// previously seeked back to byte 0 and made CreateModel loop forever
	var b bytes.Buffer
	for _, v := range []any{
		[]byte("GGUF"), uint32(3), uint64(1), uint64(0),
		uint64(1), []byte("a"), uint32(1), uint64(1<<62 - 16), uint32(0), uint64(0),
	} {
		assert.Nil(t, binary.Write(&b, binary.LittleEndian, v))
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	_, err = f.Write(b.Bytes())
	assert.Nil(t, err)
	assert.Nil(t, f.Close())

	commands, err := parser.Parse(strings.NewReader("FROM " + f.Name()))
	assert.Nil(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err = CreateModel(ctx, "overflow", "", "", commands, func(api.ProgressResponse) {})
	assert.ErrorIs(t, err, llm.ErrInvalidTensor)

	blobs, err := GetBlobsPath("")
	assert.Nil(t, err)
	partial, err := filepath.Glob(filepath.Join(blobs, "*-partial"))
	assert.Nil(t, err)
	assert.Empty(t, partial)
}
