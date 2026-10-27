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

	"github.com/stretchr/testify/assert"

	"github.com/ollama/ollama/api"
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

func TestModelPathTraversalRoutes(t *testing.T) {
	home := t.TempDir()
	t.Setenv("OLLAMA_MODELS", filepath.Join(home, ".ollama", "models"))

	victim := filepath.Join(home, ".ollama", "id_ed25519")
	assert.Nil(t, os.MkdirAll(filepath.Dir(victim), 0o755))
	original := []byte(`{"config":{"digest":"sha256:0"},"layers":[]}`)
	assert.Nil(t, os.WriteFile(victim, original, 0o600))

	s := &Server{}
	httpSrv := httptest.NewServer(s.GenerateRoutes())
	t.Cleanup(httpSrv.Close)

	stream := false
	modelfile := filepath.Join(t.TempDir(), "Modelfile")
	cases := []struct {
		method, path string
		body         any
	}{
		{http.MethodPost, "/api/create", api.CreateRequest{Name: "../..:id_ed25519", Modelfile: "FROM " + modelfile, Stream: &stream}},
		{http.MethodDelete, "/api/delete", api.DeleteRequest{Name: "../..:id_ed25519"}},
		{http.MethodPost, "/api/pull", api.PullRequest{Name: "../..:id_ed25519", Stream: &stream}},
		{http.MethodPost, "/api/push", api.PushRequest{Name: "../..:id_ed25519", Stream: &stream}},
	}

	for _, tc := range cases {
		t.Run(tc.path, func(t *testing.T) {
			b, err := json.Marshal(tc.body)
			assert.Nil(t, err)

			req, err := http.NewRequestWithContext(context.TODO(), tc.method, httpSrv.URL+tc.path, bytes.NewReader(b))
			assert.Nil(t, err)

			resp, err := httpSrv.Client().Do(req)
			assert.Nil(t, err)
			defer resp.Body.Close()

			assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
		})
	}

	got, err := os.ReadFile(victim)
	assert.Nil(t, err)
	assert.Equal(t, original, got)
}
