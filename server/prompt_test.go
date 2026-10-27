package server

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestPrompt(t *testing.T) {
	tests := []struct {
		name     string
		template string
		system   string
		prompt   string
		response string
		generate bool
		want     string
	}{
		{
			name:     "simple prompt",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			system:   "You are a Wizard.",
			prompt:   "What are the potion ingredients?",
			want:     "[INST] You are a Wizard. What are the potion ingredients? [/INST]",
		},
		{
			name:     "implicit response",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			system:   "You are a Wizard.",
			prompt:   "What are the potion ingredients?",
			response: "I don't know.",
			want:     "[INST] You are a Wizard. What are the potion ingredients? [/INST]I don't know.",
		},
		{
			name:     "response",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST] {{ .Response }}",
			system:   "You are a Wizard.",
			prompt:   "What are the potion ingredients?",
			response: "I don't know.",
			want:     "[INST] You are a Wizard. What are the potion ingredients? [/INST] I don't know.",
		},
		{
			name:     "cut",
			template: "<system>{{ .System }}</system><user>{{ .Prompt }}</user><assistant>{{ .Response }}</assistant>",
			system:   "You are a Wizard.",
			prompt:   "What are the potion ingredients?",
			response: "I don't know.",
			generate: true,
			want:     "<system>You are a Wizard.</system><user>What are the potion ingredients?</user><assistant>I don't know.",
		},
		{
			name:     "nocut",
			template: "<system>{{ .System }}</system><user>{{ .Prompt }}</user><assistant>{{ .Response }}</assistant>",
			system:   "You are a Wizard.",
			prompt:   "What are the potion ingredients?",
			response: "I don't know.",
			want:     "<system>You are a Wizard.</system><user>What are the potion ingredients?</user><assistant>I don't know.</assistant>",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Prompt(tc.template, tc.system, tc.prompt, tc.response, tc.generate)
			if err != nil {
				t.Errorf("error = %v", err)
			}

			if got != tc.want {
				t.Errorf("got = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestChatPrompt(t *testing.T) {
	tests := []struct {
		name     string
		template string
		messages []api.Message
		window   int
		want     string
	}{
		{
			name:     "simple prompt",
			template: "[INST] {{ .Prompt }} [/INST]",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			window: 1024,
			want:   "[INST] Hello [/INST]",
		},
		{
			name:     "with system message",
			template: "[INST] {{ if .System }}<<SYS>>{{ .System }}<</SYS>> {{ end }}{{ .Prompt }} [/INST]",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "Hello"},
			},
			window: 1024,
			want:   "[INST] <<SYS>>You are a Wizard.<</SYS>> Hello [/INST]",
		},
		{
			name:     "with response",
			template: "[INST] {{ if .System }}<<SYS>>{{ .System }}<</SYS>> {{ end }}{{ .Prompt }} [/INST] {{ .Response }}",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "I am?"},
			},
			window: 1024,
			want:   "[INST] <<SYS>>You are a Wizard.<</SYS>> Hello [/INST] I am?",
		},
		{
			name:     "with implicit response",
			template: "[INST] {{ if .System }}<<SYS>>{{ .System }}<</SYS>> {{ end }}{{ .Prompt }} [/INST]",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "I am?"},
			},
			window: 1024,
			want:   "[INST] <<SYS>>You are a Wizard.<</SYS>> Hello [/INST]I am?",
		},
		{
			name:     "with conversation",
			template: "[INST] {{ if .System }}<<SYS>>{{ .System }}<</SYS>> {{ end }}{{ .Prompt }} [/INST] {{ .Response }} ",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "What are the potion ingredients?"},
				{Role: "assistant", Content: "sugar"},
				{Role: "user", Content: "Anything else?"},
			},
			window: 1024,
			want:   "[INST] <<SYS>>You are a Wizard.<</SYS>> What are the potion ingredients? [/INST] sugar [INST] Anything else? [/INST] ",
		},
		{
			name:     "with truncation",
			template: "{{ .System }} {{ .Prompt }} {{ .Response }} ",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "I am?"},
				{Role: "user", Content: "Why is the sky blue?"},
				{Role: "assistant", Content: "The sky is blue from rayleigh scattering"},
			},
			window: 10,
			want:   "You are a Wizard. Why is the sky blue? The sky is blue from rayleigh scattering",
		},
		{
			name:     "images",
			template: "{{ .System }} {{ .Prompt }}",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "Hello", Images: []api.ImageData{[]byte("base64")}},
			},
			window: 1024,
			want:   "You are a Wizard. [img-0] Hello",
		},
		{
			name:     "images truncated",
			template: "{{ .System }} {{ .Prompt }}",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "Hello", Images: []api.ImageData{[]byte("img1"), []byte("img2")}},
			},
			window: 1024,
			want:   "You are a Wizard. [img-0] [img-1] Hello",
		},
		{
			name:     "empty list",
			template: "{{ .System }} {{ .Prompt }}",
			messages: []api.Message{},
			window:   1024,
			want:     "",
		},
		{
			name:     "empty prompt",
			template: "[INST] {{ if .System }}<<SYS>>{{ .System }}<</SYS>> {{ end }}{{ .Prompt }} [/INST] {{ .Response }} ",
			messages: []api.Message{
				{Role: "user", Content: ""},
			},
			window: 1024,
			want:   "",
		},
	}

	encode := func(s string) ([]int, error) {
		words := strings.Fields(s)
		return make([]int, len(words)), nil
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ChatPrompt(tc.template, tc.messages, tc.window, encode)
			if err != nil {
				t.Errorf("error = %v", err)
			}

			if got != tc.want {
				t.Errorf("got: %q, want: %q", got, tc.want)
			}
		})
	}
}

func TestPromptLimits(t *testing.T) {
	prompt := strings.Repeat("A", 4096)

	rejected := []struct {
		name     string
		template string
		prompt   string
	}{
		{"nested integer range", "{{range 100000}}{{range 100000}}{{$.Prompt}}{{end}}{{end}}", ""},
		{"empty integer range", "{{range 1000000000}}{{end}}", ""},
		{"range over variable", "{{$n := 1000000000}}{{range $n}}{{end}}", ""},
		{"range over function", "{{range len .Prompt}}{{range len .Prompt}}x{{end}}{{end}}", ""},
		{"nested range in else", "{{if .System}}{{else}}{{range 10}}{{end}}{{end}}", ""},
		{"recursive template", "{{template \"\" .}}{{template \"\" .}}", ""},
		{"define", "{{define \"a\"}}{{.Prompt}}{{end}}{{template \"a\" .}}", ""},
		{"block", "{{block \"a\" .}}{{.Prompt}}{{end}}", ""},
		{"oversized template", strings.Repeat("x", maxTemplateSize+1), ""},
		{"repeated prompt", strings.Repeat("{{.Prompt}}", 4096), ""},
		{"printf width", strings.Repeat(`{{printf "%999999d" 1}}`, 64), ""},
		{"printf star width", strings.Repeat(`{{printf "%*d" 999999 1}}`, 64), ""},
		{"doubling variables", `{{$a := print .Prompt .Prompt}}{{$a = print $a $a}}{{$a = print $a $a}}{{$a = print $a $a}}{{$a = print $a $a}}{{$a = print $a $a}}{{$a = print $a $a}}{{$a = print $a $a}}{{$a = print $a $a}}{{$a = print $a $a}}{{$a = print $a $a}}`, ""},
		{"nested escaping", `{{js (js (js (js (js (js (js (js (js (js (js (js (js (js (js (js .Prompt)))))))))))))))}}`, strings.Repeat("<", 4096)},
	}

	for _, tc := range rejected {
		t.Run(tc.name, func(t *testing.T) {
			if tc.prompt == "" {
				tc.prompt = prompt
			}

			start := time.Now()
			_, err := Prompt(tc.template, "", tc.prompt, "", false)
			if !errors.Is(err, errTemplateLimit) {
				t.Fatalf("err = %v, want %v", err, errTemplateLimit)
			}

			if elapsed := time.Since(start); elapsed > 5*time.Second {
				t.Fatalf("took %s to reject", elapsed)
			}
		})
	}

	allowed := []struct {
		name     string
		template string
		want     string
	}{
		{"conditionals", "{{ if .System }}<<SYS>>{{ .System }}<</SYS>>{{ end }}[INST] {{ .Prompt }} [/INST]", "<<SYS>>sys<</SYS>>[INST] " + prompt + " [/INST]"},
		{"range over field", "{{range .Messages}}x{{end}}{{.Prompt}}", prompt},
		{"range over dot", "{{range $k, $v := .}}{{end}}{{.Prompt}}", prompt},
		{"printf", `{{printf "%s|%5d|%.2f" .System 7 1.5}}`, "sys|    7|1.50"},
		{"html", `{{html "<b>"}}`, "&lt;b&gt;"},
	}

	for _, tc := range allowed {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Prompt(tc.template, "sys", prompt, "", true)
			if err != nil {
				t.Fatalf("err = %v", err)
			}

			if got != tc.want {
				t.Errorf("got = %q, want %q", got, tc.want)
			}
		})
	}
}
