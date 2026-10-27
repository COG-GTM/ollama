package server

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"text/template"
	"text/template/parse"
	"time"
)

const (
	// maxTemplateSize bounds the size of a prompt template.
	maxTemplateSize = 64 << 10

	// a rendered prompt may be at most promptRenderExpansion times the size
	// of its inputs plus promptRenderSlack bytes
	promptRenderExpansion = 4
	promptRenderSlack     = 1 << 20

	promptRenderTimeout = 10 * time.Second

	// fmt rejects widths and precisions larger than this
	maxFormatWidth = 1e6
)

var (
	errTemplateLimit = errors.New("template exceeds rendering limits")

	errTemplateTooLarge  = fmt.Errorf("%w: template is larger than %d bytes", errTemplateLimit, maxTemplateSize)
	errPromptTooLarge    = fmt.Errorf("%w: rendered prompt is too large", errTemplateLimit)
	errPromptTimeout     = fmt.Errorf("%w: rendering took longer than %s", errTemplateLimit, promptRenderTimeout)
	errTemplateRange     = fmt.Errorf("%w: range is only supported over fields", errTemplateLimit)
	errTemplateSubtmpl   = fmt.Errorf("%w: define, block and template actions are not supported", errTemplateLimit)
	errTemplateUnhandled = fmt.Errorf("%w: unsupported template node", errTemplateLimit)
)

// parseTemplate parses tmpl and rejects constructs whose execution cost is not
// bounded by the size of the template and its inputs: ranging over integers or
// computed values, and invoking (possibly recursive) sub-templates.
func parseTemplate(tmpl string) (*template.Template, error) {
	if len(tmpl) > maxTemplateSize {
		return nil, errTemplateTooLarge
	}

	parsed, err := template.New("").Option("missingkey=zero").Parse(tmpl)
	if err != nil {
		return nil, err
	}

	if len(parsed.Templates()) > 1 {
		return nil, errTemplateSubtmpl
	}

	if parsed.Tree != nil {
		if err := validateTemplateNode(parsed.Tree.Root); err != nil {
			return nil, err
		}
	}

	return parsed, nil
}

func validateTemplateNode(node parse.Node) error {
	switch n := node.(type) {
	case nil:
		return nil
	case *parse.ListNode:
		if n == nil {
			return nil
		}
		for _, child := range n.Nodes {
			if err := validateTemplateNode(child); err != nil {
				return err
			}
		}
		return nil
	case *parse.TextNode, *parse.ActionNode, *parse.CommentNode:
		return nil
	case *parse.IfNode:
		return validateBranch(&n.BranchNode)
	case *parse.WithNode:
		return validateBranch(&n.BranchNode)
	case *parse.RangeNode:
		if !isFieldPipe(n.Pipe) {
			return errTemplateRange
		}
		return validateBranch(&n.BranchNode)
	case *parse.TemplateNode:
		return errTemplateSubtmpl
	case *parse.BreakNode, *parse.ContinueNode:
		return nil
	default:
		return fmt.Errorf("%w: %s", errTemplateUnhandled, node)
	}
}

func validateBranch(n *parse.BranchNode) error {
	if err := validateTemplateNode(n.List); err != nil {
		return err
	}
	return validateTemplateNode(n.ElseList)
}

// isFieldPipe reports whether pipe is a single field, dot, or variable field
// reference, so a range can only iterate over data supplied to the template.
func isFieldPipe(pipe *parse.PipeNode) bool {
	if pipe == nil || len(pipe.Cmds) != 1 || len(pipe.Cmds[0].Args) != 1 {
		return false
	}

	switch arg := pipe.Cmds[0].Args[0].(type) {
	case *parse.FieldNode, *parse.DotNode:
		return true
	case *parse.VariableNode:
		return len(arg.Ident) > 1 || (len(arg.Ident) == 1 && arg.Ident[0] == "$")
	default:
		return false
	}
}

// renderBudget bounds the bytes produced while executing a template, both
// written to the output and allocated by string-producing functions.
type renderBudget struct {
	sb        strings.Builder
	remaining int
	deadline  time.Time
}

func newRenderBudget(inputs ...string) *renderBudget {
	limit := promptRenderSlack
	for _, s := range inputs {
		limit += promptRenderExpansion * len(s)
	}

	return &renderBudget{remaining: limit, deadline: time.Now().Add(promptRenderTimeout)}
}

func (b *renderBudget) reserve(n int) error {
	if n < 0 || n > b.remaining {
		return errPromptTooLarge
	}

	if time.Now().After(b.deadline) {
		return errPromptTimeout
	}

	return nil
}

func (b *renderBudget) charge(n int) error {
	if err := b.reserve(n); err != nil {
		return err
	}

	b.remaining -= n
	return nil
}

func (b *renderBudget) Write(p []byte) (int, error) {
	if err := b.charge(len(p)); err != nil {
		return 0, err
	}

	return b.sb.Write(p)
}

func (b *renderBudget) String() string {
	return b.sb.String()
}

// funcs returns replacements for the text/template builtins that can produce
// output larger than their arguments. Each checks the worst-case result size
// against the budget before allocating it.
func (b *renderBudget) funcs() template.FuncMap {
	escaper := func(factor int, fn func(...any) string) func(...any) (string, error) {
		return func(args ...any) (string, error) {
			return b.call(factor*argsSize(args), func() string { return fn(args...) })
		}
	}

	return template.FuncMap{
		"print": func(args ...any) (string, error) {
			return b.call(argsSize(args)+len(args), func() string { return fmt.Sprint(args...) })
		},
		"println": func(args ...any) (string, error) {
			return b.call(argsSize(args)+len(args)+1, func() string { return fmt.Sprintln(args...) })
		},
		"printf": func(format string, args ...any) (string, error) {
			// %q and %x can expand each byte of an argument to up to 4 bytes
			return b.call(len(format)+4*argsSize(args)+formatWidths(format, args), func() string { return fmt.Sprintf(format, args...) })
		},
		// worst case expansions: html `"` -> `&#34;`, js `<` -> `\u003C`, urlquery ` ` -> `%20`
		"html":     escaper(5, template.HTMLEscaper),
		"js":       escaper(6, template.JSEscaper),
		"urlquery": escaper(3, template.URLQueryEscaper),
	}
}

func (b *renderBudget) call(estimate int, fn func() string) (string, error) {
	if err := b.reserve(estimate); err != nil {
		return "", err
	}

	s := fn()
	if err := b.charge(len(s)); err != nil {
		return "", err
	}

	return s, nil
}

// argsSize estimates the formatted size of template function arguments.
func argsSize(args []any) int {
	var n int
	for _, arg := range args {
		switch v := arg.(type) {
		case string:
			n += len(v)
		case []byte:
			n += len(v)
		default:
			rv := reflect.ValueOf(arg)
			switch rv.Kind() {
			case reflect.Invalid, reflect.Bool,
				reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
				reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
				reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
				n += 64
			case reflect.String:
				n += rv.Len()
			default:
				n += len(fmt.Sprint(arg))
			}
		}
	}
	return n
}

// formatWidths returns an upper bound on the padding the widths and
// precisions in a printf format can add.
func formatWidths(format string, args []any) int {
	var star int
	for _, arg := range args {
		rv := reflect.ValueOf(arg)
		switch rv.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			star = max(star, int(min(max(rv.Int(), -rv.Int()), maxFormatWidth)))
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
			star = max(star, int(min(rv.Uint(), maxFormatWidth)))
		}
	}

	var total int
	for i := 0; i < len(format); i++ {
		if format[i] != '%' {
			continue
		}

	verb:
		for i++; i < len(format); i++ {
			switch c := format[i]; {
			case c == '*':
				total += star
			case c >= '0' && c <= '9':
				// fmt accepts literal widths and precisions up to ~1e7, so
				// saturate well above that and let the budget reject them.
				var n int
				for ; i+1 < len(format) && format[i+1] >= '0' && format[i+1] <= '9'; i++ {
					n = min(n*10+int(format[i]-'0'), math.MaxInt32)
				}
				total += min(n*10+int(format[i]-'0'), math.MaxInt32)
			case strings.IndexByte("+-# .[]", c) >= 0:
			default:
				break verb
			}
		}
	}

	return total
}
