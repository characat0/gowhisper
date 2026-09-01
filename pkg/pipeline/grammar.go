package pipeline

import (
	"fmt"

	"github.com/characat0/gowhisper/pkg/model"
)

// RequestOptions holds the per-request, model-agnostic decode settings.
type RequestOptions struct {
	Language string
}

// Option mutates RequestOptions.
type Option func(*RequestOptions)

// WithLanguage sets the transcription language (ISO 639-1, e.g. "de").
func WithLanguage(lang string) Option {
	return func(o *RequestOptions) { o.Language = lang }
}

// A PromptPart contributes zero or more decoder-prompt token ids from the request
// config and tokenizer.
type PromptPart func(o RequestOptions, t *model.Tokenizer) ([]int64, error)

// Tokens emits fixed tokens by name, in order, resolving each via the tokenizer.
// It errors loudly if any name is absent — this is both the building block for
// the named aliases below and the escape hatch for any custom prompt tokens.
func Tokens(names ...string) PromptPart {
	return func(_ RequestOptions, t *model.Tokenizer) ([]int64, error) {
		ids := make([]int64, 0, len(names))
		for _, n := range names {
			id, err := t.LookupToken(n)
			if err != nil {
				return nil, fmt.Errorf("pipeline: prompt token %q: %w", n, err)
			}
			ids = append(ids, id)
		}
		return ids, nil
	}
}

// Language emits the language token resolved from o.Language (e.g. "en" ->
// <|en|>) via tokenizer.Lang, which returns an error rather than panicking.
func Language() PromptPart {
	return func(o RequestOptions, t *model.Tokenizer) ([]int64, error) {
		id, err := t.Lang(o.Language)
		if err != nil {
			return nil, fmt.Errorf("pipeline: language %q: %w", o.Language, err)
		}
		return []int64{id}, nil
	}
}

// Named aliases over Tokens(...) for readable, hard-to-misuse grammars.

// StartOfTranscript emits <|startoftranscript|>.
func StartOfTranscript() PromptPart { return Tokens("<|startoftranscript|>") }

// Task emits <|transcribe|>.
func Task() PromptPart { return Tokens("<|transcribe|>") }

// NoTimestamps emits <|notimestamps|>.
func NoTimestamps() PromptPart { return Tokens("<|notimestamps|>") }

// Verbatim emits CrisperWhisper's verbatim policy prefix [verbatim_1..5]. The
// reference (crisperwhisper/prompt.py) always emits all five as an atomic block.
func Verbatim() PromptPart {
	return Tokens("[verbatim_1]", "[verbatim_2]", "[verbatim_3]", "[verbatim_4]", "[verbatim_5]")
}

// Intended emits CrisperWhisper's intended (clean) policy prefix [intended_1..5].
func Intended() PromptPart {
	return Tokens("[intended_1]", "[intended_2]", "[intended_3]", "[intended_4]", "[intended_5]")
}

// Grammar is an ordered list of prompt parts. It fully determines the decoder
// prompt for a checkpoint; callers pick or mutate it explicitly (see
// Pipeline.SetGrammar).
type Grammar []PromptPart

// build concatenates every part's tokens, in order, into the decoder prompt.
func (g Grammar) build(o RequestOptions, t *model.Tokenizer) ([]int64, error) {
	var prompt []int64
	for _, part := range g {
		toks, err := part(o, t)
		if err != nil {
			return nil, err
		}
		prompt = append(prompt, toks...)
	}
	return prompt, nil
}

// WhisperGrammar is the canonical Whisper decoder grammar: SOT onward. It's the
// default for any Whisper checkpoint and has no policy prefix.
func WhisperGrammar() Grammar {
	return Grammar{StartOfTranscript(), Language(), Task(), NoTimestamps()}
}

// CrisperVerbatimGrammar prepends the verbatim policy prefix to the Whisper
// grammar, activating verbatim output (disfluencies like [UM], [laughter]).
func CrisperVerbatimGrammar() Grammar {
	return append(Grammar{Verbatim()}, WhisperGrammar()...)
}

// CrisperIntendedGrammar prepends the intended policy prefix to the Whisper
// grammar, producing clean "what was meant" transcription.
func CrisperIntendedGrammar() Grammar {
	return append(Grammar{Intended()}, WhisperGrammar()...)
}
