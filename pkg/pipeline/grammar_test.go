package pipeline

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/characat0/gowhisper/pkg/model"
)

// loadTokenizer loads the tokenizer shipped in the repo's bin/ dir (currently
// CrisperWhisper weights, which carry the [verbatim_*] blocks).
func loadTokenizer(t *testing.T) *model.Tokenizer {
	t.Helper()
	tok, err := model.NewTokenizer(filepath.Join("..", "..", "bin", "tokenizer.json"))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	return tok
}

// mustLookup resolves a token name to its id, failing the test if absent.
func mustLookup(t *testing.T, tok *model.Tokenizer, name string) int64 {
	t.Helper()
	id, err := tok.LookupToken(name)
	if err != nil {
		t.Fatalf("lookup %q: %v", name, err)
	}
	return id
}

func TestWhisperGrammarPrompt(t *testing.T) {
	tok := loadTokenizer(t)
	got, err := WhisperGrammar().build(RequestOptions{Language: "en"}, tok)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	want := []int64{
		mustLookup(t, tok, "<|startoftranscript|>"),
		mustLookup(t, tok, "<|en|>"),
		mustLookup(t, tok, "<|transcribe|>"),
		mustLookup(t, tok, "<|notimestamps|>"),
	}
	if !slices.Equal(got, want) {
		t.Errorf("WhisperGrammar prompt = %v, want %v", got, want)
	}
}

func TestCrisperVerbatimGrammarPrompt(t *testing.T) {
	tok := loadTokenizer(t)
	got, err := CrisperVerbatimGrammar().build(RequestOptions{Language: "en"}, tok)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	// The verbatim block precedes the standard Whisper prefix, matching
	// crisperwhisper/prompt.py (mode tags + get_decoder_prefix).
	want := []int64{
		mustLookup(t, tok, "[verbatim_1]"),
		mustLookup(t, tok, "[verbatim_2]"),
		mustLookup(t, tok, "[verbatim_3]"),
		mustLookup(t, tok, "[verbatim_4]"),
		mustLookup(t, tok, "[verbatim_5]"),
		mustLookup(t, tok, "<|startoftranscript|>"),
		mustLookup(t, tok, "<|en|>"),
		mustLookup(t, tok, "<|transcribe|>"),
		mustLookup(t, tok, "<|notimestamps|>"),
	}
	if !slices.Equal(got, want) {
		t.Errorf("CrisperVerbatimGrammar prompt = %v, want %v", got, want)
	}
}

func TestWithLanguageOverridesDefault(t *testing.T) {
	tok := loadTokenizer(t)
	ro := RequestOptions{Language: "en"}
	WithLanguage("de")(&ro)
	got, err := Language()(ro, tok)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	want := []int64{mustLookup(t, tok, "<|de|>")}
	if !slices.Equal(got, want) {
		t.Errorf("Language(de) = %v, want %v", got, want)
	}
}

func TestTokensErrorsOnMissingToken(t *testing.T) {
	tok := loadTokenizer(t)
	if _, err := Tokens("[definitely-not-a-token]")(RequestOptions{}, tok); err == nil {
		t.Error("expected error for missing token, got nil")
	}
}
