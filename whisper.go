// Package gowhisper is the public API for the Whisper speech-to-text pipeline:
// it wires the mel front-end, ONNX encoder/decoder, and tokenizer into a single
// Pipeline that turns 16 kHz mono PCM into a transcript.
//
// The implementation lives in pkg/pipeline; this package re-exports it so callers
// keep the ergonomic top-level import path (github.com/characat0/gowhisper).
package gowhisper

import "github.com/characat0/gowhisper/pkg/pipeline"

// Core types.
type (
	Pipeline       = pipeline.Pipeline
	Grammar        = pipeline.Grammar
	PromptPart     = pipeline.PromptPart
	Option         = pipeline.Option
	RequestOptions = pipeline.RequestOptions
)

// Construction and per-request options.
var (
	NewPipeline  = pipeline.NewPipeline
	WithLanguage = pipeline.WithLanguage
)

// Predefined grammars.
var (
	WhisperGrammar         = pipeline.WhisperGrammar
	CrisperVerbatimGrammar = pipeline.CrisperVerbatimGrammar
	CrisperIntendedGrammar = pipeline.CrisperIntendedGrammar
)

// Prompt parts (building blocks for custom grammars).
var (
	Tokens            = pipeline.Tokens
	Language          = pipeline.Language
	StartOfTranscript = pipeline.StartOfTranscript
	Task              = pipeline.Task
	NoTimestamps      = pipeline.NoTimestamps
	Verbatim          = pipeline.Verbatim
	Intended          = pipeline.Intended
)
