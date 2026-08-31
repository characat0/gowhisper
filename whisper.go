// Package gowhisper is the public API for the Whisper speech-to-text pipeline:
// it wires the mel front-end, ONNX encoder/decoder, and tokenizer into a single
// Pipeline that turns 16 kHz mono PCM into a transcript.
package gowhisper

import (
	"cmp"
	"context"
	"fmt"
	"log/slog"

	"gowhisper/internal/mel"
	"gowhisper/pkg/model"
)

type Pipeline struct {
	encoder   *model.Encoder
	decoder   *model.Decoder
	tokenizer *model.Tokenizer
}

func NewPipeline(
	encoderPath,
	decoderFirstModelPath,
	decoderWithModelPath,
	tokenizerPath string,
	nMels int,
) (*Pipeline, error) {
	encoder, err := model.NewEncoder(encoderPath, nMels)
	if err != nil {
		return nil, err
	}
	decoder, err := model.NewDecoder(decoderFirstModelPath, decoderWithModelPath)
	if err != nil {
		return nil, err
	}
	tokenizer, err := model.NewTokenizer(tokenizerPath)
	if err != nil {
		return nil, err
	}
	if tokenizer.VocabSize() != decoder.VocabSize {
		return nil, fmt.Errorf("pipeline: vocab size mismatch, decoder has %d and tokenizer has %d", decoder.VocabSize, tokenizer.VocabSize())
	}

	return &Pipeline{
		encoder,
		decoder,
		tokenizer,
	}, nil
}

func (p *Pipeline) Process(audio []float32) (string, error) {
	ctx := context.TODO()
	audio = mel.PadOrTrim(audio, mel.NSamples)
	audio = mel.LogMelSpectrogram(audio, p.encoder.NMels, 0)
	hState, err := p.encoder.Encode(ctx, audio)
	if err != nil {
		return "", err
	}
	prompt := []int64{
		p.tokenizer.Tokens.StartOfTranscript,
		p.tokenizer.MustLang("en"),
		p.tokenizer.Tokens.Transcribe,
		p.tokenizer.Tokens.NoTimestamps,
	}
	logits, kvcache, err := p.decoder.FirstPass(ctx, prompt, hState)
	if err != nil {
		return "", err
	}
	hState.Destroy()
	// logits tensor is (b, N, T), with N being the number of tokens in the prompt and T the vocab size
	start := (0*int64(len(prompt)) + (int64(len(prompt)) - 1)) * p.decoder.VocabSize
	end := start + p.decoder.VocabSize
	lastToken := argMax(logits.GetData()[start:end])
	logits.Destroy()
	inferenceTokens := []int64{}
	for lastToken != p.tokenizer.Tokens.EndOfText && len(inferenceTokens) < 448 {
		inferenceTokens = append(inferenceTokens, lastToken)

		logits, kvcache, err = p.decoder.Step(ctx, lastToken, kvcache)
		if err != nil {
			return "", err
		}
		// logits here should be of shape (1, 1, T)
		lastToken = argMax(logits.GetData())
		logits.Destroy()
	}
	kvcache.Destroy()

	slog.Debug(fmt.Sprintf("inference tokens: %v", inferenceTokens))
	transcript := p.tokenizer.Decode(inferenceTokens)
	return transcript, nil
}

func argMax[T cmp.Ordered](arr []T) int64 {
	// breaks if the first element is NaN
	maxVal, maxIdx := arr[0], int64(0)
	for i, v := range arr {
		if v > maxVal {
			maxVal, maxIdx = v, int64(i)
		}
	}
	return maxIdx
}
