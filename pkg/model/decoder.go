package model

import (
	"context"
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type Decoder struct {
	firstPassSession     *ort.DynamicAdvancedSession
	recurrentPassSession *ort.DynamicAdvancedSession
	mu                   sync.Mutex
}

// This only support a subset of Whisper models
func NewDecoder(firstPassModelPath, recurrentPassModelPath string) (*Decoder, error) {
	firstPassOutputNames := []string{"logits"}
	for i := range 6 {
		// the first pass model outputs both encoder and decoder weights
		// the ordering is intentional to be able to slice them after inference
		firstPassOutputNames = append(
			firstPassOutputNames,
			fmt.Sprintf("present.%d.decoder.key", i),
			fmt.Sprintf("present.%d.decoder.value", i),
			fmt.Sprintf("present.%d.encoder.key", i),
			fmt.Sprintf("present.%d.encoder.value", i),
		)
	}
	firstPassSession, err := ort.NewDynamicAdvancedSession(
		firstPassModelPath,
		[]string{"input_ids", "encoder_hidden_states"},
		firstPassOutputNames,
		nil,
	)
	if err != nil {
		return nil, err
	}
	recurrentPassInputNames := []string{"input_ids"}
	recurrentPassOutputNames := []string{"logits"}
	for i := range 6 {
		// inputs 2-25 are the recurrent decoder and encoder weights updated
		recurrentPassInputNames = append(
			recurrentPassInputNames,
			fmt.Sprintf("past_key_values.%d.decoder.key", i),
			fmt.Sprintf("past_key_values.%d.decoder.value", i),
			fmt.Sprintf("past_key_values.%d.encoder.key", i),
			fmt.Sprintf("past_key_values.%d.encoder.value", i),
		)

		// the recurrent model only outputs the decoder weights
		recurrentPassOutputNames = append(
			recurrentPassOutputNames,
			fmt.Sprintf("present.%d.decoder.key", i),
			fmt.Sprintf("present.%d.decoder.value", i),
		)
	}
	recurrentPassModelSession, err := ort.NewDynamicAdvancedSession(
		recurrentPassModelPath,
		recurrentPassInputNames,
		recurrentPassOutputNames,
		nil,
	)
	if err != nil {
		return nil, err
	}
	return &Decoder{
		firstPassSession:     firstPassSession,
		recurrentPassSession: recurrentPassModelSession,
	}, nil
}

type kv[T ort.TensorData] struct {
	Key   *ort.Tensor[T]
	Value *ort.Tensor[T]
}

type kvcachelayer[T ort.TensorData] struct {
	Encoder kv[T]
	Decoder kv[T]
}

type KVCache[T ort.TensorData] struct {
	Layers []kvcachelayer[T]
}

func NewKVCacheFromArrays[T ort.TensorData](encodersKeys, encodersValues, decodersKeys, decodersValues []*ort.Tensor[T]) KVCache[T] {
	layers := []kvcachelayer[T]{}

	for i := range len(encodersKeys) {
		layers = append(layers, kvcachelayer[T]{
			Encoder: kv[T]{
				Key:   encodersKeys[i],
				Value: encodersValues[i],
			},
			Decoder: kv[T]{
				Key:   decodersKeys[i],
				Value: decodersValues[i],
			},
		})
	}

	return KVCache[T]{
		Layers: layers,
	}
}

func (cache *KVCache[T]) UpdateDecoderArrays(decodersKeys, decodersValues []*ort.Tensor[T]) error {
	if len(decodersKeys) != len(cache.Layers) {
		return fmt.Errorf("decoder: size missmatch while updating decoder arrays, encountered %d but expected %d", len(cache.Layers), len(decodersKeys))
	}
	for i := range len(decodersKeys) {
		d := cache.Layers[i].Decoder
		d.Key.Destroy()
		d.Value.Destroy()
		cache.Layers[i].Decoder = kv[T]{
			Key:   decodersKeys[i],
			Value: decodersValues[i],
		}
	}
	return nil
}

func (cache *KVCache[T]) ToValueArray() []ort.Value {
	arr := make([]ort.Value, 0, len(cache.Layers) * 2 * 2)
	for _, l := range cache.Layers {
		arr = append(arr, 
			l.Decoder.Key,
			l.Decoder.Value,
			l.Encoder.Key,
			l.Encoder.Value,
		)
	}
	return arr
}

func (d *Decoder) FirstPass(ctx context.Context, prompt []int64, hiddenState *ort.Tensor[float32]) (*ort.Tensor[float32], *KVCache[float32], error) {
	inputIDsTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(prompt))), prompt)
	if err != nil {
		return nil, nil, err
	}
	defer inputIDsTensor.Destroy()

	outputs := make([]ort.Value, 1+6*2*2) // 6 layers, decoder-encoder, key-value
	d.mu.Lock()
	err = d.firstPassSession.Run([]ort.Value{inputIDsTensor, hiddenState}, outputs)
	d.mu.Unlock()
	if err != nil {
		return nil, nil, err
	}
	outputTensors := make([]*ort.Tensor[float32], len(outputs))
	for i, v := range outputs {
		t, ok := v.(*ort.Tensor[float32])
		if !ok {
			return nil, nil, fmt.Errorf("decoder: unexpected output type %T, want *Tensor[float32] at index %d", v, i)
		}
		outputTensors[i] = t
	}

	logits := outputTensors[0]
	decodersKeys := slice(outputTensors, 1, len(outputTensors), 4)
	decoderValues := slice(outputTensors, 2, len(outputTensors), 4)
	encoderKeys := slice(outputTensors, 3, len(outputTensors), 4)
	encoderValues := slice(outputTensors, 4, len(outputTensors), 4)
	kvcache := NewKVCacheFromArrays(
		encoderKeys,
		encoderValues,
		decodersKeys,
		decoderValues,
	)
	return logits, &kvcache, nil
}

func (d *Decoder) Step(ctx context.Context, latestToken int64, cache *KVCache[float32]) (*ort.Tensor[float32], *KVCache[float32], error) {
	inputIDsTensor, err := ort.NewTensor(ort.NewShape(1, 1), []int64{latestToken})
	if err != nil {
		return nil, nil, err
	}
	defer inputIDsTensor.Destroy()
	outputs := make([]ort.Value, 1+6*2*1) // 6 layers, decoder only, key-value
	d.mu.Lock()
	err = d.recurrentPassSession.Run(
		append([]ort.Value{inputIDsTensor}, cache.ToValueArray()...),
		outputs,
	)
	d.mu.Unlock()
	if err != nil {
		return nil, nil, err
	}
	outputTensors := make([]*ort.Tensor[float32], len(outputs))
	for i, v := range outputs {
		t, ok := v.(*ort.Tensor[float32])
		if !ok {
			return nil, nil, fmt.Errorf("decoder: unexpected output type %T, want *Tensor[float32] at index %d", v, i)
		}
		outputTensors[i] = t
	}
	logits := outputTensors[0]
	decoderKeys := slice(outputTensors, 1, len(outputs), 2)
	decoderValues := slice(outputTensors, 2, len(outputs), 2)
	err = cache.UpdateDecoderArrays(decoderKeys, decoderValues)
	if err != nil {
		return nil, nil, err
	}
	return logits, cache, nil
}

func slice[T any](arr []T, start, stop, step int) []T {
	out := make([]T, 0, (stop-start+step-1)/step)
	for i := start; i < stop; i += step {
		out = append(out, arr[i])
	}
	return out
}
