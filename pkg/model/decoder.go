package model

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type Decoder struct {
	firstPassSession     *ort.DynamicAdvancedSession
	recurrentPassSession *ort.DynamicAdvancedSession
	VocabSize            int64
	NumLayers            int
	mu                   sync.Mutex
}

// presentKeyRe matches ONNX KV-cache output names like "present.11.decoder.key",
// capturing the layer index so we can count decoder layers from the graph metadata.
var presentKeyRe = regexp.MustCompile(`^present\.(\d+)\.decoder\.key$`)

// This only supports a subset of Whisper models
func NewDecoder(firstPassModelPath, recurrentPassModelPath string) (*Decoder, error) {
	// The number of decoder layers and the vocab size vary by Whisper size
	// (base=6, small=12, medium=24, ...); infer both from the recurrent model's
	// I/O metadata rather than hardcoding them, so any exported Whisper works.
	_, outputInfo, err := ort.GetInputOutputInfo(recurrentPassModelPath)
	if err != nil {
		return nil, err
	}
	vocabSize := int64(0)
	numLayers := 0
	for _, o := range outputInfo {
		if o.Name == "logits" {
			// assuming the third dimension is vocab size and fixed
			vocabSize = o.Dimensions[2]
			continue
		}
		if m := presentKeyRe.FindStringSubmatch(o.Name); m != nil {
			// layer indices are 0-based, so the count is max index + 1
			idx, _ := strconv.Atoi(m[1])
			numLayers = max(numLayers, idx + 1)
		}
	}
	if vocabSize == 0 {
		return nil, fmt.Errorf("decoder: cannot infer vocab size from onnx model metadata")
	}
	if numLayers == 0 {
		return nil, fmt.Errorf("decoder: cannot infer decoder layer count from onnx model metadata")
	}

	// first pass model setup
	firstPassOutputNames := []string{"logits"}
	for i := range numLayers {
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

	// recurrent model setup
	recurrentPassInputNames := []string{"input_ids"}
	recurrentPassOutputNames := []string{"logits"}
	for i := range numLayers {
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
		VocabSize:            vocabSize,
		NumLayers:            numLayers,
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

func NewKVCacheFromArrays[T ort.TensorData](encoderKeys, encoderValues, decoderKeys, decoderValues []*ort.Tensor[T]) KVCache[T] {
	layers := []kvcachelayer[T]{}

	for i := range len(encoderKeys) {
		layers = append(layers, kvcachelayer[T]{
			Encoder: kv[T]{
				Key:   encoderKeys[i],
				Value: encoderValues[i],
			},
			Decoder: kv[T]{
				Key:   decoderKeys[i],
				Value: decoderValues[i],
			},
		})
	}

	return KVCache[T]{
		Layers: layers,
	}
}

func (cache *KVCache[T]) UpdateDecoderArrays(decoderKeys, decoderValues []*ort.Tensor[T]) error {
	if len(decoderKeys) != len(cache.Layers) {
		return fmt.Errorf("decoder: size mismatch while updating decoder arrays, encountered %d but expected %d", len(cache.Layers), len(decoderKeys))
	}
	for i := range len(decoderKeys) {
		d := cache.Layers[i].Decoder
		d.Key.Destroy()
		d.Value.Destroy()
		cache.Layers[i].Decoder = kv[T]{
			Key:   decoderKeys[i],
			Value: decoderValues[i],
		}
	}
	return nil
}

func (cache *KVCache[T]) ToValueArray() []ort.Value {
	arr := make([]ort.Value, 0, len(cache.Layers)*2*2)
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

func (cache *KVCache[T]) Destroy() (err error) {
	for _, l := range cache.Layers {
		err = l.Decoder.Key.Destroy()
		if err != nil {
			return
		}
		err = l.Decoder.Value.Destroy()
		if err != nil {
			return
		}
		err = l.Encoder.Key.Destroy()
		if err != nil {
			return
		}
		err = l.Encoder.Value.Destroy()
		if err != nil {
			return
		}
	}
	return nil
}

func (d *Decoder) FirstPass(ctx context.Context, prompt []int64, hiddenState *ort.Tensor[float32]) (*ort.Tensor[float32], *KVCache[float32], error) {
	inputIDsTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(prompt))), prompt)
	if err != nil {
		return nil, nil, err
	}
	defer inputIDsTensor.Destroy()

	outputs := make([]ort.Value, 1+d.NumLayers*2*2) // per layer: decoder+encoder, key+value
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
	decoderKeys := slice(outputTensors, 1, len(outputTensors), 4)
	decoderValues := slice(outputTensors, 2, len(outputTensors), 4)
	encoderKeys := slice(outputTensors, 3, len(outputTensors), 4)
	encoderValues := slice(outputTensors, 4, len(outputTensors), 4)
	kvcache := NewKVCacheFromArrays(
		encoderKeys,
		encoderValues,
		decoderKeys,
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
	outputs := make([]ort.Value, 1+d.NumLayers*2*1) // per layer: decoder only, key+value
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
