package model

import (
	"context"
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type Encoder struct {
	session *ort.DynamicAdvancedSession
	NMels   int
	mu      sync.Mutex
}

func NewEncoder(modelPath string, nMels int) (*Encoder, error) {
	session, err := ort.NewDynamicAdvancedSession(
		modelPath, []string{"input_features"}, []string{"last_hidden_state"}, nil,
	)
	if err != nil {
		return nil, err
	}

	// assuming shape (batch_size, feature_size, encoder_sequence_length)
	return &Encoder{
		session: session,
		NMels:   nMels,
	}, nil
}

func (e *Encoder) Encode(ctx context.Context, features []float32) (*ort.Tensor[float32], error) {
	if len(features) == 0 || len(features)%int(e.NMels) != 0 {
		return nil, fmt.Errorf("model: %d features not divisible by nMels=%d", len(features), e.NMels)
	}
	seqLen := len(features) / int(e.NMels)

	inputShape := ort.NewShape(1, int64(e.NMels), int64(seqLen))
	inputTensor, err := ort.NewTensor(inputShape, features)
	if err != nil {
		return nil, fmt.Errorf("model: create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// A nil output slot tells ORT to allocate last_hidden_state itself, so we
	// don't need to know the encoder's output shape ahead of time.
	outputs := make([]ort.Value, 1)

	e.mu.Lock()
	err = e.session.Run([]ort.Value{inputTensor}, outputs)
	e.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("model: encoder run: %w", err)
	}

	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		if outputs[0] != nil {
			outputs[0].Destroy()
		}
		return nil, fmt.Errorf("model: unexpected output type %T, want *Tensor[float32]", outputs[0])
	}
	return out, nil
}
