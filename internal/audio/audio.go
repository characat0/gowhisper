package audio

import (
	"context"
	"encoding/binary"
	"github.com/characat0/gowhisper/internal/mel"
	"log/slog"
	"math"

	"github.com/gen2brain/malgo"
)

func BytesFloat32(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}

func GetFloatArray(aBytes []byte) []float32 {
	n := len(aBytes) / 4 // each float32 sample is 4 bytes
	aArr := make([]float32, n)

	for i := range n {
		aArr[i] = BytesFloat32(aBytes[i*4:])
	}
	return aArr
}

func CaptureAudio(ctx context.Context) ([]float32, error) {
	mCtx, err := malgo.InitContext(nil, malgo.ContextConfig{}, func(message string) {
		slog.Default().Debug("miniaudio: " + message)
	})
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = mCtx.Uninit()
		mCtx.Free()
	}()

	deviceConfig := malgo.DefaultDeviceConfig(malgo.Capture)
	deviceConfig.Capture.Format = malgo.FormatF32
	deviceConfig.Capture.Channels = 1
	deviceConfig.SampleRate = mel.SampleRate
	deviceConfig.Alsa.NoMMap = 1

	pCapturedF32Samples := make([]float32, 0)

	captureCallbacks := malgo.DeviceCallbacks{
		Data: func(_, pSample []byte, _ uint32) {
			pCapturedF32Samples = append(pCapturedF32Samples, GetFloatArray(pSample)...)
		},
	}
	device, err := malgo.InitDevice(mCtx.Context, deviceConfig, captureCallbacks)
	if err != nil {
		return nil, err
	}
	defer device.Uninit()

	if err := device.Start(); err != nil {
		return nil, err
	}

	<-ctx.Done()

	if err := device.Stop(); err != nil {
		return nil, err
	}

	return pCapturedF32Samples, nil
}
