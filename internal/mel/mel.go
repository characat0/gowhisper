package mel

import "math"

const (
	SampleRate  = 16000
	NFFT        = 400
	HopLength   = 160
	ChunkLength = 30
	NSamples    = ChunkLength * SampleRate // 480000 samples per 30s chunk
	NFrames     = NSamples / HopLength     // 3000 frames per chunk
	NFreqs      = NFFT/2 + 1               // 201 one-sided FFT bins
)

// PadOrTrim right-pads audio with zeros or trims it to exactly length samples,
// matching Whisper's pad_or_trim. The input slice is never mutated.
func PadOrTrim(audio []float32, length int) []float32 {
	out := make([]float32, length)
	copy(out, audio) // copy stops at min(len(audio), length); the rest stays zero
	return out
}

// hannWindow holds the periodic Hann window of length NFFT, matching
// torch.hann_window(NFFT) (periodic=True): w[n] = 0.5*(1 - cos(2*pi*n/N)).
var hannWindow = func() []float64 {
	w := make([]float64, NFFT)
	for n := range w {
		w[n] = 0.5 * (1 - math.Cos(2*math.Pi*float64(n)/float64(NFFT)))
	}
	return w
}()

// LogMelSpectrogram computes the Whisper log-Mel spectrogram of audio.
//
// nMels must be 80 or 128 for Whisper models (any positive value is accepted).
// padding right-pads the audio with that many zero samples before the STFT, as
// Whisper does when preparing a final partial chunk.
func LogMelSpectrogram(audio []float32, nMels, padding int) []float32 {
	// Center padding (torch.stft center=True) reflect-pads NFFT/2 on both sides.
	pad := NFFT / 2
	signal := reflectPadCenter(audio, padding, pad)

	// torch.stft yields floor((len-NFFT)/HopLength)+1 frames; Whisper then drops
	// the last (magnitudes = stft[..., :-1]), leaving (len-NFFT)/HopLength.
	nFrames := (len(signal) - NFFT) / HopLength
	if nFrames < 0 {
		nFrames = 0
	}

	filters := MelFilters(nMels)
	plan := fft400()

	frame := make([]float64, NFFT)
	power := make([]float64, NFreqs)

	// mel is the flat [nMels*nFrames] output (index m*nFrames+t); melLog keeps the
	// pre-normalized log values so the global-max floor can be applied afterwards.
	mel := make([]float32, nMels*nFrames)
	melLog := make([]float64, nMels*nFrames)

	globalMax := math.Inf(-1)
	for t := 0; t < nFrames; t++ {
		start := t * HopLength
		for i := 0; i < NFFT; i++ {
			frame[i] = signal[start+i] * hannWindow[i]
		}
		plan.powerSpectrum(frame, power)

		for m := 0; m < nMels; m++ {
			var acc float64
			fm := filters[m]
			for f := 0; f < NFreqs; f++ {
				acc += fm[f] * power[f]
			}
			// clamp(min=1e-10).log10()
			if acc < 1e-10 {
				acc = 1e-10
			}
			v := math.Log10(acc)
			melLog[m*nFrames+t] = v
			if v > globalMax {
				globalMax = v
			}
		}
	}

	// log_spec = maximum(log_spec, log_spec.max()-8); log_spec = (log_spec+4)/4.
	floor := globalMax - 8.0
	for i, v := range melLog {
		if v < floor {
			v = floor
		}
		mel[i] = float32((v + 4.0) / 4.0)
	}
	return mel
}

// reflectPadCenter returns a float64 signal: the input audio, optionally
// right-padded with rightZeros zeros (Whisper's `padding`), then reflect-padded
// by pad samples on both sides. Reflection excludes the boundary sample, matching
// numpy's pad(mode="reflect") and torch.stft center padding:
// [a,b,c,d] padded by 2 each side -> [c,b,a,b,c,d,c,b].
func reflectPadCenter(audio []float32, rightZeros, pad int) []float64 {
	body := len(audio) + rightZeros
	out := make([]float64, body+2*pad)
	for j := range out {
		out[j] = bodyReflect(audio, body, j-pad)
	}
	return out
}

// bodyReflect reads index i of the (audio ++ zeros) body of length bodyLen as
// float64, reflecting indices outside [0, bodyLen) with numpy "reflect"
// semantics (period 2*(bodyLen-1), boundaries not repeated).
func bodyReflect(audio []float32, bodyLen, i int) float64 {
	if bodyLen <= 1 {
		if i == 0 && len(audio) > 0 {
			return float64(audio[0])
		}
		return 0
	}
	period := 2 * (bodyLen - 1)
	i %= period
	if i < 0 {
		i += period
	}
	if i >= bodyLen {
		i = period - i
	}
	if i < len(audio) {
		return float64(audio[i])
	}
	return 0 // within the right zero-pad region
}
