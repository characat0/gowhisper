package mel

import (
	"fmt"
	"math"
	"sync"
)


const (
	melFmin = 0.0
	melFmax = float64(SampleRate) / 2 // 8000
)

// Slaney mel-scale breakpoints (see librosa.core.convert).
const (
	slaneyFmin     = 0.0
	slaneyFsp      = 200.0 / 3.0 // Hz per mel below minLogHz
	slaneyMinLogHz = 1000.0
)

// slaneyMinLogMel and slaneyLogstep are derived once from the constants above.
var (
	slaneyMinLogMel = (slaneyMinLogHz - slaneyFmin) / slaneyFsp // 15.0
	slaneyLogstep   = math.Log(6.4) / 27.0
)

// hzToMel converts a frequency in Hz to the Slaney mel scale.
func hzToMel(freq float64) float64 {
	mels := (freq - slaneyFmin) / slaneyFsp
	if freq >= slaneyMinLogHz {
		mels = slaneyMinLogMel + math.Log(freq/slaneyMinLogHz)/slaneyLogstep
	}
	return mels
}

// melToHz converts a Slaney-scale mel value back to Hz.
func melToHz(mel float64) float64 {
	freq := slaneyFmin + slaneyFsp*mel
	if mel >= slaneyMinLogMel {
		freq = slaneyMinLogHz * math.Exp(slaneyLogstep*(mel-slaneyMinLogMel))
	}
	return freq
}

var (
	filtersMu    sync.Mutex
	filtersCache = map[int][][]float64{}
)

// MelFilters returns the [nMels][NFreqs] Slaney mel filterbank matrix used to
// project a power spectrogram onto the mel scale. Results are cached per nMels.
// Only nMels 80 and 128 are used by Whisper, but any positive value is accepted.
func MelFilters(nMels int) [][]float64 {
	if nMels <= 0 {
		panic(fmt.Sprintf("mel: nMels must be positive, got %d", nMels))
	}
	filtersMu.Lock()
	defer filtersMu.Unlock()
	if f, ok := filtersCache[nMels]; ok {
		return f
	}
	f := computeMelFilters(nMels)
	filtersCache[nMels] = f
	return f
}

func computeMelFilters(nMels int) [][]float64 {
	// FFT bin center frequencies: linspace(0, sr/2, 1+n_fft/2).
	fftFreqs := make([]float64, NFreqs)
	for i := range fftFreqs {
		fftFreqs[i] = float64(i) * (float64(SampleRate) / 2) / float64(NFreqs-1)
	}

	// Mel band edges: nMels+2 points evenly spaced on the mel scale, back to Hz.
	melMin := hzToMel(melFmin)
	melMax := hzToMel(melFmax)
	melF := make([]float64, nMels+2)
	for i := range melF {
		m := melMin + (melMax-melMin)*float64(i)/float64(nMels+1)
		melF[i] = melToHz(m)
	}

	fdiff := make([]float64, nMels+1)
	for i := range fdiff {
		fdiff[i] = melF[i+1] - melF[i]
	}

	weights := make([][]float64, nMels)
	for i := 0; i < nMels; i++ {
		row := make([]float64, NFreqs)
		// Slaney normalization: 2 / (mel_f[i+2] - mel_f[i]).
		enorm := 2.0 / (melF[i+2] - melF[i])
		for j := 0; j < NFreqs; j++ {
			lower := (fftFreqs[j] - melF[i]) / fdiff[i]
			upper := (melF[i+2] - fftFreqs[j]) / fdiff[i+1]
			w := math.Min(lower, upper)
			if w < 0 {
				w = 0
			}
			row[j] = w * enorm
		}
		weights[i] = row
	}
	return weights
}
