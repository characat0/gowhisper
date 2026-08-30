package mel

import (
	"math"
	"sync"
)

// realSTFT computes the one-sided DFT of a real, already-windowed length-NFFT
// frame, returning the NFreqs (= NFFT/2+1) complex bins.
//
// NFFT (400) is not a power of two, so we use Bluestein's algorithm (a chirp-z
// transform) which expresses a length-N DFT as a convolution that can be carried
// out with a power-of-two FFT. All the size-dependent factors are precomputed
// once and reused across every frame.
type bluestein struct {
	n    int          // transform length (NFFT)
	m    int          // convolution FFT size, power of two >= 2n-1
	w    []complex128 // chirp: w[k] = exp(-i*pi*k^2/n), length n
	bfft []complex128 // FFT of the zero-padded conjugate chirp, length m
}

var (
	bluesteinOnce sync.Once
	bluestein400  *bluestein
)

// fft400 returns the shared Bluestein plan for the NFFT-point transform.
func fft400() *bluestein {
	bluesteinOnce.Do(func() {
		bluestein400 = newBluestein(NFFT)
	})
	return bluestein400
}

func newBluestein(n int) *bluestein {
	// Smallest power of two >= 2n-1 gives linear (non-circular) convolution.
	m := 1
	for m < 2*n-1 {
		m <<= 1
	}

	w := make([]complex128, n)
	for k := 0; k < n; k++ {
		// Use k^2 mod 2n to keep the angle small and accurate.
		ang := -math.Pi * float64((k*k)%(2*n)) / float64(n)
		w[k] = complex(math.Cos(ang), math.Sin(ang))
	}

	// b[k] = conj(w[k]) for k in [0,n), mirrored, zero-padded to m; then FFT it.
	b := make([]complex128, m)
	b[0] = cmplxConj(w[0])
	for k := 1; k < n; k++ {
		c := cmplxConj(w[k])
		b[k] = c
		b[m-k] = c
	}
	fftInPlace(b, false)

	return &bluestein{n: n, m: m, w: w, bfft: b}
}

// transform returns the full length-n DFT of a real input frame.
func (bs *bluestein) transform(frame []float64) []complex128 {
	n, m := bs.n, bs.m
	a := make([]complex128, m)
	for k := 0; k < n; k++ {
		a[k] = complex(frame[k], 0) * bs.w[k]
	}
	fftInPlace(a, false)
	for i := 0; i < m; i++ {
		a[i] *= bs.bfft[i]
	}
	fftInPlace(a, true) // inverse

	out := make([]complex128, n)
	for k := 0; k < n; k++ {
		out[k] = a[k] * bs.w[k]
	}
	return out
}

// powerSpectrum returns the one-sided power spectrum (|X|^2) of a windowed
// real frame: NFreqs values for bins 0..NFFT/2.
func (bs *bluestein) powerSpectrum(frame []float64, out []float64) {
	x := bs.transform(frame)
	for k := 0; k < NFreqs; k++ {
		re, im := real(x[k]), imag(x[k])
		out[k] = re*re + im*im
	}
}

func cmplxConj(c complex128) complex128 {
	return complex(real(c), -imag(c))
}

// fftInPlace is an iterative radix-2 Cooley-Tukey FFT. len(x) must be a power
// of two. When inverse is true it computes the inverse transform (with 1/N
// scaling).
func fftInPlace(x []complex128, inverse bool) {
	n := len(x)
	// Bit-reversal permutation.
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}

	sign := -1.0
	if inverse {
		sign = 1.0
	}
	for length := 2; length <= n; length <<= 1 {
		ang := sign * 2 * math.Pi / float64(length)
		wlen := complex(math.Cos(ang), math.Sin(ang))
		for i := 0; i < n; i += length {
			w := complex(1, 0)
			half := length >> 1
			for k := 0; k < half; k++ {
				u := x[i+k]
				v := x[i+k+half] * w
				x[i+k] = u + v
				x[i+k+half] = u - v
				w *= wlen
			}
		}
	}

	if inverse {
		inv := complex(1/float64(n), 0)
		for i := range x {
			x[i] *= inv
		}
	}
}
