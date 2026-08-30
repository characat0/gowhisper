package mel

import (
	"archive/zip"
	"encoding/binary"
	"math"
	"regexp"
	"strconv"
	"testing"
)

// TestMelFiltersMatchWhisper checks that our Go-computed Slaney filterbank
// reproduces OpenAI Whisper's precomputed assets/mel_filters.npz to float
// precision, for both supported band counts.
func TestMelFiltersMatchWhisper(t *testing.T) {
	for _, nMels := range []int{80, 128} {
		want := readNPZFilters(t, "mel_"+strconv.Itoa(nMels))
		got := MelFilters(nMels)

		if len(got) != len(want) {
			t.Fatalf("nMels=%d: got %d rows, want %d", nMels, len(got), len(want))
		}
		var maxDiff float64
		for m := range want {
			if len(got[m]) != len(want[m]) {
				t.Fatalf("nMels=%d row %d: got %d cols, want %d", nMels, m, len(got[m]), len(want[m]))
			}
			for f := range want[m] {
				d := math.Abs(got[m][f] - float64(want[m][f]))
				if d > maxDiff {
					maxDiff = d
				}
			}
		}
		const tol = 1e-6
		if maxDiff > tol {
			t.Errorf("nMels=%d: max filter diff %g exceeds tol %g", nMels, maxDiff, tol)
		} else {
			t.Logf("nMels=%d: max filter diff %g (tol %g)", nMels, maxDiff, tol)
		}
	}
}

// TestLogMelSpectrogramShapeAndRange runs the full pipeline on a synthetic
// 30-second signal and checks the output shape and Whisper's normalization
// invariants for both band counts.
func TestLogMelSpectrogramShapeAndRange(t *testing.T) {
	// 440 Hz tone for the first half, silence for the second half.
	audio := make([]float32, NSamples)
	for i := range audio {
		if i < NSamples/2 {
			audio[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/float64(SampleRate)))
		}
	}

	for _, nMels := range []int{80, 128} {
		mel := LogMelSpectrogram(audio, nMels, 0)
		if len(mel) != nMels*NFrames {
			t.Fatalf("nMels=%d: got %d values, want %d", nMels, len(mel), nMels*NFrames)
		}
		// nFrames is recovered from the flat length exactly as Run does it.
		if nFrames := len(mel) / nMels; nFrames != NFrames {
			t.Fatalf("nMels=%d: inferred %d frames, want %d", nMels, nFrames, NFrames)
		}

		vmax := math.Inf(-1)
		vmin := math.Inf(1)
		for _, v := range mel {
			fv := float64(v)
			if math.IsNaN(fv) || math.IsInf(fv, 0) {
				t.Fatalf("nMels=%d: non-finite value %v", nMels, fv)
			}
			if fv > vmax {
				vmax = fv
			}
			if fv < vmin {
				vmin = fv
			}
		}
		// Whisper floors log_spec at max-8 then applies (x+4)/4, so values lie in
		// [(M-4)/4, (M+4)/4] for M = log_spec.max(): the span is at most 2.0, and
		// exactly 2.0 once any bin hits the floor (guaranteed here by the silence).
		if span := vmax - vmin; math.Abs(span-2.0) > 1e-5 {
			t.Errorf("nMels=%d: value span %g, want 2.0", nMels, span)
		}
		t.Logf("nMels=%d: min=%g max=%g", nMels, vmin, vmax)
	}
}

// TestPadOrTrim covers both the padding and trimming branches.
func TestPadOrTrim(t *testing.T) {
	in := []float32{1, 2, 3}
	if got := PadOrTrim(in, 5); len(got) != 5 || got[2] != 3 || got[3] != 0 || got[4] != 0 {
		t.Errorf("pad: got %v", got)
	}
	if got := PadOrTrim(in, 2); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Errorf("trim: got %v", got)
	}
}

var npyShapeRe = regexp.MustCompile(`'shape':\s*\((\d+),\s*(\d+)\)`)

// readNPZFilters reads a 2-D float32 array named <name>.npy from the whisper
// mel_filters.npz asset in testdata.
func readNPZFilters(t *testing.T, name string) [][]float32 {
	t.Helper()
	zr, err := zip.OpenReader("testdata/mel_filters.npz")
	if err != nil {
		t.Fatalf("open npz: %v (run: curl -sL -o internal/mel/testdata/mel_filters.npz "+
			"https://github.com/openai/whisper/raw/main/whisper/assets/mel_filters.npz)", err)
	}
	defer zr.Close()

	for _, f := range zr.File {
		if f.Name != name+".npy" {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			t.Fatalf("open %s: %v", f.Name, err)
		}
		defer rc.Close()
		raw := make([]byte, f.UncompressedSize64)
		if _, err := readFull(rc, raw); err != nil {
			t.Fatalf("read %s: %v", f.Name, err)
		}
		return parseNPY(t, raw)
	}
	t.Fatalf("%s.npy not found in npz", name)
	return nil
}

func parseNPY(t *testing.T, raw []byte) [][]float32 {
	t.Helper()
	if len(raw) < 10 || string(raw[:6]) != "\x93NUMPY" {
		t.Fatalf("bad npy magic")
	}
	if raw[6] != 1 {
		t.Fatalf("unsupported npy major version %d", raw[6])
	}
	hdrLen := int(binary.LittleEndian.Uint16(raw[8:10]))
	header := string(raw[10 : 10+hdrLen])
	if !regexpContains(header, "'<f4'") {
		t.Fatalf("expected <f4 dtype, header: %s", header)
	}
	mm := npyShapeRe.FindStringSubmatch(header)
	if mm == nil {
		t.Fatalf("cannot parse shape from header: %s", header)
	}
	rows, _ := strconv.Atoi(mm[1])
	cols, _ := strconv.Atoi(mm[2])

	data := raw[10+hdrLen:]
	if len(data) < rows*cols*4 {
		t.Fatalf("npy data too short: have %d, want %d", len(data), rows*cols*4)
	}
	out := make([][]float32, rows)
	idx := 0
	for r := 0; r < rows; r++ {
		out[r] = make([]float32, cols)
		for c := 0; c < cols; c++ {
			out[r][c] = math.Float32frombits(binary.LittleEndian.Uint32(data[idx : idx+4]))
			idx += 4
		}
	}
	return out
}

func regexpContains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// readFull fills buf, tolerating short reads from the zip reader.
func readFull(r interface{ Read([]byte) (int, error) }, buf []byte) (int, error) {
	total := 0
	for total < len(buf) {
		n, err := r.Read(buf[total:])
		total += n
		if err != nil {
			if total == len(buf) {
				return total, nil
			}
			return total, err
		}
	}
	return total, nil
}
