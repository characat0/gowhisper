// Command record captures 5 seconds of microphone audio and writes it to a
// WAV file. It's a smoke test for internal/audio.CaptureAudio.
package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/characat0/gowhisper/internal/audio"
	"github.com/characat0/gowhisper/internal/mel"
)

const outPath = "recording.wav"

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	fmt.Println("Recording for 5 seconds...")
	samples, err := audio.CaptureAudio(ctx)
	if err != nil {
		log.Fatalf("capture: %v", err)
	}
	fmt.Printf("Captured %d samples (%.2fs at %d Hz)\n",
		len(samples), float64(len(samples))/float64(mel.SampleRate), mel.SampleRate)

	if err := writeWAV(outPath, samples, mel.SampleRate, 1); err != nil {
		log.Fatalf("write wav: %v", err)
	}
	fmt.Printf("Wrote %s\n", outPath)
}

// writeWAV encodes float32 PCM in [-1, 1] as a 16-bit mono/stereo PCM WAV file.
func writeWAV(path string, samples []float32, sampleRate, channels int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	const bitsPerSample = 16
	blockAlign := channels * bitsPerSample / 8
	byteRate := sampleRate * blockAlign
	dataSize := len(samples) * bitsPerSample / 8

	// RIFF header + fmt chunk + data chunk header.
	if _, err := f.WriteString("RIFF"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(36+dataSize)); err != nil {
		return err
	}
	if _, err := f.WriteString("WAVEfmt "); err != nil {
		return err
	}
	for _, v := range []any{
		uint32(16),            // fmt chunk size
		uint16(1),             // audio format: PCM
		uint16(channels),      //
		uint32(sampleRate),    //
		uint32(byteRate),      //
		uint16(blockAlign),    //
		uint16(bitsPerSample), //
	} {
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	if _, err := f.WriteString("data"); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(dataSize)); err != nil {
		return err
	}

	// Samples: clamp to [-1, 1] and scale to int16.
	buf := make([]byte, 2*len(samples))
	for i, s := range samples {
		if s > 1 {
			s = 1
		} else if s < -1 {
			s = -1
		}
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(int16(s*32767)))
	}
	_, err = f.Write(buf)
	return err
}
