// Command transcribe captures microphone audio and runs it through the Whisper
// pipeline, printing the resulting transcript. It's a smoke test for
// pkg/api.Pipeline end to end (encoder + decoder + tokenizer).
//
// Usage:
//
//	go run ./cmd/transcribe [-dur 5s] [-lib /path/to/libonnxruntime.dylib]
//
// The ONNX Runtime shared library is located via the -lib flag or, if unset,
// the ONNXRUNTIME_LIB_PATH environment variable. If neither is provided the
// onnxruntime_go default library name is used.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/characat0/gowhisper"
	"github.com/characat0/gowhisper/internal/audio"
	"github.com/characat0/gowhisper/internal/mel"
)

const (
	encoderPath   = "bin/encoder_model.onnx"
	decoderPath   = "bin/decoder_model.onnx"
	decoderPast   = "bin/decoder_with_past_model.onnx"
	tokenizerPath = "bin/tokenizer.json"
)

func main() {
	dur := flag.Duration("dur", 5*time.Second, "how long to capture microphone audio")
	memprofile := flag.String("memprofile", "", "write a heap profile to this file after transcription")
	cpuprofile := flag.String("cpuprofile", "", "write a CPU profile to this file during transcription")
	defaultLib := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if defaultLib == "" {
		defaultLib = "/opt/homebrew/lib/libonnxruntime.dylib"
	}
	lib := flag.String("lib", defaultLib,
		"path to the ONNX Runtime shared library (falls back to $ONNXRUNTIME_LIB_PATH)")
	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatalf("create cpuprofile: %v", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("start cpuprofile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	if *lib != "" {
		ort.SetSharedLibraryPath(*lib)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("init onnxruntime: %v", err)
	}
	defer ort.DestroyEnvironment()

	pipeline, err := gowhisper.NewPipeline(encoderPath, decoderPath, decoderPast, tokenizerPath, 80)
	if err != nil {
		log.Fatalf("new pipeline: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), *dur)
	defer cancel()

	fmt.Printf("Recording for %s...\n", *dur)
	samples, err := audio.CaptureAudio(ctx)
	if err != nil {
		log.Fatalf("capture: %v", err)
	}
	fmt.Printf("Captured %d samples (%.2fs at %d Hz), transcribing...\n",
		len(samples), float64(len(samples))/float64(mel.SampleRate), mel.SampleRate)

	transcript, err := pipeline.Process(samples)
	if err != nil {
		log.Fatalf("transcribe: %v", err)
	}

	fmt.Printf("\nTranscript:\n%s\n", transcript)

	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatalf("create memprofile: %v", err)
		}
		defer f.Close()
		runtime.GC() // get up-to-date statistics into the heap profile
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatalf("write memprofile: %v", err)
		}
		fmt.Printf("Wrote heap profile to %s\n", *memprofile)
	}
}
