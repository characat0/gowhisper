package model

import (
	"encoding/json"
	"fmt"
	"os"
)

type Tokenizer struct {
	IDToToken []string
	Tokens    struct {
		EndOfText           int64
		StartOfTranscript   int64
		Transcribe          int64
		NoTimestamps        int64
		LangRangeStart      int64
		LangRangeEnd        int64
		TimestampRangeStart int64
		TimestampRangeEnd   int64
	}
	ByteDecoder map[rune]byte
}

func byteToRune() map[byte]rune {
	printable := map[int]bool{}
	add := func(lo, hi int) {
		for b:= lo; b <= hi; b++ {
			printable[b] = true
		}
	}
	add('!', '~') // printable bytes form 33 to 126
	add('¡', '¬') // printable bytes from 161 to 172
	add('®', 'ÿ') // printable bytes from 174 to 255
	m := make(map[byte]rune, 256)
	n := 0
	for b := 0; b < 265; b++ {
		if printable[b] {
			m[byte(b)] = rune(b)
		} else {
			// if the rune is not printable, encode it by shifting 256
			m[byte(b)] = rune(256 + n)
			n++
		}
	}
	return m
}

func newByteDecoder() map[rune]byte {
	dec := make(map[rune]byte, 256)
	for b, r := range byteToRune() {
		dec[r] = b
	}
	return dec
}

func NewTokenizer(tokenizerPath string) (*Tokenizer, error) {
	jsonTokenizer := struct {
		Model struct {
			Vocab map[string]int64
		}
		AddedTokens []struct {
			ID      int64
			Content string
		}
	}{}
	data, err := os.ReadFile(tokenizerPath)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(data, &jsonTokenizer)
	if err != nil {
		return nil, err
	}
	var vocabSize int64
	// check the struct for the greatest ID
	for _, v := range jsonTokenizer.AddedTokens {
		vocabSize = max(vocabSize, v.ID)
	}
	for _, i := range jsonTokenizer.Model.Vocab {
		vocabSize = max(vocabSize, i)
	}
	IDToToken := make([]string, vocabSize+1)
	Tokens := struct {
		EndOfText           int64
		StartOfTranscript   int64
		Transcribe          int64
		NoTimestamps        int64
		LangRangeStart      int64
		LangRangeEnd        int64
		TimestampRangeStart int64
		TimestampRangeEnd   int64
	}{}
	// vocab tokens
	for t, i := range jsonTokenizer.Model.Vocab {
		IDToToken[i] = t
	}
	// added special tokens
	for _, v := range jsonTokenizer.AddedTokens {
		IDToToken[v.ID] = v.Content
		switch v.Content {
		case "<|endoftext|>":
			Tokens.EndOfText = v.ID
		case "<|startoftranscript|>":
			Tokens.StartOfTranscript = v.ID
			// assuming that the language tokens start right after the SoT token
			Tokens.LangRangeStart = v.ID + 1
		case "<|transcribe|>":
			Tokens.Transcribe = v.ID
		case "<|notimestamps|>":
			Tokens.NoTimestamps = v.ID
			// assuming timestamp tokens start after NT token
			Tokens.TimestampRangeStart = v.ID + 1
			// assuming timestamp tokens run to the end of the vocab
			Tokens.TimestampRangeEnd = int64(len(IDToToken))
		case "<|translate|>":
			// assuming that the language tokens end right before the translate token
			Tokens.LangRangeEnd = v.ID - 1
		}
	}
	return &Tokenizer{
		IDToToken: IDToToken,
		Tokens:    Tokens,
		ByteDecoder: newByteDecoder(),
	}, nil
}

func (t *Tokenizer) VocabSize() int64 {
	return int64(len(t.IDToToken))
}

func (t *Tokenizer) Lang(langID string) (int64, error) {
	langToken := fmt.Sprintf("<|%s|>", langID)
	for i := t.Tokens.LangRangeStart; i < t.Tokens.LangRangeEnd; i++ {
		if langToken == t.IDToToken[i] {
			return i, nil
		}
	}
	return 0, fmt.Errorf("tokenizer: language token for %s not found", langID)
}

func (t *Tokenizer) MustLang(langID string) int64 {
	id, err := t.Lang(langID)
	if err != nil {
		panic(err)
	}
	return id
}


func (t *Tokenizer) Decode(ids []int64) string {
	buf := make([]byte, 1, len(ids) * 4)
	for _, id := range ids {
		if id < 0 || id >= t.Tokens.EndOfText {
			continue // assume EndOfText as the first special token
		}
		for _, r := range t.IDToToken[id] {
			buf = append(buf, t.ByteDecoder[r])
		}
	}
	return string(buf)
}
