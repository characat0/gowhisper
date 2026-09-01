package model

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type Tokenizer struct {
	IDToToken []string
	TokenToID map[string]int64
	Tokens    struct {
		EndOfText           int64
		StartOfTranscript   int64
		Transcribe          int64
		NoTimestamps        int64
	}
	ByteDecoder map[rune]byte
}

func byteToRune() map[byte]rune {
	printable := map[int]bool{}
	add := func(lo, hi int) {
		for b := lo; b <= hi; b++ {
			printable[b] = true
		}
	}
	add('!', '~') // printable bytes from 33 to 126
	add('¡', '¬') // printable bytes from 161 to 172
	add('®', 'ÿ') // printable bytes from 174 to 255
	m := make(map[byte]rune, 256)
	n := 0
	for b := 0; b < 256; b++ {
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
			Vocab map[string]int64 `json:"vocab"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int64  `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
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
	TokenToID := map[string]int64{}
	Tokens := struct {
		EndOfText           int64
		StartOfTranscript   int64
		Transcribe          int64
		NoTimestamps        int64
	}{}
	// vocab tokens
	for t, i := range jsonTokenizer.Model.Vocab {
		IDToToken[i] = t
		TokenToID[t] = i
	}
	// added special tokens
	for _, v := range jsonTokenizer.AddedTokens {
		IDToToken[v.ID] = v.Content
		TokenToID[v.Content] = v.ID
	}
	var ok bool
	Tokens.EndOfText, ok = TokenToID["<|endoftext|>"]
	if !ok {
		return nil, fmt.Errorf("token <|endoftext|> not found in vocab")
	}
	Tokens.StartOfTranscript, ok = TokenToID["<|startoftranscript|>"]
	if !ok {
		return nil, fmt.Errorf("token <|startoftranscript|> not found in vocab")
	}

	Tokens.Transcribe, ok = TokenToID["<|transcribe|>"]
	if !ok {
		return nil, fmt.Errorf("token <|transcribe|> not found in vocab")
	}
	Tokens.NoTimestamps, ok = TokenToID["<|notimestamps|>"]
	if !ok {
		return nil, fmt.Errorf("token <|notimestamps|> not found in vocab")
	}
	return &Tokenizer{
		IDToToken:   IDToToken,
		TokenToID: TokenToID,
		Tokens:      Tokens,
		ByteDecoder: newByteDecoder(),
	}, nil
}

func (t *Tokenizer) VocabSize() int64 {
	return int64(len(t.IDToToken))
}

func (t *Tokenizer) LookupToken(tok string) (int64, error) {
	tokID, ok := t.TokenToID[tok]
	if !ok {
		return -1, fmt.Errorf("token %s not found", tok)
	}
	return tokID, nil
}

func (t *Tokenizer) LookupControlToken(tok string) (int64, error) {
	return t.LookupToken(fmt.Sprintf("<|%s|>", tok))
}

func (t *Tokenizer) Lang(langID string) (int64, error) {
	return t.LookupControlToken(langID)
}

func (t *Tokenizer) MustLang(langID string) int64 {
	id, err := t.Lang(langID)
	if err != nil {
		panic(err)
	}
	return id
}

func (t *Tokenizer) ShouldEmit(id int64) bool {
	if id < 0 || id >= int64(len(t.IDToToken)) {
		return false // out of array tokens
	}
	if id < t.Tokens.EndOfText {
		return true // valid vocab tokens
	}
	tok := t.IDToToken[id]
	if len(tok) >= 4 && strings.HasPrefix(tok, "<|") && strings.HasSuffix(tok, "|>") {
		return false // control tokens, like <|transcribe|>
	}
	return true // everything else fail in favor
}

func (t *Tokenizer) Decode(ids []int64) string {
	buf := make([]byte, 0, len(ids)*4)
	for _, id := range ids {
		if !t.ShouldEmit(id) {
			continue // skip token
		}
		for _, r := range t.IDToToken[id] {
			buf = append(buf, t.ByteDecoder[r])
		}
	}
	return string(buf)
}
