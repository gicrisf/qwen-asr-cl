# qwen-asr-cl

Common Lisp inference engine for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
speech-to-text models (`0.6B` and `1.7B` variants).

## Dependencies

- **SBCL** with [Quicklisp](https://www.quicklisp.org/)
- **lla** — BLAS-backed linear algebra (`(ql:quickload :lla)`)
- **babel** — UTF-8 string/byte encoding
- **cffi** — BF16→F32 reinterpretation
- **com.inuoe.jzon** — JSON parsing for tokenizer/safetensors index

Install deps once:
```lisp
(ql:quickload '(:lla :babel :cffi :com.inuoe.jzon))
```

## Build

```lisp
(asdf:load-system :qwen-asr)
```

Run from the `qwen-asr-cl/` directory (or set ASDF source registry appropriately).

## Quick start

```lisp
(asdf:load-system :qwen-asr)

(defvar *enc* (qwen-asr:load-encoder "/path/to/qwen3-asr-0.6b"))
(defvar *dec* (qwen-asr:load-decoder "/path/to/qwen3-asr-0.6b"))
(defvar *tok* (qwen-asr:load-tokenizer "/path/to/qwen3-asr-0.6b"))

(qwen-asr:transcribe *enc* *dec* *tok* "/path/to/samples/jfk.wav")
;; => "And so my fellow Americans, ask not what your country can do for you, ..."
```

## Public API

### Tokenizer

```lisp
;; Load tokenizer from model directory
(defvar *tok* (qwen-asr:load-tokenizer "/path/to/model-dir"))

;; Vocabulary size
(qwen-asr:tokenizer-vocab-size *tok*)   ; => 151643 or 151936

;; Encode a string to token IDs
(qwen-asr:tokenizer-encode *tok* "hello")  ; => #(15339)

;; Decode a single token ID to string piece
(qwen-asr:tokenizer-decode *tok* 15339)    ; => "hello"
```

### Audio

```lisp
;; Load a 16-bit PCM WAV file => (simple-array single-float (*)) at 16 kHz
(defvar *samples* (qwen-asr:load-wav "/path/to/audio.wav"))

;; Compute log-mel spectrogram => (simple-array single-float (128 n-frames))
(defvar *mel* (qwen-asr:mel-spectrogram *samples*))
```

### Encoder

```lisp
;; Load encoder weights from model directory
(defvar *enc* (qwen-asr:load-encoder "/path/to/model-dir"))

;; Run encoder forward pass
;; Returns (values enc-output n-tokens)
;;   enc-output: flat (simple-array single-float (*)) of shape [n-tokens x output-dim]
(multiple-value-bind (enc-out n-tok)
    (qwen-asr:encoder-forward *enc* *mel* (array-dimension *mel* 1))
  (format t "tokens=~a enc[0]=~a~%" n-tok (aref enc-out 0)))

;; Architecture accessors
(qwen-asr:encoder-d-model *enc*)       ; model dimension (896 or 1024)
(qwen-asr:encoder-n-layers *enc*)      ; 18 or 24
(qwen-asr:encoder-output-dim *enc*)    ; 1024 or 2048
```

### Decoder

```lisp
;; Load decoder weights from model directory
(defvar *dec* (qwen-asr:load-decoder "/path/to/model-dir"))

;; Architecture accessors
(qwen-asr:decoder-hidden *dec*)     ; 1024 (0.6B) or 2048 (1.7B)
(qwen-asr:decoder-n-layers *dec*)   ; always 28

;; Create a fresh KV-cache state (optional initial-cap, default 2048)
(defvar *state* (qwen-asr:make-decoder-state *dec*))
(qwen-asr:decoder-state-cur-len *state*)  ; => 0

;; Look up a token embedding => flat [hidden] array
(defvar *embed* (qwen-asr:decoder-embed *dec* 151644))  ; <|im_start|>

;; Prefill: process a prompt sequence, update KV cache
;; input-embeds: flat [(seq-len * hidden)]
;; Returns x (flat [seq-len * hidden])
(qwen-asr:decoder-prefill *dec* *state* input-embeds seq-len)

;; Single autoregressive step => greedy token ID
;; Processes one embed, updates KV cache, returns next token
(qwen-asr:decoder-step *dec* *state* *embed*)
```

### Full pipeline

```lisp
;; One-liner: WAV path => transcribed string
(qwen-asr:transcribe *enc* *dec* *tok* "/path/to/audio.wav")

;; Also accepts pre-loaded samples
(qwen-asr:transcribe *enc* *dec* *tok* *samples*)
```

## Model variant detection

Both encoder and decoder auto-detect the model variant (0.6B vs 1.7B) from
weight shapes at load time. No explicit variant flag is required.

|              | 0.6B | 1.7B |
|--------------|------|------|
| dec hidden   | 1024 | 2048 |
| enc d-model  | 896  | 1024 |
| enc layers   | 18   | 24   |
| dec layers   | 28   | 28   |

## Memory note

All weights are loaded as `single-float` (f32), including those stored as BF16
in the model files. This doubles BF16 weight memory: 0.6B is ~1.2 GB total,
1.7B is ~3.4 GB total.

## Performance note

- **Encoder**: linear projections and conv2D use LLA/BLAS.
- **Decoder attention**: pure Common Lisp loop — correct but slower than C.
- **LM head**: tied embedding matrix via LLA GEMM (BLAS-accelerated).

For production throughput, the C binary (`../qwen_asr`) is recommended.
This port is intended for research, experimentation, and further optimization.
