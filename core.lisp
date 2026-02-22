;;;; core.lisp — Full transcription pipeline

(in-package #:qwen-asr)

;;; =========================================================================
;;; Special token IDs (from qwen_asr.c)
;;; =========================================================================

(defconstant +tok-im-start+    151644)
(defconstant +tok-im-end+      151645)
(defconstant +tok-endoftext+   151643)
(defconstant +tok-audio-start+ 151669)
(defconstant +tok-audio-end+   151670)
(defconstant +tok-asr-text+    151704)

;;; Fixed prompt token sequences (match C PROMPT_PREFIX_HEAD + PROMPT_PREFIX_TAIL
;;; merged into one, and PROMPT_SUFFIX_BASE).
;;;
;;;   PREFIX: <|im_start|> "system" \n  <|im_end|> \n  <|im_start|> "user" \n  <|audio_start|>
;;;   SUFFIX: <|audio_end|> <|im_end|> \n  <|im_start|> "assistant" \n
(defparameter +prompt-prefix+
  #(151644 8948 198 151645 198 151644 872 198 151669))

(defparameter +prompt-suffix+
  #(151670 151645 198 151644 77091 198))

;;; =========================================================================
;;; Transcription pipeline
;;; =========================================================================

(defun transcribe (enc dec tok wav-path-or-samples)
  "Full transcription pipeline.
ENC:  ENCODER struct (from LOAD-ENCODER).
DEC:  DECODER struct (from LOAD-DECODER).
TOK:  tokenizer struct (from LOAD-TOKENIZER).
WAV-PATH-OR-SAMPLES: a pathname/string or a (simple-array single-float (*))
  of 16 kHz mono samples.
Returns the ASR text as a string."
  (let* (;; ── Step 1: Audio → mel spectrogram ──────────────────────────────
         (samples (if (stringp wav-path-or-samples)
                      (load-wav wav-path-or-samples)
                      wav-path-or-samples))
         (mel     (mel-spectrogram samples))
         (hidden  (decoder-hidden dec)))

    ;; ── Step 2: Encoder forward pass ─────────────────────────────────────
    (multiple-value-bind (enc-output n-enc-tokens)
        (encoder-forward enc mel (array-dimension mel 1))

      ;; ── Step 3: Build full input-embeds ──────────────────────────────────
      ;; Layout: [prefix(9)] ++ [encoder-output(n-enc)] ++ [suffix(6)]
      (let* ((n-prefix    (length +prompt-prefix+))
             (n-suffix    (length +prompt-suffix+))
             (total-seq   (+ n-prefix n-enc-tokens n-suffix))
             (input-embeds (make-array (* total-seq hidden)
                                       :element-type 'single-float))
             (embed-tokens (decoder-embed-tokens dec))
             (embed-2d?   (= (array-rank embed-tokens) 2))
             (off          0))
        ;; Embed prefix tokens
        (dotimes (i n-prefix)
          (let ((tok (aref +prompt-prefix+ i))
                (dst (* off hidden)))
            (if embed-2d?
                (dotimes (d hidden)
                  (setf (aref input-embeds (+ dst d))
                        (aref embed-tokens tok d)))
                (let ((src (* tok hidden)))
                  (dotimes (d hidden)
                    (setf (aref input-embeds (+ dst d))
                          (aref embed-tokens (+ src d)))))))
          (incf off))
        ;; Copy encoder output directly (enc_dim == dec_hidden)
        (let ((enc-base (* off hidden)))
          (dotimes (i (* n-enc-tokens hidden))
            (setf (aref input-embeds (+ enc-base i))
                  (aref enc-output i))))
        (incf off n-enc-tokens)
        ;; Embed suffix tokens
        (dotimes (i n-suffix)
          (let ((tok (aref +prompt-suffix+ i))
                (dst (* off hidden)))
            (if embed-2d?
                (dotimes (d hidden)
                  (setf (aref input-embeds (+ dst d))
                        (aref embed-tokens tok d)))
                (let ((src (* tok hidden)))
                  (dotimes (d hidden)
                    (setf (aref input-embeds (+ dst d))
                          (aref embed-tokens (+ src d)))))))
          (incf off))

        ;; ── Step 4: Init decoder state ────────────────────────────────────
        (let ((state       (make-decoder-state dec))
              (prefill-len (1- total-seq)))

          ;; ── Step 5: Prefill all but last input token ──────────────────
          (decoder-prefill dec state input-embeds prefill-len)

          ;; ── Step 6: First decode step with last prompt token embed ─────
          (let ((last-embed (make-array hidden :element-type 'single-float)))
            (let ((last-base (* prefill-len hidden)))
              (dotimes (d hidden)
                (setf (aref last-embed d)
                      (aref input-embeds (+ last-base d)))))

            (let* ((last-token-id (aref +prompt-suffix+
                                        (1- (length +prompt-suffix+))))
                   (token          (decoder-step dec state last-embed last-token-id))
                  (result         (make-array 0
                                              :element-type 'character
                                              :adjustable t
                                              :fill-pointer 0))
                  (past-asr-text  nil))

              ;; ── Step 7: Autoregressive decode loop ─────────────────────
              (dotimes (_ 2048)
                ;; EOS check
                (when (or (= token +tok-im-end+)
                          (= token +tok-endoftext+))
                  (return))
                ;; Token handling
                (cond
                  ((= token +tok-asr-text+)
                   (setf past-asr-text t))
                  (past-asr-text
                   ;; Decode and accumulate text
                   (let ((piece (tokenizer-decode tok token)))
                     (loop for c across piece do
                       (vector-push-extend c result)))))
                ;; Embed generated token and run next step
                (let ((embed (decoder-embed dec token)))
                  (setf token (decoder-step dec state embed token))))

              ;; ── Step 8: Return accumulated text ────────────────────────
              (coerce result 'string))))))))
