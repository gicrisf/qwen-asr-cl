(in-package #:qwen-asr)

;;; =========================================================================
;;; Structs
;;; =========================================================================

(defstruct enc-layer
  ;; Attention projection weights [d-model, d-model] and biases [d-model]
  wq wq-bias
  wk wk-bias
  wv wv-bias
  wo wo-bias
  ;; Pre-attention LayerNorm weight + bias [d-model]
  attn-norm attn-norm-bias
  ;; FFN: fc1 [ffn-dim, d-model], fc2 [d-model, ffn-dim], + biases
  fc1 fc1-bias
  fc2 fc2-bias
  ;; Pre-FFN LayerNorm weight + bias [d-model]
  ffn-norm ffn-norm-bias)

(defstruct encoder
  ;; Conv2D stem weights [c-out, c-in*kh*kw] and biases [c-out]
  conv1-w conv1-b
  conv2-w conv2-b
  conv3-w conv3-b
  ;; Conv output projection [d-model, 7680], no bias
  conv-out-w
  ;; Transformer layers: simple-vector of ENC-LAYER
  layers
  n-layers
  ;; Post-encoder LayerNorm [d-model] + bias
  ln-post-w ln-post-b
  ;; Final projections
  proj1-w proj1-b   ; [d-model, d-model]
  proj2-w proj2-b   ; [output-dim, d-model]
  ;; Architecture config
  d-model ffn-dim n-heads head-dim output-dim chunk-size n-window-infer)

;;; =========================================================================
;;; Weight loading
;;; =========================================================================

(defun %enc-tensor (ms prefix key)
  "Load tensor (PREFIX ++ KEY) from MS as a fresh flat f32 array."
  (let ((name (concatenate 'string prefix key)))
    (multiple-value-bind (sf tensor) (find-tensor ms name)
      (unless tensor
        (error "encoder: weight not found: ~a" name))
      (tensor-f32 sf tensor))))

(defun %enc-tensor-2d (ms prefix key)
  "Load rank-2 tensor (PREFIX ++ KEY) as a simple 2D f32 array."
  (let ((name (concatenate 'string prefix key)))
    (multiple-value-bind (sf tensor) (find-tensor ms name)
      (unless tensor
        (error "encoder: weight not found: ~a" name))
      (tensor-f32-2d sf tensor))))

(defun load-encoder (model-dir)
  "Load encoder weights from MODEL-DIR.  Returns an ENCODER struct.
Variant is auto-detected: 1.7B has 24 layers, 0.6B has 18."
  (let* ((ms  (open-model model-dir))
         (pfx "thinker.audio_tower.")
         ;; Variant detection: 1.7B has layer index 18, 0.6B only 0..17
         (is-17b    (nth-value 1
                      (find-tensor ms "thinker.audio_tower.layers.18.self_attn.q_proj.weight")))
         (n-layers    (if is-17b 24 18))
         (d-model     (if is-17b 1024 896))
         (n-heads     (if is-17b 16   14))
         (head-dim    64)
         (ffn-dim     (if is-17b 4096 3584))
         (output-dim  (if is-17b 2048 1024))
         (chunk-size  100)
         (n-window-infer 800))
    ;; Conv stem
    (let ((conv1-w   (%enc-tensor ms pfx "conv2d1.weight"))
          (conv1-b   (%enc-tensor ms pfx "conv2d1.bias"))
          (conv2-w   (%enc-tensor ms pfx "conv2d2.weight"))
          (conv2-b   (%enc-tensor ms pfx "conv2d2.bias"))
          (conv3-w   (%enc-tensor ms pfx "conv2d3.weight"))
          (conv3-b   (%enc-tensor ms pfx "conv2d3.bias"))
          (conv-out-w (%enc-tensor-2d ms pfx "conv_out.weight")))
      ;; Transformer layers
      (let ((layers (make-array n-layers)))
        (dotimes (i n-layers)
          (let ((lp (format nil "~alayers.~d." pfx i)))
            (setf (aref layers i)
                  (make-enc-layer
                   :wq             (%enc-tensor-2d ms lp "self_attn.q_proj.weight")
                   :wq-bias        (%enc-tensor ms lp "self_attn.q_proj.bias")
                   :wk             (%enc-tensor-2d ms lp "self_attn.k_proj.weight")
                   :wk-bias        (%enc-tensor ms lp "self_attn.k_proj.bias")
                   :wv             (%enc-tensor-2d ms lp "self_attn.v_proj.weight")
                   :wv-bias        (%enc-tensor ms lp "self_attn.v_proj.bias")
                   :wo             (%enc-tensor-2d ms lp "self_attn.out_proj.weight")
                   :wo-bias        (%enc-tensor ms lp "self_attn.out_proj.bias")
                   :attn-norm      (%enc-tensor ms lp "self_attn_layer_norm.weight")
                   :attn-norm-bias (%enc-tensor ms lp "self_attn_layer_norm.bias")
                   :fc1            (%enc-tensor-2d ms lp "fc1.weight")
                   :fc1-bias       (%enc-tensor ms lp "fc1.bias")
                   :fc2            (%enc-tensor-2d ms lp "fc2.weight")
                   :fc2-bias       (%enc-tensor ms lp "fc2.bias")
                   :ffn-norm       (%enc-tensor ms lp "final_layer_norm.weight")
                   :ffn-norm-bias  (%enc-tensor ms lp "final_layer_norm.bias")))))
        (make-encoder
         :conv1-w conv1-w :conv1-b conv1-b
         :conv2-w conv2-w :conv2-b conv2-b
         :conv3-w conv3-w :conv3-b conv3-b
         :conv-out-w  conv-out-w
         :layers      layers
         :n-layers    n-layers
         :ln-post-w   (%enc-tensor ms pfx "ln_post.weight")
         :ln-post-b   (%enc-tensor ms pfx "ln_post.bias")
         :proj1-w     (%enc-tensor-2d ms pfx "proj1.weight")
         :proj1-b     (%enc-tensor ms pfx "proj1.bias")
         :proj2-w     (%enc-tensor-2d ms pfx "proj2.weight")
         :proj2-b     (%enc-tensor ms pfx "proj2.bias")
         :d-model     d-model  :ffn-dim    ffn-dim
         :n-heads     n-heads  :head-dim   head-dim
         :output-dim  output-dim
         :chunk-size  chunk-size
         :n-window-infer n-window-infer)))))

;;; =========================================================================
;;; Bidirectional windowed attention (pure CL, window ≤ 104 tokens)
;;; =========================================================================

(defun %bidirectional-attention (out q k v total-tokens n-heads head-dim scale
                                  window-starts n-windows)
  "Online-softmax bidirectional windowed attention.
Q, K, V, OUT: flat [total-tokens × (n-heads × head-dim)].
Matches C qwen_bidirectional_attention exactly. Returns OUT."
  (let ((hidden (* n-heads head-dim)))
    (dotimes (h n-heads out)
      (dotimes (win n-windows)
        (let ((ws (aref window-starts win))
              (we (aref window-starts (1+ win))))
          (loop for i from ws below we do
            (let ((q-base (+ (* i hidden) (* h head-dim)))
                  (o-base (+ (* i hidden) (* h head-dim))))
              ;; Zero the output slice for this query × head
              (dotimes (d head-dim)
                (setf (aref out (+ o-base d)) 0.0f0))
              (let ((max-score -1.0f30)
                    (sum-exp    0.0f0))
                (loop for j from ws below we do
                  (let ((k-base (+ (* j hidden) (* h head-dim)))
                        (v-base (+ (* j hidden) (* h head-dim))))
                    (let ((score 0.0f0))
                      (dotimes (d head-dim)
                        (incf score (* (aref q (+ q-base d))
                                       (aref k (+ k-base d)))))
                      (setf score (* score scale))
                      (if (> score max-score)
                          ;; New maximum: rescale o-row and update
                          (let ((corr (exp (- max-score score))))
                            (dotimes (d head-dim)
                              (setf (aref out (+ o-base d))
                                    (+ (* (aref out (+ o-base d)) corr)
                                       (aref v (+ v-base d)))))
                            (setf sum-exp (+ (* sum-exp corr) 1.0f0))
                            (setf max-score score))
                          ;; Normal: weighted accumulate
                          (let ((wt (exp (- score max-score))))
                            (dotimes (d head-dim)
                              (incf (aref out (+ o-base d))
                                    (* wt (aref v (+ v-base d)))))
                            (incf sum-exp wt))))))
                (when (> sum-exp 0.0f0)
                  (let ((inv-sum (/ 1.0f0 sum-exp)))
                    (dotimes (d head-dim)
                      (setf (aref out (+ o-base d))
                            (* (aref out (+ o-base d)) inv-sum)))))))))))))

;;; =========================================================================
;;; Encoder forward pass
;;; =========================================================================

(defun encoder-forward (enc mel mel-frames)
  "Run encoder forward pass.
MEL:       2D (simple-array single-float (128 n-frames)) from MEL-SPECTROGRAM.
MEL-FRAMES: number of frames (second dimension of MEL).
Returns (values enc-output total-tokens) where enc-output is a flat
(simple-array single-float (*)) of shape [total-tokens × output-dim]."
  (let* ((d-model        (encoder-d-model enc))
         (n-heads        (encoder-n-heads enc))
         (head-dim       (encoder-head-dim enc))
         (ffn-dim        (encoder-ffn-dim enc))
         (output-dim     (encoder-output-dim enc))
         (chunk-size     (encoder-chunk-size enc))
         (n-window-infer (encoder-n-window-infer enc))
         (n-layers       (encoder-n-layers enc))
         (layers         (encoder-layers enc))
         (scale          (/ 1.0f0 (sqrt (float head-dim 1.0f0))))
         (n-chunks       (ceiling mel-frames chunk-size)))

    ;; ── Phase 1a: Count total output tokens ──────────────────────────────
    (let ((total-tokens
           (loop for c below n-chunks
                 sum (let* ((cw (min chunk-size (- mel-frames (* c chunk-size))))
                            (w1 (1+ (floor (- cw  1) 2)))
                            (w2 (1+ (floor (- w1  1) 2)))
                            (w3 (1+ (floor (- w2  1) 2))))
                       w3))))

      ;; ── Phase 1b: Conv2D stem + sinusoidal PE → x [total-tokens, d-model]
      (let ((x            (make-array (* total-tokens d-model)
                                      :element-type 'single-float
                                      :initial-element 0.0f0))
            (token-offset 0))

        (dotimes (c n-chunks)
          (let* ((start   (* c chunk-size))
                 (chunk-w (min chunk-size (- mel-frames start)))
                 ;; chunk-mel flat [1 * 128 * chunk-w], treated as [1, 128, chunk-w]
                 (chunk-mel (make-array (* 128 chunk-w)
                                        :element-type 'single-float)))
            ;; Extract columns [start, start+chunk-w) from mel [128, mel-frames]
            (dotimes (m 128)
              (dotimes (col chunk-w)
                (setf (aref chunk-mel (+ (* m chunk-w) col))
                      (aref mel m (+ start col)))))

            ;; Conv1: c-in=1, c-out=480, h-in=128, kh=kw=3, stride=2, pad=1
            (let* ((h1  (1+ (floor (- 128    1) 2)))   ; 64
                   (w1  (1+ (floor (- chunk-w 1) 2)))
                   (c1  (conv2d chunk-mel
                                (encoder-conv1-w enc) (encoder-conv1-b enc)
                                1 480 128 chunk-w 3 3 2 1)))
              (gelu! c1 (* 480 h1 w1))

              ;; Conv2: c-in=480, c-out=480, h-in=h1
              (let* ((h2  (1+ (floor (- h1 1) 2)))     ; 32
                     (w2  (1+ (floor (- w1 1) 2)))
                     (c2  (conv2d c1
                                  (encoder-conv2-w enc) (encoder-conv2-b enc)
                                  480 480 h1 w1 3 3 2 1)))
                (gelu! c2 (* 480 h2 w2))

                ;; Conv3: c-in=480, c-out=480, h-in=h2
                (let* ((h3  (1+ (floor (- h2 1) 2)))   ; 16
                       (w3  (1+ (floor (- w2 1) 2)))
                       (c3  (conv2d c2
                                    (encoder-conv3-w enc) (encoder-conv3-b enc)
                                    480 480 h2 w2 3 3 2 1)))
                  (gelu! c3 (* 480 h3 w3))

                  ;; Reshape c3 [480, h3, w3] → reshaped [w3, 480*h3]
                  (let* ((cpd      (* 480 h3))       ; conv-proj-dim = 7680
                         (reshaped (make-array (* w3 cpd)
                                               :element-type 'single-float)))
                    (dotimes (tok w3)
                      (dotimes (ch 480)
                        (dotimes (f h3)
                          (setf (aref reshaped (+ (* tok cpd) (* ch h3) f))
                                (aref c3 (+ (* ch h3 w3) (* f w3) tok))))))

                    ;; Project [w3, 7680] → [w3, d-model], write into x at offset
                    (let ((proj (linear-nobias reshaped w3 cpd
                                              (encoder-conv-out-w enc) d-model)))
                      (dotimes (tok w3)
                        (let ((x-base   (* (+ token-offset tok) d-model))
                              (pr-base  (* tok d-model)))
                          (dotimes (d d-model)
                            (setf (aref x (+ x-base d))
                                  (aref proj (+ pr-base d))))))

                      ;; Add per-chunk sinusoidal PE (positions 0..w3-1)
                      (let ((pe (make-array (* w3 d-model)
                                            :element-type 'single-float)))
                        (sinusoidal-pe! pe w3 d-model)
                        (dotimes (tok w3)
                          (let ((x-base  (* (+ token-offset tok) d-model))
                                (pe-base (* tok d-model)))
                            (dotimes (d d-model)
                              (incf (aref x (+ x-base d))
                                    (aref pe (+ pe-base d))))))))

                    (incf token-offset w3)))))))

        ;; ── Phase 2: Build attention window boundaries ────────────────────
        ;; window-token-size = tokens from a full chunk × (n-window-infer / chunk-size)
        (let* ((fw1               (1+ (floor (- chunk-size 1) 2)))   ; 50
               (fw2               (1+ (floor (- fw1 1) 2)))           ; 25
               (fw3               (1+ (floor (- fw2 1) 2)))           ; 13
               (window-token-size (* fw3 (floor n-window-infer chunk-size))) ; 104
               (n-windows         (ceiling total-tokens window-token-size))
               (window-starts     (make-array (1+ n-windows)
                                              :element-type 'fixnum)))
          (dotimes (w n-windows)
            (setf (aref window-starts w) (* w window-token-size)))
          (setf (aref window-starts n-windows) total-tokens)

          ;; ── Phase 3: Transformer layers ──────────────────────────────────
          (let ((x-norm   (make-array (* total-tokens d-model)
                                      :element-type 'single-float))
                (attn-out (make-array (* total-tokens d-model)
                                      :element-type 'single-float)))
            (dotimes (l n-layers)
              (let ((layer (aref layers l)))

                ;; Self-attention
                (layer-norm! x-norm x
                             (enc-layer-attn-norm layer)
                             (enc-layer-attn-norm-bias layer)
                             total-tokens d-model)
                (let* ((q (linear x-norm total-tokens d-model
                                  (enc-layer-wq layer) d-model
                                  (enc-layer-wq-bias layer)))
                       (k (linear x-norm total-tokens d-model
                                  (enc-layer-wk layer) d-model
                                  (enc-layer-wk-bias layer)))
                       (v (linear x-norm total-tokens d-model
                                  (enc-layer-wv layer) d-model
                                  (enc-layer-wv-bias layer))))
                  (%bidirectional-attention attn-out q k v
                                            total-tokens n-heads head-dim scale
                                            window-starts n-windows))
                (let ((proj-out (linear attn-out total-tokens d-model
                                        (enc-layer-wo layer) d-model
                                        (enc-layer-wo-bias layer))))
                  (add! x proj-out (* total-tokens d-model)))

                ;; FFN
                (layer-norm! x-norm x
                             (enc-layer-ffn-norm layer)
                             (enc-layer-ffn-norm-bias layer)
                             total-tokens d-model)
                (let ((ffn-mid (linear x-norm total-tokens d-model
                                       (enc-layer-fc1 layer) ffn-dim
                                       (enc-layer-fc1-bias layer))))
                  (gelu! ffn-mid (* total-tokens ffn-dim))
                  (let ((ffn-out (linear ffn-mid total-tokens ffn-dim
                                         (enc-layer-fc2 layer) d-model
                                         (enc-layer-fc2-bias layer))))
                    (add! x ffn-out (* total-tokens d-model))))))

            ;; ── Phase 4: Final projection ───────────────────────────────────
            ;; ln-post in-place (out aliases x, safe: reads full row before writing)
            (layer-norm! x x
                         (encoder-ln-post-w enc) (encoder-ln-post-b enc)
                         total-tokens d-model)
            (let ((proj-mid (linear x total-tokens d-model
                                    (encoder-proj1-w enc) d-model
                                    (encoder-proj1-b enc))))
              (gelu! proj-mid (* total-tokens d-model))
              (let ((enc-output (linear proj-mid total-tokens d-model
                                        (encoder-proj2-w enc) output-dim
                                        (encoder-proj2-b enc))))
                (values enc-output total-tokens)))))))))
