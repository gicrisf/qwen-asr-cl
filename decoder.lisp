(in-package #:qwen-asr)

;;; =========================================================================
;;; Structs
;;; =========================================================================

(defstruct dec-layer
  ;; Attention projection weights [out, in], f32 (converted from bf16 at load)
  wq   ; [q-dim, hidden]   q-dim = n-heads * head-dim
  wk   ; [kv-dim, hidden]  kv-dim = n-kv-heads * head-dim
  wv   ; [kv-dim, hidden]
  wo   ; [hidden, q-dim]
  ;; Per-head Q/K RMSNorm [head-dim]
  q-norm
  k-norm
  ;; Pre-layer RMSNorm [hidden]
  input-norm
  post-attn-norm
  ;; SwiGLU MLP weights [out, in], f32 (no bias)
  gate-w   ; [intermediate, hidden]
  up-w     ; [intermediate, hidden]
  down-w)  ; [hidden, intermediate]

(defstruct decoder
  ;; Tied token embeddings / lm-head, flat [(vocab * hidden)], f32
  embed-tokens
  ;; Transformer layers: simple-vector of DEC-LAYER, length n-layers
  layers
  ;; Final RMSNorm weight [hidden]
  norm-w
  ;; Architecture config
  hidden
  n-layers
  n-heads
  n-kv-heads
  head-dim
  intermediate
  vocab-size
  rms-eps
  rope-theta)

(defstruct (decoder-state (:constructor %make-decoder-state-raw))
  ;; KV cache: flat [(n-layers * max-seq * kv-dim)]
  kv-k
  kv-v
  ;; RoPE tables: flat [(max-pos * head-dim)]
  rope-cos
  rope-sin
  ;; Shape / capacity
  n-layers
  kv-dim
  head-dim
  max-seq
  max-pos
  rope-theta
  ;; Current fill
  cur-len)

;;; =========================================================================
;;; Weight loading
;;; =========================================================================

(defun %dec-tensor (ms prefix key)
  "Load tensor (PREFIX ++ KEY) from MS as a fresh flat f32 array."
  (let ((name (concatenate 'string prefix key)))
    (multiple-value-bind (sf tensor) (find-tensor ms name)
      (unless tensor
        (error "decoder: weight not found: ~a" name))
      (tensor-f32 sf tensor))))

(defun load-decoder (model-dir)
  "Load decoder weights from MODEL-DIR. Returns a DECODER struct.
Variant auto-detected: hidden=1024 → 0.6B, hidden=2048 → 1.7B."
  (let* ((ms  (open-model model-dir))
         (pfx "thinker.model.")
         ;; Variant detection via norm weight shape
         (norm-key "thinker.model.norm.weight")
         (norm-t (nth-value 1 (find-tensor ms norm-key)))
         (hidden (if norm-t
                     (aref (safetensor-shape norm-t) 0)
                     (error "decoder: cannot find ~a" norm-key)))
         (n-layers    28)
         (n-heads     16)
         (n-kv-heads  8)
         (head-dim    128)
         (intermediate (if (= hidden 1024) 3072 6144))
         (vocab-size  151936)
         (rms-eps     1.0f-6)
         (rope-theta  1.0f6))
    (let ((embed-tokens (%dec-tensor ms pfx "embed_tokens.weight"))
          (layers (make-array n-layers)))
      (dotimes (i n-layers)
        (let ((lp (format nil "~alayers.~d." pfx i)))
          (setf (aref layers i)
                (make-dec-layer
                 :wq             (%dec-tensor ms lp "self_attn.q_proj.weight")
                 :wk             (%dec-tensor ms lp "self_attn.k_proj.weight")
                 :wv             (%dec-tensor ms lp "self_attn.v_proj.weight")
                 :wo             (%dec-tensor ms lp "self_attn.o_proj.weight")
                 :q-norm         (%dec-tensor ms lp "self_attn.q_norm.weight")
                 :k-norm         (%dec-tensor ms lp "self_attn.k_norm.weight")
                 :input-norm     (%dec-tensor ms lp "input_layernorm.weight")
                 :post-attn-norm (%dec-tensor ms lp "post_attention_layernorm.weight")
                 :gate-w         (%dec-tensor ms lp "mlp.gate_proj.weight")
                 :up-w           (%dec-tensor ms lp "mlp.up_proj.weight")
                 :down-w         (%dec-tensor ms lp "mlp.down_proj.weight")))))
      (make-decoder
       :embed-tokens embed-tokens
       :layers       layers
       :norm-w       (%dec-tensor ms pfx "norm.weight")
       :hidden       hidden
       :n-layers     n-layers
       :n-heads      n-heads
       :n-kv-heads   n-kv-heads
       :head-dim     head-dim
       :intermediate intermediate
       :vocab-size   vocab-size
       :rms-eps      rms-eps
       :rope-theta   rope-theta))))

;;; =========================================================================
;;; KV cache and RoPE state management
;;; =========================================================================

(defun %ensure-rope! (state min-pos)
  "Ensure rope-cos/sin tables cover positions [0, min-pos). Returns STATE."
  (when (>= (decoder-state-max-pos state) min-pos)
    (return-from %ensure-rope! state))
  (let* ((old-max  (decoder-state-max-pos state))
         (new-max  (max (* 2 (max 1 old-max)) min-pos))
         (head-dim (decoder-state-head-dim state))
         (theta    (decoder-state-rope-theta state))
         (half     (floor head-dim 2))
         (new-cos  (make-array (* new-max head-dim)
                               :element-type 'single-float
                               :initial-element 0.0f0))
         (new-sin  (make-array (* new-max head-dim)
                               :element-type 'single-float
                               :initial-element 0.0f0))
         (old-cos  (decoder-state-rope-cos state))
         (old-sin  (decoder-state-rope-sin state))
         (log-theta (log (float theta 1.0f0))))
    ;; Copy existing entries
    (dotimes (i (* old-max head-dim))
      (setf (aref new-cos i) (aref old-cos i))
      (setf (aref new-sin i) (aref old-sin i)))
    ;; Fill new positions
    (loop for p from old-max below new-max do
      (let ((base (* p head-dim))
            (fp   (float p 1.0f0)))
        (dotimes (d half)
          (let* ((power (/ (* 2.0f0 (float d 1.0f0)) (float head-dim 1.0f0)))
                 (freq  (exp (- (* power log-theta))))  ; 1/theta^(2d/head-dim)
                 (angle (* fp freq))
                 (c     (float (cos angle) 1.0f0))
                 (s     (float (sin angle) 1.0f0)))
            (setf (aref new-cos (+ base d))      c)
            (setf (aref new-cos (+ base half d)) c)  ; duplicate for NeoX
            (setf (aref new-sin (+ base d))      s)
            (setf (aref new-sin (+ base half d)) s)))))
    (setf (decoder-state-rope-cos state) new-cos)
    (setf (decoder-state-rope-sin state) new-sin)
    (setf (decoder-state-max-pos state) new-max)
    state))

(defun %grow-kv! (state new-max)
  "Grow KV cache arrays to NEW-MAX positions per layer. Returns STATE."
  (let* ((n-layers (decoder-state-n-layers state))
         (kv-dim   (decoder-state-kv-dim state))
         (old-max  (decoder-state-max-seq state))
         (cur-len  (decoder-state-cur-len state))
         (new-k    (make-array (* n-layers new-max kv-dim)
                               :element-type 'single-float
                               :initial-element 0.0f0))
         (new-v    (make-array (* n-layers new-max kv-dim)
                               :element-type 'single-float
                               :initial-element 0.0f0))
         (old-k    (decoder-state-kv-k state))
         (old-v    (decoder-state-kv-v state)))
    (dotimes (l n-layers)
      (let ((new-off (* l new-max kv-dim))
            (old-off (* l old-max kv-dim)))
        (dotimes (p cur-len)
          (dotimes (d kv-dim)
            (let ((ni (+ new-off (* p kv-dim) d))
                  (oi (+ old-off (* p kv-dim) d)))
              (setf (aref new-k ni) (aref old-k oi))
              (setf (aref new-v ni) (aref old-v oi)))))))
    (setf (decoder-state-kv-k state) new-k)
    (setf (decoder-state-kv-v state) new-v)
    (setf (decoder-state-max-seq state) new-max)
    state))

(defun make-decoder-state (dec &optional (initial-cap 2048))
  "Create a fresh DECODER-STATE for DEC, pre-allocated for INITIAL-CAP positions."
  (let* ((n-layers   (decoder-n-layers dec))
         (n-kv-heads (decoder-n-kv-heads dec))
         (head-dim   (decoder-head-dim dec))
         (kv-dim     (* n-kv-heads head-dim))
         (state (%make-decoder-state-raw
                 :kv-k       (make-array (* n-layers initial-cap kv-dim)
                                         :element-type 'single-float
                                         :initial-element 0.0f0)
                 :kv-v       (make-array (* n-layers initial-cap kv-dim)
                                         :element-type 'single-float
                                         :initial-element 0.0f0)
                 :rope-cos   (make-array 0 :element-type 'single-float)
                 :rope-sin   (make-array 0 :element-type 'single-float)
                 :n-layers   n-layers
                 :kv-dim     kv-dim
                 :head-dim   head-dim
                 :max-seq    initial-cap
                 :max-pos    0
                 :rope-theta (decoder-rope-theta dec)
                 :cur-len    0)))
    (%ensure-rope! state initial-cap)
    state))

;;; =========================================================================
;;; Internal helpers: RoPE and causal GQA attention
;;; =========================================================================

(defun %apply-rope-neox! (x rope-cos rope-sin seq-len n-heads head-dim pos-start)
  "Apply NeoX-style RoPE to X (flat [seq * n-heads * head-dim]) in-place.
For each position p = pos-start + s and head h:
  x1 = x[s,h,d], x2 = x[s,h,half+d]
  x[s,h,d]      = x1*cos[p,d] - x2*sin[p,d]
  x[s,h,half+d] = x2*cos[p,d] + x1*sin[p,d]
Returns X."
  (let* ((half   (floor head-dim 2))
         (hidden (* n-heads head-dim)))
    (dotimes (s seq-len x)
      (let ((cos-base (* (+ pos-start s) head-dim))
            (sin-base (* (+ pos-start s) head-dim)))
        (dotimes (h n-heads)
          (let ((vec-base (+ (* s hidden) (* h head-dim))))
            (dotimes (d half)
              (let* ((x1 (aref x (+ vec-base d)))
                     (x2 (aref x (+ vec-base half d)))
                     (c  (aref rope-cos (+ cos-base d)))
                     (sn (aref rope-sin (+ sin-base d))))
                (setf (aref x (+ vec-base d))
                      (- (* x1 c) (* x2 sn)))
                (setf (aref x (+ vec-base half d))
                      (+ (* x2 c) (* x1 sn)))))))))))

(defun %causal-attn-gqa! (out q kv-k kv-k-layer-off kv-v kv-v-layer-off
                           seq-q seq-k n-heads n-kv-heads head-dim
                           scale q-offset kv-dim)
  "Online-softmax causal GQA attention.
OUT, Q: flat [seq-q * n-heads * head-dim].
KV-K, KV-V: full flat KV cache; KV-K-LAYER-OFF/KV-V-LAYER-OFF are element offsets
for the current layer. Stride within a layer: pos * kv-dim.
KV-DIM = n-kv-heads * head-dim.
Q-OFFSET: global position of first query token (for causal mask).
GQA: kv-head = h / (n-heads / n-kv-heads).
Returns OUT."
  (let* ((heads-per-kv (floor n-heads n-kv-heads))
         (q-hidden     (* n-heads head-dim)))
    (dotimes (h n-heads out)
      (let ((kv-h (floor h heads-per-kv)))
        (dotimes (i seq-q)
          (let* ((q-base   (+ (* i q-hidden) (* h head-dim)))
                 (o-base   (+ (* i q-hidden) (* h head-dim)))
                 (k-end    (min (+ q-offset i 1) seq-k))
                 (max-scr  -1.0f30)
                 (sum-exp  0.0f0))
            ;; Zero output for this query position × head
            (dotimes (d head-dim)
              (setf (aref out (+ o-base d)) 0.0f0))
            ;; Online softmax over causal key positions
            (dotimes (j k-end)
              (let* ((k-base (+ kv-k-layer-off (* j kv-dim) (* kv-h head-dim)))
                     (v-base (+ kv-v-layer-off (* j kv-dim) (* kv-h head-dim)))
                     (score  0.0f0))
                (dotimes (d head-dim)
                  (incf score (* (aref q   (+ q-base d))
                                 (aref kv-k (+ k-base d)))))
                (setf score (* score scale))
                (cond
                  ((> score max-scr)
                   (let ((corr (exp (- max-scr score))))
                     (setf sum-exp (+ (* sum-exp corr) 1.0f0))
                     (dotimes (d head-dim)
                       (setf (aref out (+ o-base d))
                             (+ (* (aref out (+ o-base d)) corr)
                                (aref kv-v (+ v-base d)))))
                     (setf max-scr score)))
                  (t
                   (let ((wt (exp (- score max-scr))))
                     (incf sum-exp wt)
                     (dotimes (d head-dim)
                       (incf (aref out (+ o-base d))
                             (* wt (aref kv-v (+ v-base d))))))))))
            ;; Normalize
            (when (> sum-exp 0.0f0)
              (let ((inv-sum (/ 1.0f0 sum-exp)))
                (dotimes (d head-dim)
                  (setf (aref out (+ o-base d))
                        (* (aref out (+ o-base d)) inv-sum)))))))))))

;;; =========================================================================
;;; Core forward pass (shared by prefill and step)
;;; =========================================================================

(defun %decoder-forward! (dec state input-embeds seq-len)
  "Run SEQ-LEN token embeddings through all decoder layers, starting at
state.cur-len. Updates KV cache and increments cur-len. Returns modified
hidden-state X (flat [seq-len * hidden])."
  (let* ((hidden       (decoder-hidden dec))
         (n-layers     (decoder-n-layers dec))
         (n-heads      (decoder-n-heads dec))
         (n-kv-heads   (decoder-n-kv-heads dec))
         (head-dim     (decoder-head-dim dec))
         (intermediate (decoder-intermediate dec))
         (eps          (decoder-rms-eps dec))
         (q-dim        (* n-heads head-dim))
         (kv-dim       (* n-kv-heads head-dim))
         (scale        (/ 1.0f0 (sqrt (float head-dim 1.0f0))))
         (start        (decoder-state-cur-len state))
         (total-kv     (+ start seq-len))
         (x            (make-array (* seq-len hidden) :element-type 'single-float)))
    ;; Copy input embeddings
    (dotimes (i (* seq-len hidden))
      (setf (aref x i) (aref input-embeds i)))
    ;; Grow KV cache if needed
    (when (> total-kv (decoder-state-max-seq state))
      (%grow-kv! state (max (* 2 (decoder-state-max-seq state))
                            (+ total-kv 1024))))
    ;; Extend RoPE tables if needed
    (%ensure-rope! state total-kv)
    ;; Scratch buffers (reused across layers)
    (let ((x-norm   (make-array (* seq-len hidden) :element-type 'single-float))
          (attn-out (make-array (* seq-len q-dim)  :element-type 'single-float)))
      (dotimes (l n-layers)
        (let* ((layer        (aref (decoder-layers dec) l))
               (kv-layer-off (* l (decoder-state-max-seq state) kv-dim)))
          ;; ── Attention block ─────────────────────────────────────────────
          ;; Input RMSNorm
          (rms-norm! x-norm x (dec-layer-input-norm layer) seq-len hidden eps)
          ;; QKV projections (no bias)
          (let* ((q (linear x-norm seq-len hidden (dec-layer-wq layer) q-dim  nil))
                 (k (linear x-norm seq-len hidden (dec-layer-wk layer) kv-dim nil))
                 (v (linear x-norm seq-len hidden (dec-layer-wv layer) kv-dim nil)))
            ;; Per-head Q/K RMSNorm
            (rms-norm-per-head! q (dec-layer-q-norm layer)
                                seq-len n-heads    head-dim eps)
            (rms-norm-per-head! k (dec-layer-k-norm layer)
                                seq-len n-kv-heads head-dim eps)
            ;; NeoX RoPE
            (%apply-rope-neox! q
                               (decoder-state-rope-cos state)
                               (decoder-state-rope-sin state)
                               seq-len n-heads head-dim start)
            (%apply-rope-neox! k
                               (decoder-state-rope-cos state)
                               (decoder-state-rope-sin state)
                               seq-len n-kv-heads head-dim start)
            ;; Store K/V in cache at positions [start, start+seq-len)
            (dotimes (s seq-len)
              (let ((pos (+ start s)))
                (dotimes (d kv-dim)
                  (setf (aref (decoder-state-kv-k state)
                              (+ kv-layer-off (* pos kv-dim) d))
                        (aref k (+ (* s kv-dim) d)))
                  (setf (aref (decoder-state-kv-v state)
                              (+ kv-layer-off (* pos kv-dim) d))
                        (aref v (+ (* s kv-dim) d))))))
            ;; Causal GQA attention
            (%causal-attn-gqa! attn-out q
                               (decoder-state-kv-k state) kv-layer-off
                               (decoder-state-kv-v state) kv-layer-off
                               seq-len total-kv n-heads n-kv-heads head-dim
                               scale start kv-dim))
          ;; Output projection + residual
          (add! x (linear attn-out seq-len q-dim (dec-layer-wo layer) hidden nil)
                (* seq-len hidden))
          ;; ── FFN block ────────────────────────────────────────────────────
          ;; Post-attention RMSNorm
          (rms-norm! x-norm x (dec-layer-post-attn-norm layer) seq-len hidden eps)
          ;; SwiGLU: gate = silu(gate_proj(x)) * up_proj(x)
          (let* ((gate (linear x-norm seq-len hidden (dec-layer-gate-w layer) intermediate nil))
                 (up   (linear x-norm seq-len hidden (dec-layer-up-w   layer) intermediate nil))
                 (n-elem (* seq-len intermediate)))
            (silu! gate n-elem)
            (dotimes (i n-elem)
              (setf (aref gate i) (* (aref gate i) (aref up i))))
            ;; FFN output + residual
            (add! x (linear gate seq-len intermediate (dec-layer-down-w layer) hidden nil)
                  (* seq-len hidden)))))
      ;; Advance KV cache position
      (setf (decoder-state-cur-len state) total-kv)
      x)))

;;; =========================================================================
;;; Public decoder API
;;; =========================================================================

(defun decoder-embed (dec token-id)
  "Return a fresh flat [hidden] array with the embedding for TOKEN-ID."
  (let* ((hidden (decoder-hidden dec))
         (base   (* token-id hidden))
         (embed  (make-array hidden :element-type 'single-float)))
    (dotimes (d hidden embed)
      (setf (aref embed d)
            (aref (decoder-embed-tokens dec) (+ base d))))))

(defun decoder-prefill (dec state input-embeds seq-len)
  "Run all layers for SEQ-LEN tokens starting at state.cur-len. Updates KV cache.
INPUT-EMBEDS is flat [seq-len * hidden]. Returns x (flat [seq-len * hidden])."
  (%decoder-forward! dec state input-embeds seq-len))

(defun decoder-step (dec state embed)
  "Run one autoregressive step for a single token. EMBED is flat [hidden].
Updates KV cache and increments state.cur-len. Returns greedy token ID."
  (let* ((hidden (decoder-hidden dec))
         (vocab  (decoder-vocab-size dec))
         (x      (%decoder-forward! dec state embed 1)))
    ;; Final RMSNorm (in-place: out aliases x)
    (rms-norm! x x (decoder-norm-w dec) 1 hidden (decoder-rms-eps dec))
    ;; LM head = embed-tokens (tied weights); logits [1, vocab]
    (let ((logits (linear x 1 hidden (decoder-embed-tokens dec) vocab nil)))
      ;; Greedy argmax
      (let ((best-id  0)
            (best-val (aref logits 0)))
        (dotimes (i vocab best-id)
          (when (> (aref logits i) best-val)
            (setf best-val (aref logits i))
            (setf best-id  i)))))))
