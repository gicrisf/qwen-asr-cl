(in-package #:qwen-asr)

;;; =========================================================================
;;; Shared math primitives (encoder, and later decoder)
;;; =========================================================================

;;; ─── 2D view ─────────────────────────────────────────────────────────────

(defun as-matrix (flat-arr rows cols)
  "Return a 2D [rows × cols] displaced view over FLAT-ARR (no copy).
LLA handles displaced arrays by copying to a work area internally."
  (make-array (list rows cols)
              :element-type 'single-float
              :displaced-to flat-arr))

;;; ─── Element-wise ops ────────────────────────────────────────────────────

(defun add! (a b n)
  "a[i] += b[i] for i in [0,n). Modifies A in-place, returns A."
  (dotimes (i n a)
    (incf (aref a i) (aref b i))))

;;; ─── Normalization ───────────────────────────────────────────────────────

(defun layer-norm! (out x weight bias seq-len hidden &optional (eps 1.0f-5))
  "Per-row LayerNorm. OUT may alias X (reads full row before writing). Returns OUT."
  (dotimes (s seq-len out)
    (let ((base (* s hidden))
          (mean 0.0f0)
          (var  0.0f0))
      ;; mean
      (dotimes (i hidden)
        (incf mean (aref x (+ base i))))
      (setf mean (/ mean (float hidden 1.0f0)))
      ;; variance
      (dotimes (i hidden)
        (let ((d (- (aref x (+ base i)) mean)))
          (incf var (* d d))))
      (setf var (/ var (float hidden 1.0f0)))
      ;; normalize
      (let ((inv-std (/ 1.0f0 (sqrt (+ var eps)))))
        (dotimes (i hidden)
          (setf (aref out (+ base i))
                (+ (* (- (aref x (+ base i)) mean)
                      inv-std
                      (aref weight i))
                   (aref bias i))))))))

;;; ─── Activation functions ────────────────────────────────────────────────

(defun gelu! (x n)
  "GELU in-place (tanh approximation, matches C exactly). Returns X."
  (dotimes (i n x)
    (let* ((v     (aref x i))
           (inner (* 0.7978845608028654f0
                     (+ v (* 0.044715f0 (* v (* v v)))))))
      (setf (aref x i)
            (* 0.5f0 v (+ 1.0f0 (tanh inner)))))))

;;; ─── Sinusoidal position embeddings ──────────────────────────────────────

(defun sinusoidal-pe! (pe n-pos d-model)
  "Fill flat PE [n-pos * d-model] with sinusoidal embeddings. Returns PE.
First half of each row = sin, second half = cos (matches C qwen_sinusoidal_pe)."
  (let* ((half          (floor d-model 2))
         (log-timescale (/ (log 10000.0f0)
                           (float (max 1 (1- half)) 1.0f0))))
    (dotimes (p n-pos pe)
      (dotimes (d half)
        (let* ((inv-ts (exp (* (- (float d 1.0f0)) log-timescale)))
               (angle  (* (float p 1.0f0) inv-ts)))
          (setf (aref pe (+ (* p d-model) d))        (sin angle))
          (setf (aref pe (+ (* p d-model) half d))   (cos angle)))))))

;;; ─── Linear layer ────────────────────────────────────────────────────────
;;;
;;; y[seq, out] = x[seq, in] @ W[out, in]^T + b[out]
;;;
;;; Uses lla:gemm! with :transpose-b? t — avoids explicit W transposition.
;;; The result 2D array is kept alive by the returned 1D displaced view.

(defun linear (x seq-len in-dim w out-dim b)
  "y = x [seq,in] @ W[out,in]^T + b. Returns fresh flat [seq*out] array.
If W is a simple rank-2 array, pass it directly to BLAS to avoid a copy."
  (let* ((y-2d (make-array (list seq-len out-dim) :element-type 'single-float))
         (y    (make-array (* seq-len out-dim)
                           :element-type 'single-float
                           :displaced-to y-2d))
         (x2d  (if (= (array-rank x) 2)
                   x
                   (as-matrix x seq-len in-dim)))
         (w2d  (if (= (array-rank w) 2)
                   w
                   (as-matrix w out-dim in-dim))))
    (lla:gemm! 1.0f0
               x2d
               w2d
               0.0f0 y-2d
               :transpose-b? t)
    (when b
      (dotimes (s seq-len)
        (let ((base (* s out-dim)))
          (dotimes (o out-dim)
            (incf (aref y (+ base o)) (aref b o))))))
    y))

(defun linear-nobias (x seq-len in-dim w out-dim)
  "y = x [seq,in] @ W[out,in]^T.  Returns fresh flat array."
  (linear x seq-len in-dim w out-dim nil))

;;; ─── RMSNorm ─────────────────────────────────────────────────────────────

(defun rms-norm! (out x weight seq-len hidden &optional (eps 1.0f-6))
  "Per-row RMSNorm (no mean subtraction). OUT may alias X (reads full row before
writing). Returns OUT.
  rms = sqrt(mean(x²) + eps)
  out[s,i] = x[s,i] / rms * weight[i]"
  (dotimes (s seq-len out)
    (let* ((base   (* s hidden))
           (sum-sq 0.0f0))
      (dotimes (i hidden)
        (let ((v (aref x (+ base i))))
          (incf sum-sq (* v v))))
      (let ((rms-inv (/ 1.0f0 (sqrt (+ (/ sum-sq (float hidden 1.0f0)) eps)))))
        (dotimes (i hidden)
          (setf (aref out (+ base i))
                (* (aref x (+ base i)) rms-inv (aref weight i))))))))

(defun rms-norm-per-head! (x weight seq-len n-heads head-dim &optional (eps 1.0f-6))
  "RMSNorm each head segment of X in-place. X is flat [seq * n-heads * head-dim].
WEIGHT is flat [head-dim], shared across all heads. Returns X."
  (let ((hidden (* n-heads head-dim)))
    (dotimes (s seq-len x)
      (dotimes (h n-heads)
        (let* ((base   (+ (* s hidden) (* h head-dim)))
               (sum-sq 0.0f0))
          (dotimes (d head-dim)
            (let ((v (aref x (+ base d))))
              (incf sum-sq (* v v))))
          (let ((rms-inv (/ 1.0f0 (sqrt (+ (/ sum-sq (float head-dim 1.0f0)) eps)))))
            (dotimes (d head-dim)
              (setf (aref x (+ base d))
                    (* (aref x (+ base d)) rms-inv (aref weight d))))))))))

;;; ─── SiLU ────────────────────────────────────────────────────────────────

(defun silu! (x n)
  "SiLU in-place: x[i] = x[i] * sigmoid(x[i]). Numerically stable. Returns X.
Two-branch form ensures exp() is only ever called with non-positive arguments,
preventing single-float overflow for large negative inputs (e.g. v < -88)."
  (dotimes (i n x)
    (let ((v (aref x i)))
      (setf (aref x i)
            (if (>= v 0.0f0)
                (/ v (+ 1.0f0 (exp (- v))))       ; exp(-v) in (0,1], safe
                (* v (/ (exp v) (+ 1.0f0 (exp v)))))))))  ; exp(v) in (0,1), safe

;;; ─── 2D Convolution (im2col + BLAS GEMM) ────────────────────────────────

(defun %im2col (in cols c-in h-in w-in kh kw stride padding h-out w-out)
  "Unroll IN [c-in,h-in,w-in] patches into COLS flat [patch-size, h-out*w-out]."
  (let ((col-len (* h-out w-out)))
    (dotimes (ic c-in)
      (dotimes (ki kh)
        (dotimes (kj kw)
          (let* ((col-row  (+ (* (+ (* ic kh) ki) kw) kj))
                 (col-base (* col-row col-len)))
            (dotimes (oh h-out)
              (let ((ih (+ (* oh stride) (- ki padding))))
                (dotimes (ow w-out)
                  (let ((iw (+ (* ow stride) (- kj padding))))
                    (setf (aref cols (+ col-base (* oh w-out) ow))
                          (if (and (>= ih 0) (< ih h-in)
                                   (>= iw 0) (< iw w-in))
                              (aref in (+ (* ic (* h-in w-in))
                                          (* ih w-in)
                                          iw))
                              0.0f0))))))))))))

(defun conv2d (in weight bias c-in c-out h-in w-in kh kw stride padding)
  "Conv2D via im2col + BLAS GEMM (matches C qwen_conv2d exactly).
IN:     flat [c-in * h-in * w-in]
WEIGHT: flat [c-out * c-in * kh * kw]
BIAS:   flat [c-out] or NIL
Returns fresh flat [c-out * h-out * w-out]."
  (let* ((h-out      (1+ (floor (+ h-in (* 2 padding) (- kh)) stride)))
         (w-out      (1+ (floor (+ w-in (* 2 padding) (- kw)) stride)))
         (patch-size (* c-in kh kw))
         (spatial    (* h-out w-out))
         (cols       (make-array (* patch-size spatial) :element-type 'single-float))
         (out-2d     (make-array (list c-out spatial) :element-type 'single-float))
         (out        (make-array (* c-out spatial)
                                 :element-type 'single-float
                                 :displaced-to out-2d)))
    (%im2col in cols c-in h-in w-in kh kw stride padding h-out w-out)
    ;; weight [c-out, patch-size] @ cols [patch-size, spatial] → out-2d [c-out, spatial]
    (lla:gemm! 1.0f0
               (as-matrix weight c-out patch-size)
               (as-matrix cols patch-size spatial)
               0.0f0 out-2d)
    (when bias
      (dotimes (oc c-out)
        (let ((b   (aref bias oc))
              (row (* oc spatial)))
          (dotimes (s spatial)
            (incf (aref out (+ row s)) b)))))
    out))
