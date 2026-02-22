(in-package #:qwen-asr)

;;; =========================================================================
;;; Constants (must match C exactly)
;;; =========================================================================

(defconstant +sample-rate+ 16000)
(defconstant +n-mel+       128)
(defconstant +hop-length+  160)
(defconstant +win-length+  400)
(defconstant +n-fft+       400)
(defconstant +n-freq+      201)   ; N-FFT/2 + 1

;;; =========================================================================
;;; WAV loading helpers
;;; =========================================================================

(defun le-u16 (bytes offset)
  "Read 16-bit unsigned little-endian integer from BYTES at OFFSET."
  (logior (aref bytes offset)
          (ash (aref bytes (+ offset 1)) 8)))

(defun le-u32 (bytes offset)
  "Read 32-bit unsigned little-endian integer from BYTES at OFFSET."
  (logior (aref bytes offset)
          (ash (aref bytes (+ offset 1)) 8)
          (ash (aref bytes (+ offset 2)) 16)
          (ash (aref bytes (+ offset 3)) 24)))

(defun wav-tag= (bytes offset tag)
  "Return true if the 4 bytes at OFFSET in BYTES equal the 4-char TAG string."
  (and (= (aref bytes offset)       (char-code (char tag 0)))
       (= (aref bytes (+ offset 1)) (char-code (char tag 1)))
       (= (aref bytes (+ offset 2)) (char-code (char tag 2)))
       (= (aref bytes (+ offset 3)) (char-code (char tag 3)))))

;;; =========================================================================
;;; Resampling: windowed-sinc with Kaiser window (matches C double precision)
;;; =========================================================================

(defun bessel-i0 (x)
  "Modified Bessel function I₀(x) via 20-term power series."
  (let* ((xd   (float x 1.0d0))
         (xx   (* xd xd))
         (sum  1.0d0)
         (term 1.0d0))
    (loop for k from 1 to 20 do
      (setf term (* term (/ xx (* 4.0d0 (float k 1.0d0) (float k 1.0d0)))))
      (incf sum term))
    sum))

(defun resample-kaiser (samples src-rate tgt-rate)
  "Windowed-sinc resampler with Kaiser window (beta=6, 16 zero-crossings).
SAMPLES: (simple-array single-float (*)).  Returns a new single-float array."
  (let* ((n-src     (length samples))
         (ratio     (/ (float tgt-rate 1.0d0) (float src-rate 1.0d0)))
         (n-new     (floor (* n-src ratio)))
         (sinc-half 16)
         (beta      6.0d0)
         (cutoff    (if (< ratio 1.0d0) ratio 1.0d0))
         (inv-i0    (/ 1.0d0 (bessel-i0 beta)))
         (result    (make-array n-new :element-type 'single-float)))
    (loop for i below n-new do
      (let* ((src-pos (/ (float i 1.0d0) ratio))
             (center  (floor src-pos))
             (acc     0.0d0)
             (wsum    0.0d0))
        ;; j from center-sinc_half+1 to center+sinc_half (matches C j_lo/j_hi)
        (loop for j from (1+ (- center sinc-half)) to (+ center sinc-half) do
          (let* ((d    (- (float j 1.0d0) src-pos))
                 (x    (* d cutoff))
                 (s    (if (< (abs x) 1.0d-9)
                           1.0d0
                           (/ (sin (* pi x)) (* pi x))))
                 (npos (/ d (float sinc-half 1.0d0)))
                 (w    (if (or (<= npos -1.0d0) (>= npos 1.0d0))
                           0.0d0
                           (* (bessel-i0 (* beta (sqrt (- 1.0d0 (* npos npos)))))
                              inv-i0)))
                 (coeff (* s w cutoff)))
            (when (and (>= j 0) (< j n-src))
              (incf acc (* (float (aref samples j) 1.0d0) coeff)))
            (incf wsum coeff)))
        (setf (aref result i)
              (float (if (> wsum 1.0d-9) (/ acc wsum) 0.0d0) 1.0f0))))
    result))

;;; =========================================================================
;;; WAV loading
;;; =========================================================================

(defun load-wav (path)
  "Load a WAV file at PATH.
Returns a (simple-array single-float (*)) of mono 16 kHz samples in [-1, 1].
Signals an error for files that are not 16-bit PCM WAV."
  (let* ((data      (read-file-bytes path))
         (file-size (length data)))
    (unless (and (>= file-size 44)
                 (wav-tag= data 0 "RIFF")
                 (wav-tag= data 8 "WAVE"))
      (error "load-wav: ~a is not a valid WAV file" path))

    (let ((audio-format    0)
          (channels        0)
          (sample-rate     0)
          (bits-per-sample 0)
          (pcm-offset      nil)
          (pcm-size        0)
          (p               12))

      ;; Scan RIFF chunks for fmt and data
      (loop while (<= (+ p 8) file-size) do
        (let ((chunk-size (le-u32 data (+ p 4))))
          (cond
            ((wav-tag= data p "fmt ")
             (when (>= chunk-size 16)
               (setf audio-format    (le-u16 data (+ p  8))
                     channels        (le-u16 data (+ p 10))
                     sample-rate     (le-u32 data (+ p 12))
                     bits-per-sample (le-u16 data (+ p 22)))))
            ((wav-tag= data p "data")
             (setf pcm-offset (+ p 8)
                   pcm-size   (min chunk-size (- file-size (+ p 8))))))
          (incf p (+ 8 chunk-size))
          (when (oddp chunk-size) (incf p))))

      (unless (and (= audio-format 1)
                   (= bits-per-sample 16)
                   pcm-offset
                   (>= channels 1))
        (error "load-wav: unsupported format in ~a (need 16-bit PCM, got fmt=~a bits=~a)"
               path audio-format bits-per-sample))

      ;; Decode interleaved s16le PCM → mono single-float
      (let* ((bytes-per-frame (* channels 2))
             (n-frames        (floor pcm-size bytes-per-frame))
             (samples         (make-array n-frames :element-type 'single-float)))
        (loop for i below n-frames
              for frame-base = (+ pcm-offset (* i bytes-per-frame))
              do
          (if (= channels 1)
              (let* ((b0  (aref data frame-base))
                     (b1  (aref data (1+ frame-base)))
                     (u   (logior b0 (ash b1 8)))
                     (s16 (if (>= u 32768) (- u 65536) u)))
                (setf (aref samples i) (* (float s16 1.0f0) (/ 1.0f0 32768.0f0))))
              (let ((sum 0.0f0))
                (loop for c below channels do
                  (let* ((off (+ frame-base (* c 2)))
                         (b0  (aref data off))
                         (b1  (aref data (1+ off)))
                         (u   (logior b0 (ash b1 8)))
                         (s16 (if (>= u 32768) (- u 65536) u)))
                    (incf sum (float s16 1.0f0))))
                (setf (aref samples i)
                      (/ (/ sum (float channels 1.0f0)) 32768.0f0)))))

        ;; Resample to 16 kHz if the source rate differs
        (if (/= sample-rate +sample-rate+)
            (resample-kaiser samples sample-rate +sample-rate+)
            samples)))))

;;; =========================================================================
;;; Mel filterbank helpers
;;; =========================================================================

(defun hertz->mel (freq)
  "Slaney Hz→mel: linear below 1 kHz, log above."
  (let ((f           (float freq 1.0f0))
        (min-log-hz  1000.0f0)
        (min-log-mel 15.0f0)
        (logstep     (/ 27.0f0 (log 6.4f0))))
    (if (>= f min-log-hz)
        (+ min-log-mel (* logstep (log (/ f min-log-hz))))
        (* 3.0f0 f (/ 1.0f0 200.0f0)))))

(defun mel->hertz (mels)
  "Slaney mel→Hz inverse."
  (let ((m           (float mels 1.0f0))
        (min-log-hz  1000.0f0)
        (min-log-mel 15.0f0)
        (logstep     (/ (log 6.4f0) 27.0f0)))
    (if (>= m min-log-mel)
        (* min-log-hz (exp (* logstep (- m min-log-mel))))
        (* 200.0f0 m (/ 1.0f0 3.0f0)))))

(defun build-mel-filters ()
  "Build Slaney-style triangular mel filterbank.
Returns (simple-array single-float (128 201)) — filters[mel-bin, fft-bin]."
  (let* ((n-pts        (+ +n-mel+ 2))                ; 130
         (mel-min      (hertz->mel 0.0f0))
         (mel-max      (hertz->mel (* +sample-rate+ 0.5f0)))
         (fft-freqs    (make-array +n-freq+  :element-type 'single-float))
         (filter-freqs (make-array n-pts     :element-type 'single-float))
         (filter-diff  (make-array (1+ +n-mel+) :element-type 'single-float))
         (filters      (make-array (list +n-mel+ +n-freq+)
                                   :element-type 'single-float
                                   :initial-element 0.0f0)))

    ;; FFT bin center frequencies: i * (sr/2) / (N_FREQ-1)
    (dotimes (i +n-freq+)
      (setf (aref fft-freqs i)
            (* (float i 1.0f0)
               (/ (* +sample-rate+ 0.5f0) (float (1- +n-freq+) 1.0f0)))))

    ;; N_MEL+2 evenly-spaced mel points → Hz  (matches C (float)i / (float)(N_MEL+1))
    (dotimes (i n-pts)
      (let ((mel (+ mel-min (* (- mel-max mel-min)
                               (/ (float i 1.0f0) (float (1- n-pts) 1.0f0))))))
        (setf (aref filter-freqs i) (float (mel->hertz mel) 1.0f0))))

    ;; Successive differences (denominator for slope)
    (dotimes (i (1+ +n-mel+))
      (let ((d (- (aref filter-freqs (1+ i)) (aref filter-freqs i))))
        (setf (aref filter-diff i) (if (= d 0.0f0) 1.0f-6 d))))

    ;; Slaney triangular filter: min(down, up), clamped ≥ 0, times enorm
    (dotimes (m +n-mel+)
      (let ((enorm (/ 2.0f0 (- (aref filter-freqs (+ m 2)) (aref filter-freqs m)))))
        (dotimes (f +n-freq+)
          (let* ((ff   (aref fft-freqs f))
                 (down (/ (- ff (aref filter-freqs m))       (aref filter-diff m)))
                 (up   (/ (- (aref filter-freqs (+ m 2)) ff) (aref filter-diff (1+ m))))
                 (val  (max 0.0f0 (min down up))))
            (setf (aref filters m f) (* val enorm))))))

    filters))

;;; =========================================================================
;;; Mel spectrogram
;;; =========================================================================

(defun mel-spectrogram (samples)
  "Compute log-mel spectrogram.
SAMPLES: (simple-array single-float (*)) at 16 kHz.
Returns: (simple-array single-float (128 n-frames)) — layout [mel-bin, frame].

Steps: reflect-pad → windowed DFT → power → mel filterbank → log10 →
       dynamic-max normalisation."
  (let* ((n-samples  (length samples))
         (pad-len    (floor +n-fft+ 2))        ; 200
         (padded-len (+ n-samples (* 2 pad-len)))
         (padded     (make-array padded-len :element-type 'single-float)))

    ;; Reflect-pad left: samples[pad_len - i] for i in [0, pad_len)
    (dotimes (i pad-len)
      (let ((src (- pad-len i)))
        (setf (aref padded i)
              (if (< src n-samples) (aref samples src) 0.0f0))))

    ;; Center copy
    (dotimes (i n-samples)
      (setf (aref padded (+ pad-len i)) (aref samples i)))

    ;; Reflect-pad right: samples[n_samples - 2 - i] for i in [0, pad_len)
    (dotimes (i pad-len)
      (let ((src (- n-samples 2 i)))
        (setf (aref padded (+ pad-len n-samples i))
              (if (>= src 0) (aref samples src) 0.0f0))))

    (let* ((n-frames-total (1+ (floor (- padded-len +n-fft+) +hop-length+)))
           (n-frames       (1- n-frames-total)))   ; drop last frame, matches C
      (when (<= n-frames 0)
        (error "mel-spectrogram: audio too short (~a samples)" n-samples))

      (let* ((mel-filters (build-mel-filters))
             ;; Periodic Hann window
             (window      (make-array +win-length+ :element-type 'single-float))
             ;; Precomputed DFT tables [N-FREQ × N-FFT]
             (dft-cos     (make-array (list +n-freq+ +n-fft+) :element-type 'single-float))
             (dft-sin     (make-array (list +n-freq+ +n-fft+) :element-type 'single-float))
             ;; Intermediate mel values stored as [n-frames × N-MEL] for pass 1
             (mel-tmp     (make-array (list n-frames +n-mel+) :element-type 'single-float))
             ;; Final output [N-MEL × n-frames]
             (mel-out     (make-array (list +n-mel+ n-frames) :element-type 'single-float))
             ;; Per-frame work buffers
             (windowed    (make-array +n-fft+  :element-type 'single-float))
             (power       (make-array +n-freq+ :element-type 'single-float))
             ;; Precompute 2π and log10 denominator in double for table accuracy
             (two-pi      (* 2.0d0 pi))
             (log10-denom (log 10.0d0)))

        ;; Periodic Hann: 0.5*(1 - cos(2π·i / WIN_LENGTH))
        (dotimes (i +win-length+)
          (setf (aref window i)
                (float (* 0.5d0 (- 1.0d0 (cos (* two-pi
                                                   (float i 1.0d0)
                                                   (/ 1.0d0 (float +win-length+ 1.0d0))))))
                       1.0f0)))

        ;; DFT tables: cos/sin at angles 2π·k·n / N_FFT
        (dotimes (k +n-freq+)
          (dotimes (n +n-fft+)
            (let ((angle (* two-pi
                            (float k 1.0d0)
                            (float n 1.0d0)
                            (/ 1.0d0 (float +n-fft+ 1.0d0)))))
              (setf (aref dft-cos k n) (float (cos angle) 1.0f0))
              (setf (aref dft-sin k n) (float (sin angle) 1.0f0)))))

        ;; ─── Pass 1: STFT → power → mel energy → log10, track global max ───
        (let ((global-max -1.0f30))
          (dotimes (frame n-frames)
            (let ((start (* frame +hop-length+)))

              ;; Apply Hann window to the current frame
              (dotimes (i +n-fft+)
                (setf (aref windowed i)
                      (* (aref padded (+ start i)) (aref window i))))

              ;; DFT: re[k] = Σ windowed[n]*cos, im[k] = Σ windowed[n]*sin
              ;; power[k] = re² + im²
              (dotimes (k +n-freq+)
                (let ((re 0.0f0) (im 0.0f0))
                  (dotimes (n +n-fft+)
                    (incf re (* (aref windowed n) (aref dft-cos k n)))
                    (incf im (* (aref windowed n) (aref dft-sin k n))))
                  (setf (aref power k) (+ (* re re) (* im im)))))

              ;; Mel filterbank + log10
              (dotimes (m +n-mel+)
                (let ((sum 0.0f0))
                  (dotimes (f +n-freq+)
                    (incf sum (* (aref mel-filters m f) (aref power f))))
                  ;; log10(max(sum, 1e-10))
                  (let ((val (float (/ (log (float (max sum 1.0f-10) 1.0d0))
                                       log10-denom)
                                    1.0f0)))
                    (setf (aref mel-tmp frame m) val)
                    (when (> val global-max)
                      (setf global-max val)))))))

          ;; ─── Pass 2: clamp with dynamic max, normalize, transpose ───
          ;; val = clamp(log_mel, global_max - 8, global_max)
          ;; output = (val + 4) / 4
          (let ((min-val (- global-max 8.0f0)))
            (dotimes (frame n-frames)
              (dotimes (m +n-mel+)
                (let ((val (max (aref mel-tmp frame m) min-val)))
                  (setf (aref mel-out m frame)
                        (* (+ val 4.0f0) (/ 1.0f0 4.0f0)))))))

          mel-out)))))
