(in-package #:qwen-asr)

;;;; safetensors.lisp — load safetensors weight files
;;;;
;;;; Binary layout:
;;;;   [8 bytes, LE uint64]  — byte length of the JSON header
;;;;   [N bytes, UTF-8]      — JSON: tensor-name → {dtype, shape, data_offsets}
;;;;   [raw bytes]           — concatenated tensor data

;;; ─────────────────────────────────────────────────────────────────────
;;; Byte helpers
;;; ─────────────────────────────────────────────────────────────────────

(defun read-le-u16 (bytes offset)
  (logior (aref bytes offset)
          (ash (aref bytes (1+ offset)) 8)))

(defun read-le-u32 (bytes offset)
  (logior (aref bytes offset)
          (ash (aref bytes (+ offset 1)) 8)
          (ash (aref bytes (+ offset 2)) 16)
          (ash (aref bytes (+ offset 3)) 24)))

(defun read-le-u64 (bytes offset)
  (loop for i below 8
        sum (ash (aref bytes (+ offset i)) (* 8 i))))

(defun bf16->f32 (u16)
  "Reinterpret a bf16 bit pattern as single-float.
   BF16 and F32 share the same sign + exponent layout; bf16 is just
   F32 with the lower 16 mantissa bits dropped, so shifting left 16
   gives a valid F32 bit pattern."
  (cffi:with-foreign-object (p :uint32)
    (setf (cffi:mem-ref p :uint32) (ash u16 16))
    (cffi:mem-ref p :float)))

(defun u32->f32 (u32)
  "Reinterpret a uint32 bit pattern as single-float."
  (cffi:with-foreign-object (p :uint32)
    (setf (cffi:mem-ref p :uint32) u32)
    (cffi:mem-ref p :float)))

;;; ─────────────────────────────────────────────────────────────────────
;;; Structs
;;; ─────────────────────────────────────────────────────────────────────

(defstruct safetensor
  "Metadata for one tensor within a safetensors file."
  (name   "" :type string)
  (dtype  :unknown :type keyword)   ; :f32 :f16 :bf16 :i32 :i64 :bool
  (shape  #() :type vector)         ; vector of dimension sizes
  (offset 0   :type (unsigned-byte 64)) ; byte offset within the data section
  (nbytes 0   :type (unsigned-byte 64)))

(defstruct safetensors-file
  "A single loaded safetensors shard."
  (path       ""  :type string)
  (bytes      #() :type (simple-array (unsigned-byte 8) (*))) ; entire file
  (data-start 0   :type (unsigned-byte 64))  ; = 8 + header-size
  (tensors    (make-hash-table :test #'equal) :type hash-table)) ; name → safetensor

(defstruct multi-safetensors
  "All shards for a model, providing a unified tensor namespace."
  (shards '() :type list)) ; list of safetensors-file

;;; ─────────────────────────────────────────────────────────────────────
;;; JSON → tensor metadata
;;; ─────────────────────────────────────────────────────────────────────

(defun parse-dtype (s)
  (cond ((string= s "F32")  :f32)
        ((string= s "F16")  :f16)
        ((string= s "BF16") :bf16)
        ((string= s "I32")  :i32)
        ((string= s "I64")  :i64)
        ((string= s "BOOL") :bool)
        (t :unknown)))

(defun parse-tensor-entry (name obj)
  "Build a SAFETENSOR from jzon-parsed object OBJ for tensor NAME."
  (let* ((offsets (gethash "data_offsets" obj))
         (start   (aref offsets 0))
         (end     (aref offsets 1)))
    (make-safetensor :name   name
                     :dtype  (parse-dtype (gethash "dtype" obj ""))
                     :shape  (gethash "shape" obj #())
                     :offset start
                     :nbytes (- end start))))

;;; ─────────────────────────────────────────────────────────────────────
;;; Single-file loader
;;; ─────────────────────────────────────────────────────────────────────

(defun read-file-bytes (path)
  (with-open-file (s path :element-type '(unsigned-byte 8))
    (let ((buf (make-array (file-length s) :element-type '(unsigned-byte 8))))
      (read-sequence buf s)
      buf)))

(defun open-safetensors (path)
  "Load a single safetensors file from PATH.
   Returns a SAFETENSORS-FILE or signals an error."
  (let* ((bytes       (read-file-bytes path))
         (header-size (read-le-u64 bytes 0))
         (data-start  (+ 8 header-size))
         (json-str    (babel:octets-to-string bytes :start 8 :end data-start))
         (header      (jzon:parse json-str))
         (tensors     (make-hash-table :test #'equal)))
    (maphash (lambda (name obj)
               (unless (string= name "__metadata__")
                 (setf (gethash name tensors)
                       (parse-tensor-entry name obj))))
             header)
    (make-safetensors-file :path       (namestring path)
                           :bytes      bytes
                           :data-start data-start
                           :tensors    tensors)))

;;; ─────────────────────────────────────────────────────────────────────
;;; Multi-shard support
;;; ─────────────────────────────────────────────────────────────────────

(defun shard-file-p (path)
  "True if PATH looks like a shard file (model-NNNNN-of-NNNNN.safetensors)."
  (and (string= (pathname-type path) "safetensors")
       (let ((name (pathname-name path)))
         (and (>= (length name) 6)
              (string= name "model-" :end1 6)))))

(defun open-model (model-dir)
  "Open a model directory.
   Tries model.safetensors first; falls back to sorted multi-shard files.
   Returns a MULTI-SAFETENSORS."
  (let* ((dir    (uiop:ensure-directory-pathname model-dir))
         (single (merge-pathnames "model.safetensors" dir)))
    (if (probe-file single)
        (make-multi-safetensors :shards (list (open-safetensors single)))
        (let ((shards (sort (remove-if-not #'shard-file-p
                                           (uiop:directory-files dir))
                            #'string< :key #'namestring)))
          (when (null shards)
            (error "No safetensors files found in ~A" model-dir))
          (make-multi-safetensors
           :shards (mapcar #'open-safetensors shards))))))

(defun find-tensor (ms name)
  "Find tensor NAME across all shards of MS.
   Returns (values safetensors-file safetensor), or (values nil nil) if not found."
  (dolist (shard (multi-safetensors-shards ms))
    (let ((tensor (gethash name (safetensors-file-tensors shard))))
      (when tensor
        (return-from find-tensor (values shard tensor)))))
  (values nil nil))

;;; ─────────────────────────────────────────────────────────────────────
;;; Tensor data access
;;; ─────────────────────────────────────────────────────────────────────

(defun tensor-numel (tensor)
  "Total number of elements in TENSOR."
  (reduce #'* (safetensor-shape tensor) :initial-value 1))

(defun tensor-raw-bytes (sf tensor)
  "Return the raw bytes of TENSOR as a displaced array into SF's buffer (no copy)."
  (make-array (safetensor-nbytes tensor)
              :element-type '(unsigned-byte 8)
              :displaced-to (safetensors-file-bytes sf)
              :displaced-index-offset (+ (safetensors-file-data-start sf)
                                         (safetensor-offset tensor))))

(defun tensor-f32 (sf tensor)
  "Return tensor data as a (simple-array single-float (*)).
   Allocates fresh storage. Supports :f32 and :bf16 source dtypes."
  (let* ((n    (tensor-numel tensor))
         (out  (make-array n :element-type 'single-float))
         (data (safetensors-file-bytes sf))
         (base (+ (safetensors-file-data-start sf)
                  (safetensor-offset tensor))))
    (ecase (safetensor-dtype tensor)
      (:f32
       (loop for i below n
             do (setf (aref out i) (u32->f32 (read-le-u32 data (+ base (* i 4)))))))
      (:bf16
       (loop for i below n
             do (setf (aref out i) (bf16->f32 (read-le-u16 data (+ base (* i 2))))))))
    out))

(defun tensor-f32-2d (sf tensor)
  "Return tensor data as a (simple-array single-float (rows cols)).
Tensor must be rank-2. Allocates fresh storage. Supports :f32 and :bf16."
  (let* ((shape (safetensor-shape tensor)))
    (unless (= (length shape) 2)
      (error "tensor-f32-2d: expected rank-2 tensor, got shape ~a" shape))
    (let* ((rows (aref shape 0))
           (cols (aref shape 1))
           (out  (make-array (list rows cols) :element-type 'single-float))
           (data (safetensors-file-bytes sf))
           (base (+ (safetensors-file-data-start sf)
                    (safetensor-offset tensor)))
           (n    (* rows cols)))
      (ecase (safetensor-dtype tensor)
        (:f32
         (loop for i below n
               do (setf (row-major-aref out i)
                        (u32->f32 (read-le-u32 data (+ base (* i 4)))))))
        (:bf16
         (loop for i below n
               do (setf (row-major-aref out i)
                        (bf16->f32 (read-le-u16 data (+ base (* i 2))))))))
      out)))
