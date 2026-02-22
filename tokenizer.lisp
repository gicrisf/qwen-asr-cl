;;;; tokenizer.lisp — GPT-2 byte-level BPE tokenizer (Qwen3-ASR)
;;;;
;;;; Mirrors qwen_asr_tokenizer.c.  Loads vocab.json + merges.txt from
;;;; the model directory and supports encode (text → token IDs) and
;;;; decode (token ID → text).

(in-package #:qwen-asr)

;;; ─────────────────────────────────────────────────────────────────────
;;; GPT-2 byte ↔ unicode mapping
;;; ─────────────────────────────────────────────────────────────────────
;;;
;;; Bytes that are "normal" printable ASCII (33–126, 161–172, 174–255)
;;; map to themselves.  The remaining 68 bytes map to codepoints 256–323.
;;; This creates a bijection: every byte → one unique printable codepoint.

(defun make-gpt2-tables ()
  "Return (values byte->unicode unicode->byte).
   byte->unicode : (simple-array fixnum (256))  byte b → codepoint
   unicode->byte : (simple-array fixnum (512))  codepoint → byte (-1 = none)"
  (let ((b->u (make-array 256 :element-type 'fixnum :initial-element 0))
        (u->b (make-array 512 :element-type 'fixnum :initial-element -1)))
    (let ((n 0))
      (dotimes (b 256)
        (if (or (and (>= b 33)  (<= b 126))
                (and (>= b 161) (<= b 172))
                (and (>= b 174) (<= b 255)))
            (setf (aref b->u b) b)
            (progn
              (setf (aref b->u b) (+ 256 n))
              (incf n)))))
    (dotimes (b 256)
      (let ((cp (aref b->u b)))
        (when (< cp 512)
          (setf (aref u->b cp) b))))
    (values b->u u->b)))

;;; ─────────────────────────────────────────────────────────────────────
;;; Tokenizer struct
;;; ─────────────────────────────────────────────────────────────────────

(defstruct tokenizer
  "GPT-2 byte-level BPE tokenizer for Qwen3-ASR."
  (id-to-text    #()  :type simple-vector)          ; string per token id
  (vocab-map     (make-hash-table :test #'equal)
                      :type hash-table)             ; bpe-string → id
  (merge-map     (make-hash-table :test #'equal)
                      :type hash-table)             ; "a b" → rank
  (vocab-size    0    :type fixnum)
  (byte->unicode (make-array 256 :element-type 'fixnum)
                      :type (simple-array fixnum (256))))

;;; ─────────────────────────────────────────────────────────────────────
;;; Internal: BPE vocab key → raw text
;;; ─────────────────────────────────────────────────────────────────────

(defun decode-gpt2-token (bpe-string unicode->byte)
  "Map each character of BPE-STRING back through UNICODE->BYTE, producing
   a byte vector, then decode as UTF-8.  One BPE character = one byte."
  (let ((bytes (make-array (length bpe-string) :element-type '(unsigned-byte 8))))
    (loop for i below (length bpe-string)
          for cp = (char-code (char bpe-string i))
          do (setf (aref bytes i)
                   (let ((b (if (< cp 512) (aref unicode->byte cp) -1)))
                     (if (>= b 0) b (char-code #\?)))))
    (handler-case
        (babel:octets-to-string bytes :encoding :utf-8)
      (error () ""))))

;;; ─────────────────────────────────────────────────────────────────────
;;; Load vocab.json
;;; ─────────────────────────────────────────────────────────────────────

(defun load-vocab (path unicode->byte)
  "Parse PATH (vocab.json).  Return (values id-to-text vocab-map vocab-size).
   id-to-text : simple-vector of strings indexed by token id
   vocab-map  : hash-table bpe-string → id"
  (let ((json-obj (jzon:parse (uiop:read-file-string path)))
        (max-id   0))
    ;; first pass: find max id
    (maphash (lambda (k id)
               (declare (ignore k))
               (when (> id max-id) (setf max-id id)))
             json-obj)
    (let* ((vocab-size (1+ max-id))
           (id-to-text (make-array vocab-size :initial-element ""))
           (vocab-map  (make-hash-table :test #'equal :size vocab-size)))
      ;; second pass: fill tables
      (maphash (lambda (bpe-str id)
                 (let ((text (decode-gpt2-token bpe-str unicode->byte)))
                   (setf (aref id-to-text id) text
                         (gethash bpe-str vocab-map) id)))
               json-obj)
      (values id-to-text vocab-map vocab-size))))

;;; ─────────────────────────────────────────────────────────────────────
;;; Load merges.txt
;;; ─────────────────────────────────────────────────────────────────────

(defun load-merges (path)
  "Parse PATH (merges.txt).  Return hash-table mapping 'a b' → rank (0-based).
   Lines starting with '#' or blank lines are skipped."
  (let ((merge-map (make-hash-table :test #'equal))
        (rank 0))
    (with-open-file (f path :direction :input :if-does-not-exist nil)
      (when f
        (loop for raw = (read-line f nil nil)
              while raw
              do (let ((line (string-right-trim '(#\Return #\Newline) raw)))
                   (when (and (plusp (length line))
                              (char/= (char line 0) #\#))
                     (let ((sp (position #\Space line)))
                       (when sp
                         (let* ((a (subseq line 0 sp))
                                (b (string-left-trim
                                    '(#\Space)
                                    (subseq line (1+ sp)))))
                           (when (and (plusp (length a))
                                      (plusp (length b)))
                             (setf (gethash (concatenate 'string a " " b)
                                            merge-map)
                                   rank)
                             (incf rank))))))))))
    merge-map))

;;; ─────────────────────────────────────────────────────────────────────
;;; Top-level loader
;;; ─────────────────────────────────────────────────────────────────────

(defun load-tokenizer (model-dir)
  "Load a BPE tokenizer from MODEL-DIR (must contain vocab.json and merges.txt).
   Returns a TOKENIZER struct."
  (let* ((dir   (uiop:ensure-directory-pathname model-dir))
         (vpath (merge-pathnames "vocab.json" dir))
         (mpath (merge-pathnames "merges.txt" dir)))
    (multiple-value-bind (b->u u->b) (make-gpt2-tables)
      (multiple-value-bind (id-to-text vocab-map vocab-size)
          (load-vocab vpath u->b)
        (make-tokenizer :id-to-text    id-to-text
                        :vocab-map     vocab-map
                        :merge-map     (load-merges mpath)
                        :vocab-size    vocab-size
                        :byte->unicode b->u)))))

;;; ─────────────────────────────────────────────────────────────────────
;;; Decode
;;; ─────────────────────────────────────────────────────────────────────

(defun tokenizer-decode (tok id)
  "Return the text string for token ID (empty string if out of range)."
  (if (and (>= id 0) (< id (tokenizer-vocab-size tok)))
      (aref (tokenizer-id-to-text tok) id)
      ""))

;;; ─────────────────────────────────────────────────────────────────────
;;; Encode
;;; ─────────────────────────────────────────────────────────────────────

(defun text->bpe-unicode (text byte->unicode)
  "Convert TEXT to a GPT-2 BPE unicode string.
   Each input byte is mapped to its corresponding Unicode codepoint."
  (let* ((raw   (babel:string-to-octets text :encoding :utf-8))
         (chars (make-array (length raw) :element-type 'character)))
    (loop for i below (length raw)
          do (setf (aref chars i)
                   (code-char (aref byte->unicode (aref raw i)))))
    (coerce chars 'string)))

(defun bpe-merge (symbols merge-map)
  "Apply BPE merges to SYMBOLS (a sequence of strings).
   Repeatedly finds the adjacent pair with the lowest rank and merges it.
   Returns a list of the final symbols."
  (let ((syms (coerce symbols 'vector)))
    (loop
      (when (<= (length syms) 1) (return))
      (let ((best-rank most-positive-fixnum)
            (best-i    -1))
        (loop for i below (1- (length syms))
              for r = (or (gethash (concatenate 'string
                                                (aref syms i)
                                                " "
                                                (aref syms (1+ i)))
                                   merge-map)
                          most-positive-fixnum)
              when (< r best-rank)
                do (setf best-rank r best-i i))
        (when (or (= best-i -1) (= best-rank most-positive-fixnum))
          (return))
        ;; Merge syms[best-i] and syms[best-i+1] into one symbol
        (let* ((n-old  (length syms))
               (n-new  (1- n-old))
               (merged (concatenate 'string (aref syms best-i) (aref syms (1+ best-i))))
               (new    (make-array n-new)))
          (loop for i below best-i            do (setf (aref new i)       (aref syms i)))
          (setf (aref new best-i) merged)
          (loop for i from (+ best-i 2) below n-old do (setf (aref new (1- i)) (aref syms i)))
          (setf syms new))))
    (coerce syms 'list)))

(defun tokenizer-encode (tok text)
  "Encode TEXT string to a (vector fixnum) of token IDs using BPE."
  (when (or (null text) (zerop (length text)))
    (return-from tokenizer-encode (make-array 0 :element-type 'fixnum)))
  (let* ((mapped  (text->bpe-unicode text (tokenizer-byte->unicode tok)))
         ;; Split mapped string into single-character symbol strings
         (symbols (loop for ch across mapped collect (string ch)))
         (merged  (bpe-merge symbols (tokenizer-merge-map tok)))
         (n       (length merged))
         (ids     (make-array n :element-type 'fixnum)))
    (loop for sym in merged
          for i from 0
          do (let ((id (gethash sym (tokenizer-vocab-map tok))))
               (if id
                   (setf (aref ids i) id)
                   (error "tokenizer-encode: symbol not in vocab: ~S" sym))))
    ids))
