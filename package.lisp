;;;; package.lisp

(defpackage #:qwen-asr
  (:use #:cl)
  (:local-nicknames (#:jzon #:com.inuoe.jzon))
  (:export
   ;; safetensors
   #:open-model
   #:find-tensor
   #:open-safetensors
   #:tensor-f32
   #:tensor-raw-bytes
   #:tensor-numel
   ;; safetensor accessors
   #:safetensor-name
   #:safetensor-dtype
   #:safetensor-shape
   #:safetensor-offset
   #:safetensor-nbytes
   ;; safetensors-file accessors
   #:safetensors-file-path
   #:safetensors-file-tensors
   ;; multi-safetensors accessors
   #:multi-safetensors-shards
   ;; tokenizer
   #:load-tokenizer
   #:tokenizer-encode
   #:tokenizer-decode
   #:tokenizer-vocab-size))
