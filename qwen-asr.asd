;;;; qwen-asr.asd

(asdf:defsystem #:qwen-asr
  :description "Inference pipeline for Qwen3 ASR models."
  :author "Giovanni Crisalfi <giovanni.crisalfi@protonmail.com>"
  :license  ""
  :version "0.0.1"
  :serial t
  :depends-on (#:com.inuoe.jzon
               #:babel
               #:cffi
               #:lla)
  :components ((:file "package")
               (:file "safetensors")
               (:file "tokenizer")
               (:file "audio")
               (:file "kernels")
               (:file "encoder")
               (:file "decoder")
               (:file "core")))
