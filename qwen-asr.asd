;;;; qwen-asr.asd

(asdf:defsystem #:qwen-asr
  :description "Inference pipeline for Qwen3 ASR models."
  :author "Giovanni Crisalfi <giovanni.crisalfi@protonmail.com>"
  :license  ""
  :version "0.0.1"
  :serial t
  :components ((:file "package")
               (:file "qwen-asr")))
