;;; coarch.el --- Coarch code search for Emacs  -*- lexical-binding: t; -*-

;; Copyright (C) 2025 syedazeez337

;; Author: syedazeez337 <syedazeez337@github.com>
;; Version: 0.1.0
;; Package-Requires: ((emacs "27.1"))
;; License: MIT

;;; Commentary:

;; Coarch is a local-first semantic code search engine.
;; This package provides Emacs integration.

;;; Code:

(require 'json)
(require 'url)
(require 'url-http)

(defgroup coarch nil
  "Coarch code search integration."
  :group 'tools
  :prefix "coarch-")

(defcustom coarch-server-url "http://localhost:8000"
  "URL of the Coarch server."
  :type 'string
  :group 'coarch)

(defcustom coarch-max-results 20
  "Maximum number of results to return."
  :type 'integer
  :group 'coarch)

(defcustom coarch-display-buffer-fn #'switch-to-buffer-other-window
  "Function to display results buffer."
  :type 'function
  :group 'coarch)

(defun coarch-search (query &optional language)
  "Search for QUERY using Coarch.
Optional LANGUAGE to filter by programming language."
  (interactive "sSearch query: ")
  (let ((url (concat coarch-server-url "/search"))
        (body (json-encode `(("query" . ,query)
                            ("limit" . ,coarch-max-results)
                            ("language" . ,language)))))
    (url-http-post url body
      (lambda (status)
        (if (eq status 200)
            (coarch-display-results (json-read-from-string (buffer-string)) query)
          (message "Coarch search failed: %s" (buffer-string)))))))

(defun coarch-display-results (results query)
  "Display search RESULTS in a buffer.
QUERY is the original search query."
  (let ((buf (get-buffer-create "*Coarch Results*")))
    (funcall coarch-display-buffer-fn buf)
    (with-current-buffer buf
      (coarch-mode)
      (setq header-line-format (format "Coarch: %s" query))
      (erase-buffer)
      (insert (format "Search results for: %s\n\n" query))
      (dolist (result results)
        (let ((file (assoc-default 'file_path result))
              (lines (assoc-default 'lines result))
              (score (assoc-default 'score result))
              (lang (assoc-default 'language result)))
          (insert (format "[%.2f] %s:%s (%s)\n"
                          score file lines lang))))
      (goto-char (point-min)))))

(defun coarch-index-repo (path &optional name)
  "Index repository at PATH.
Optional NAME for the repository."
  (interactive "DDirectory: ")
  (let* ((name (or name (file-name-base (directory-file-name path))))
         (url (concat coarch-server-url "/index/repo"))
         (body (json-encode `(("path" . ,path) ("name" . ,name)))))
    (message "Indexing repository: %s" name)
    (url-http-post url body
      (lambda (status)
        (if (eq status 200)
            (message "Repository indexed successfully")
          (message "Indexing failed: %s" (buffer-string)))))))

(defun coarch-status ()
  "Show Coarch server status."
  (interactive)
  (let ((url (concat coarch-server-url "/index/status")))
    (url-http-get url
      (lambda (status)
        (if (eq status 200)
            (message "Coarch is ready")
          (message "Coarch server not responding"))))))

(defun coarch-serve ()
  "Start Coarch server."
  (interactive)
  (message "Starting Coarch server...")
  (start-process "coarch-server" "*Coarch Server*" "python" "-m" "backend.server")
  (message "Coarch server started"))

(define-derived-mode coarch-mode special-mode "Coarch"
  "Major mode for Coarch search results."
  (setq-local truncate-lines t))

;; Key bindings
(define-key coarch-mode-map (kbd "RET") 'coarch-open-result)
(define-key coarch-mode-map (kbd "o") 'coarch-open-result-other-window)

(defun coarch-open-result ()
  "Open the result at point."
  (interactive)
  (let* ((line (thing-at-point 'line t))
         (file (car (split-string (substring line 1) " "))))
    (find-file file)))

(defun coarch-open-result-other-window ()
  "Open the result at point in other window."
  (interactive)
  (let* ((line (thing-at-point 'line t))
         (file (car (split-string (substring line 1) " "))))
    (find-file-other-window file)))

;; Commands
(global-set-key (kbd "C-c s s") 'coarch-search)
(global-set-key (kbd "C-c s w") (lambda () (interactive) (coarch-search (word-at-point))))
(global-set-key (kbd "C-c s i") 'coarch-index-repo)
(global-set-key (kbd "C-c s c") 'coarch-status)

(provide 'coarch)
;;; coarch.el ends here
