
(define (script-fu-microscope-save-grid image path)
    (let* (
        ; Get the result from gimp-image-get-layers
        (layers-result (gimp-image-get-layers image))
        ; Extract the number of layers
        (num-layers (car layers-result)) 
        ; Extract the vector of layer IDs
        (layer-ids (cadr layers-result))
        ; Initialize an empty list to store filenames and offsets
        (layer-info '())
    )
    ; Loop through layers
    (do ((i 0 (+ i 1)))
        ((= i num-layers))
    (let* ((layer-id (aref layer-ids i))
            (layer-name (car (gimp-item-get-name layer-id)))
            (layer-offsets (gimp-drawable-offsets layer-id))
            (layer-x-offset (car layer-offsets))
            (layer-y-offset (cadr layer-offsets)))
    ; And append
    (set! layer-info (cons (list layer-name layer-x-offset layer-y-offset) layer-info))
    )
    )
    ; Run Saving
    (savedata layer-info path)
    )
)

(define (fix-path path)
  (if (not (char=? (string-ref path (- (string-length path) 1)) #\/))
      (string-append path "/")  ; Append "/" if no trailing slash
      path)
)  

(define (savedata layer-info path)
    ; Fix path and append grid.txt
    (set! path (fix-path path))
    (define output-file-path (string-append path "grid.txt"))
    (let ((output-file (open-output-file output-file-path)))
        (if output-file ; Do error checking in case no write permission or invalid path
        (begin
        (for-each
            ;Use lambda function to create string and push to output-file: display targets open stream?
            (lambda (layer)
                (display (string-append (car layer)
                              "," (number->string (cadr layer))
                              "," (number->string (caddr layer))
                              "\n")
                output-file)
                ) ; Lambda End 
        (reverse layer-info)) ; For each end
        (gimp-message (string-append "Layer info written to " output-file-path "\n")) 
        ) ; End Begin 
        (gimp-message "Failed to write layer information."))
    )
)









; register the script, what sort of type is this?
 (script-fu-register
    "script-fu-microscope-save-grid"                        ;function name
    "Save Microscope Grid"                                  ;menu label
    "Saves images from path and places them based on name"  ;description
    "Felix Semler"                             ;author
    "Felix Semler"        ;copyright notice
    "2024"                          ;date created
    ""                                      ;image type that the script works on
    SF-IMAGE       "Image"         0
    SF-STRING      "path"          "/home/felix/RapidStorage2/GarlicShell/"   ;a string variable
  )
  (script-fu-menu-register "script-fu-microscope-save-grid" "<Image>/Microscope/")