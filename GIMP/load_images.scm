(define (script-fu-microscope-load-grid path)
    ; Grab and extract
    (gimp-message "Starting Load Grid")
    (set! path (fix-path path))
    (let* ((results (load-grid path))
       (y-list (remove-duplicates (car results)))
       (z-list (remove-duplicates (cadr results)))
       (e-list (remove-duplicates (caddr results))))
    ; Now we need to determine the image width and height 
    (define steps-to-px (/ (/ 0.5 4096) 1.55e-3)) ; 0.5 /4096 /2 /1.55e-3
    (define single_image_height 2028);(- (list-ref y-list 1) (list-ref y-list 2)))
    ; z increases to the left
    (define single_image_width 1520);(- (list-ref z-list 1) (list-ref z-list 2)))    
    (define width (+ (* (max-list z-list) steps-to-px) single_image_width))
    (define height (+ (* (max-list y-list) steps-to-px) single_image_height))
    (display "height") (newline)


    ;(error "Readddd")
    ; Make int 
    (set! width (truncate width))
    (set! height (truncate height))
    ; Finally we make a goddamn image, we need to extract the first item as all gimp 
    ; functions return a list
    (define image (car (gimp-image-new width height RGB-IMAGE)))
    (define e-value (car e-list))
    (define z-max (max-list z-list))
    ; We now iterate y and z to populate the image
    ; LOOP START IDK I hope this works GPT wrote it
    (let* ((y-length (length y-list))
           (z-length (length z-list)))
    (for-each (lambda (y)
        (for-each (lambda (z)
                            ; Make Filename
                    (let* ((filename (construct-filename y z e-value path))  
                            ; Make Layer
                           (layer (car (gimp-file-load-layer 0 image filename))))
                           ; Add layer to imag
                           (gimp-image-insert-layer image layer 0 -1)
                            ; Compute and set position
                           (gimp-layer-set-offsets layer (truncate (* (abs (- z z-max)) steps-to-px)) (truncate (* y steps-to-px)))))
            z-list)) ; these two are whats being iterated by lambda
        y-list))
    ;LOOP END
    (gimp-display-new image)
    )
)

(define (remove-duplicates lst)
  (define (contains? elem lst)
    (cond ((null? lst) #f)
          ((equal? elem (car lst)) #t)
          (else (contains? elem (cdr lst)))))
  
  (define (unique-helper lst acc)
    (if (null? lst)
        acc
        (if (contains? (car lst) acc)
            (unique-helper (cdr lst) acc)
            (unique-helper (cdr lst) (cons (car lst) acc)))))
  
  (unique-helper lst '()))


(define (append-png-wildcard path)
  (if (not (char=? (string-ref path (- (string-length path) 1)) #\/))
      (string-append path "/*.png")  ; Append "/ *.png" if no trailing slash
      (string-append path "*.png"))
)  ; Append "*.png" directly if trailing slash exists

(define (fix-path path)
  (if (not (char=? (string-ref path (- (string-length path) 1)) #\/))
      (string-append path "/")  ; Append "/" if no trailing slash
      path)
)  

(define (construct-filename y z e path)
  ;; Construct the filename based on y and z values
  (string-append path "Focused_y=" (number->string y) "_z=" (number->string z) "_e=" (number->string e) ".png")
)


(define (load-grid path)
    ; we define file_name as the return of list directory with encoding UTF-8 cadr removes first element which is a number
    (display "At glob")(newline)
    (let* ((file-results (file-glob (append-png-wildcard path) 0))
       (file-names (cadr file-results))  ; This retrieves the list of file names
       (file-count (car file-results))) ; This retrieves the count of files
    (if (> file-count 0)
        (set! file-names file-names)  ; This line is actually redundant in this context
        (error "No files found at path"))
    (let* ((results (extract-grid file-names))
       (y-list (car results))
       (z-list (cadr results))
       (e-list (caddr results)))
       ; Sort the lists
       (set! y-list (qsort y-list))
       (set! z-list (qsort z-list))
       (set! e-list (qsort e-list))
       ; And return 
       (list y-list z-list e-list)))
)

(define (extract-grid file-names)
        ; ' tells lisp to not evaluate this bracket
    (let ((y-list '())
          (z-list '())
          (e-list '()))
          ; for each apply the lambda function
     (for-each (lambda (filename)
                ; we let the output be values
                 (let ((values (parse-filename filename)))
                    ; car -> give first item cadr -> (car (cadr X)) return first of remainder of first etc 
                    (set! y-list (cons (car values) y-list))
                    ; cons appends lists together 
                    (set! z-list (cons (cadr values) z-list))
                    ; set! because objects are immutable and we are concatenating
                    (set! e-list (cons (caddr values) e-list))))
            ; this is iterated
            file-names)
        ; Reverse because cons reverted by appending in front 
    (list (reverse y-list) (reverse z-list) (reverse e-list))
    )
)

(define (parse-filename file-name)
    ; Create parts list where each filename is split by _ (fnames are Focused_y=0_z=1_e=2.png)
    (let* ((parts (string-split underscore? file-name)))
        ; We now proceed to grab the relevant element with list-ref split along the equals
        ; remove the first element and then convert to numbers, for e we also split along fullstop
        (define y-values (string->number (cadr (string-split equalsign? (list-ref parts 1)))))
        (define z-values (string->number (cadr (string-split equalsign? (list-ref parts 2)))))
        (define e-part (cadr (string-split equalsign? (list-ref parts 3))))
        (define e-values (string->number (car (string-split fullstop? e-part))))

    (list y-values z-values e-values)
    )
)


(define (fullstop? char) 
    ; char=? says compare chars, char is the variable and #\ defines the char: =
    (char=? char #\.)
)
(define (equalsign? char) 
    ; char=? says compare chars, char is the variable and #\ defines the char: =
    (char=? char #\=)
)
(define (underscore? char) 
    ; char=? says compare chars, char is the variable and #\ defines the char: =
    (char=? char #\_)
)

; Not part of tinyscheme - stolen from scheme cookbook - note char-delimiter is a function returning true when the delimiter is found
(define (string-split char-delimiter? string_)
  (define (maybe-add a b parts)
    (if (= a b) parts (cons (substring string_ a b) parts)))
  (let ((n (string-length string_)))
    (let loop ((a 0) (b 0) (parts '()))
      (if (< b n)
          (if (not (char-delimiter? (string-ref string_ b)))
              (loop a (+ b 1) parts)
              (loop (+ b 1) (+ b 1) (maybe-add a b parts)))
          (reverse (maybe-add a b parts))))))

; Not part of tinyscheme - quicksort, taken from stack overflow
(define (qsort e)
  (if (or (null? e) (<= (length e) 1)) e
      (let loop ((left '()) (right '())
                   (pivot (car e)) (rest (cdr e)))
            (if (null? rest)
                (append (append (qsort left) (list pivot)) (qsort right))
               (if (<= (car rest) pivot)
                    (loop (append left (list (car rest))) right pivot (cdr rest))
                    (loop left (append right (list (car rest))) pivot (cdr rest)))))))

; Because why would there be a function for the maximum of a list 
(define (max-list lst)
  (if (null? (cdr lst))  ; If there's only one element left
      (car lst)          ; Return that element
      (max (car lst) (max-list (cdr lst))))
)  ; Otherwise, return the max of the first element and the max of the rest


; register the script, what sort of type is this?
 (script-fu-register
    "script-fu-microscope-load-grid"                        ;function name
    "Load Microscope Grid"                                  ;menu label
    "Loads images from path and places them based on name"  ;description
    "Felix Semler"                             ;author
    "Felix Semler"        ;copyright notice
    "2024"                          ;date created
    ""                                      ;image type that the script works on
    SF-STRING      "path"          "/home/felix/RapidStorage2/GarlicShell/"   ;a string variable TODO
  )
  (script-fu-menu-register "script-fu-microscope-load-grid" "<Image>/Microscope/")