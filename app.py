import imutils
import cv2 as cv
import numpy as np

vid = cv.VideoCapture(0)

fgbg = cv.createBackgroundSubtractorMOG2()
while True:
    ret, frame = vid.read()
    print(ret)
    frame = imutils.resize(frame, width = 750)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (21, 21), 0)
    fgmask=fgbg.apply(blur)
    thresh = cv.threshold(fgmask, 50, 255, cv.THRESH_BINARY)[1]
    dil = cv.dilate(thresh, None, iterations = 1)

    cnts, _ = cv.findContours(dil.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        if cv.contourArea(c) > 500:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(frame, str("Movement"), (10,35), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2, cv.LINE_AA)

    cv.imshow("Masked", dil)
    cv.imshow("Original", frame)
    ch = cv.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

vid.release();
cv.destroyAllWindows()
