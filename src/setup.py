def eyeDetection():
    import numpy as np
    import cv2 as cv
    eye_casc=cv.CascadeClassifier('haarcascade_eye.xml')
    face_casc=cv.CascadeClassifier('haarcascade_frontalface_defaults.xml')
    color=(0,255,0)
    thickness=2

    cam = cv.VideoCapture(0)
    while True:
        ret,frame = cam.read()
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        faces=eye_casc.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

        img=frame
        for(x,y,w,h) in faces:
            eye_range=frame[y:y+h,x:x+w]
            roi_color=gray[y:y+h,x:x+w]
            eyes=eye_casc.detectMultiScale(roi_color)
            for(x_eye,y_eye,w_eye,h_eye) in eyes:
             img=cv.rectangle(eye_range, (x_eye,y_eye), (x_eye + w_eye,y_eye + h_eye),color, thickness)
        cv.imshow('Camera',frame)

        if cv.waitKey(1) == ord('q'):
            break
    cam.release()
    cv.destroyAllWindows()