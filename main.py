from imutils.video import VideoStream
import numpy as np
import time
import cv2
import os

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vs = VideoStream(src=0).start()
time.sleep(2)
COUNTER = 0
CONSEC_FRAMES = 5
DELAY_COUNTER = 0
DELAY = 30
FACEYARR = []
Y_TOLERANCE = 5
FACE_Y_CACHE = 600

# loop indefinitely
while True:
    # grab the frame from the threaded video stream and flip it
    # vertically (since our camera was upside down)
    frame = vs.read()

    # calculate the center of the frame as this is where we will
    # try to keep the object
    (H, W) = frame.shape[:2]
    centerX = W // 2
    centerY = H // 2

    # find the object's location
    # faceLoc = obj.update(frame, (centerX, centerY))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.05,
                                      minNeighbors=9, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    # check to see if a face was found
    if len(rects) > 0:
        # extract the bounding box coordinates of the face and
        # use the coordinates to determine the center of the
        # face
        (x, y, w, h) = rects[0]
        faceX = int(x + (w / 2.0))
        faceY = int(y + (h / 2.0))

        # return the center (x, y)-coordinates of the face
        faceLoc = ((faceX, faceY), rects[0])
    else:
        continue

    # extract the bounding box and draw it
    if rects[0] is not None:
        (x, y, w, h) = rects[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    FACEYARR.append(faceY)
    if len(FACEYARR) > FACE_Y_CACHE:
        del FACEYARR[0]
    MEAN_Y = sum(FACEYARR) / len(FACEYARR)
    print("CURRENT:" + str(faceY))
    print("MEAN-->" + str(MEAN_Y))
    if faceY > (MEAN_Y + Y_TOLERANCE):
        COUNTER += 1
        if COUNTER >= CONSEC_FRAMES:
            print("FIX YOUR POSTURE!")
            cv2.putText(frame, "POSTURE!!", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    else:
        COUNTER = 0

    # display the frame to the screen
    cv2.imshow("OUTPUT", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
