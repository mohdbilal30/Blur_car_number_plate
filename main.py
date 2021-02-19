import cv2
import matplotlib.pyplot as plt
import  func
import numpy as np
plate = cv2.imread("car_plate.jpg")
plate2 = cv2.imread("plate_2.jpg")
plate3 = cv2.imread("plate_3.jpg")
plate4 = cv2.imread("plate_4.jpg")
plate_rgb = cv2.cvtColor(plate,cv2.COLOR_BGR2RGB)
plate_img_resize = cv2.resize(plate,dsize=(900,500))
plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

##################display function to resize image and color correct image#####################

# def detect_plate(img):
#
#     plate_img = img.copy()
#
#     roi = img.copy()
#
#     plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3,minNeighbors=3)
#
#     for (x,y,w,h) in plate_rects:
#         roi = roi[y:y+h,x:x+w]
#
#         blurred_roi = cv2.medianBlur(roi,7)
#
#         plate_img[y:y+h,x:x+w] = blurred_roi
#
#         cv2.rectangle(plate_img,(x,y),(x+w,y+h),(0,0,255),3)
#
#     return plate_img

result = func.detect_plate(plate_img_resize)
# func.display(result)

cap = cv2.VideoCapture("plate_video.mp4")
#---------opening image using openCV-----------#
while True:

    ret,frame = cap.read()
    # Checking if the camera is found
    if ret == True:
        frame = func.detect_plate(frame)

        cv2.imshow("image",frame)

        k=cv2.waitKey(1)
    # Getting keyboard input to exit from the programm
    # 27 number is for Esc button in keyboard
        if k == 27:
            break
    # if there is no camera or video source break out of the while true loop
    else:

        break
cap.release()
cv2.destroyAllWindows()

