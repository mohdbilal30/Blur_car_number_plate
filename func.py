import matplotlib.pyplot as plt
import cv2
# Pretrained machine learning files(xml files)
plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

def display(img):
    # display image in larger size (10,8)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    # Converting color channels of the image as matplotlib takes image as RGB
    # And openCv takes image in BGR channel
    new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)
    # display image
    plt.show()

def detect_plate(img):

    plate_img = img.copy()

    #roi = img.copy()

    plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.2,minNeighbors=5)

    for (x,y,w,h) in plate_rects:
        #roi = roi[y:y+h,x:x+w]

        #blurred_roi = cv2.medianBlur(roi,9)

        plate_img[y:y+h,x:x+w] = cv2.medianBlur(plate_img[y:y+h,x:x+w],39)

        cv2.rectangle(plate_img,(x,y),(x+w,y+h),(0,0,255),3)

    return plate_img

if __name__ == '__main__':
    print("Completed")


