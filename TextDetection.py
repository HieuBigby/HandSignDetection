import cv2
import pytesseract

# pytesseract.tesseract_cmd = r'/usr/bin/tesseract/'
pytesseract.pytesseract.tesseract_cmd = '/home/hieubigby/Applications/tesseract-5.3.0-x86_64.AppImage'
img = cv2.imread('Data/Other/test3.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(pytesseract.image_to_string(img))
cv2.imshow('Result', img)
cv2.waitKey(0)