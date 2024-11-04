# Importing all required packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read in the cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
 
# create a function to detect face
def adjusted_detect_face(img):
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor = 1.05, minNeighbors = 5)
         
    return face_img
 
 
# create a function to detect eyes
def detect_eyes(img):
     
    eye_img = img.copy()   
    eye_rect = eye_cascade.detectMultiScale(eye_img, scaleFactor = 1.2, minNeighbors = 5)   
    for (x, y, w, h) in eye_rect:
        cv2.rectangle(eye_img, (x, y), (x + w, y + h), (255, 255, 255), 10)       
    
    return eye_img
 
# Reading in the image and creating copies
img = cv2.imread('twitter_profiles_1675655693_724032926757711873.jpg')
img_copy1 = img.copy()
img_copy2 = img.copy()
img_copy3 = img.copy()
 
# Detecting the face
face = adjusted_detect_face(img_copy1)
plt.imshow(face)
cv2.imwrite('face1.jpg', face)

eyes = detect_eyes(img_copy2)
plt.imshow(eyes)
cv2.imwrite('face_eyes1.jpg', eyes)

eyes_face = adjusted_detect_face(img_copy3)
eyes_face = detect_eyes(eyes_face)
plt.imshow(eyes_face)
cv2.imwrite('face+eyes1.jpg', eyes_face)

