
# coding: utf-8

# In[12]:


import cv2 
import numpy as np
import matplotlib.pyplot as plt


test_image = cv2.imread('C://Users//PTA3HO//Downloads//obama.jpeg')
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
plt.imshow(test_image_gray, cmap='gray')

haar_cascade_face = cv2.CascadeClassifier('C://Users//PTA3HO//haarcascades//haarcascade_frontalface_default.xml')
faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);

# Let us print the no. of faces found
print('Faces found: ', len(faces_rects))
for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))


