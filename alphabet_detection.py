import cv2
import numpy as np
import pandas as pd
import seaborn as sns #graph [heatmap etc]
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score
import os, ssl, time
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L','M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

print(nclasses)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 7500, test_size = 2500, random_state = 9)

X_train = X_train/255.0
X_test = X_test/255.0

model = lr(solver = 'saga', multi_class= 'multinomial')

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

accuracy = accuracy_score(y_predict, y_test)

print(accuracy)

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        upper_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right = (int(width/2 + 56), int(height/2 + 56))

        cv2.rectangle(gray, upper_left, bottom_right, (255, 0, 0), 2)

        ROI = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        image_PIL = Image.fromarray(ROI)

        image_bw = image_PIL.convert('L') #cvt to grayscale

        image_bw_resized = image_bw.resize((28,28))

        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)

        pixel_filter = 20
        
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
        
        max_pixel = np.max(image_bw_resized_inverted)
        
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        
        test_pred = model .predict(test_sample)
        
        print("Predicted class is: ", test_pred)

        # Display the resulting frame
        cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()