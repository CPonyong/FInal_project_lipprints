#!/usr/bin/env python
# coding: utf-8

# In[13]:


import joblib
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
import pymysql
from PIL import Image
import io

mydb = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    database="lab_test"
)

mycursor = mydb.cursor()
#
mycursor.execute("SELECT images, image_descriptions FROM unknown_image ORDER BY id DESC LIMIT 1")
myresult = mycursor.fetchall()
# print(myresult[0])
class Preprocess:
    def __init__(self, file, FIG_NAME):
        self.img = Image.open(file)
        self.img = np.asarray(self.img)
        self.original_img = self.img
        self.figname = FIG_NAME

    def toGray(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.original_img = self.img

    def resize(self, x, y):
        self.img = cv2.resize(self.img, (x, y))

    def crop(self, x, y, w, h):
        self.img = self.img[y:h, x:h]

    def gaussianBlur(self, times, window=(5, 5)):
        img = self.img
        for i in range(times):
            img = cv2.GaussianBlur(img, window, 0)
            self.img = img

    def sobel(self, axis):
        if axis == 'x':
            sobel_img = cv2.Sobel(self.img, cv2.CV_8U, 1, 0, ksize=5)
        elif axis == 'y':
            sobel_img = cv2.Sobel(self.img, cv2.CV_8U, 0, 1, ksize=5)
        elif axis == 'xy':
            sobelx = cv2.Sobel(self.img, cv2.CV_8U, 1, 0, ksize=5)
            sobely = cv2.Sobel(self.img, cv2.CV_8U, 0, 1, ksize=5)
            abs_grad_x = cv2.convertScaleAbs(sobelx)
            abs_grad_y = cv2.convertScaleAbs(sobely)
            sobel_img = cv2.addWeighted(abs_grad_x, 0.1, abs_grad_y, 0.1, 0)
        self.img = sobel_img
        # print type(sobel_img), type(self.img)

    def threshold(self):
        th, self.img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def get(self):
        return self.original_img, self.img


def prepare_inputs(FIG_DIR):
    ukn_image = []
    for i in range(len(myresult)):
        file_like2 = io.BytesIO(myresult[i][0])
        FIG_NAME = myresult[i][1]

        # print(FIG_NAME)

        # file_like2 = Image.open(file_like2)
        # file_like2.show()

        p = Preprocess(file_like2, FIG_NAME)
        p.toGray()
        p.crop(600, 700, 3200, 3500)
        p.resize(600, 400)
        p.gaussianBlur(4)
        p.sobel('xy')
        p.threshold()
        orginal_img, processed_img = p.get()
        ukn_image.append(processed_img)

    return ukn_image


FIG_DIR = myresult
SAVEMD_PATH = "../05_lipprint_model/02_savemodel/02_model20pp3r_2000.pkl"
#
ukn_image = prepare_inputs(FIG_DIR)
loaded_model = joblib.load(SAVEMD_PATH)

x_ukn = np.asarray(ukn_image)
x_ukn = x_ukn.reshape((x_ukn.shape[0], x_ukn.shape[1] * x_ukn.shape[2]))

y_predict = loaded_model.predict(x_ukn)
y_matching = loaded_model.predict_proba(x_ukn)

count_ypred = 0
count_pic = 1

for i in y_matching:
    print("Lipprint", count_pic, ": The result of the prediction is",
          y_predict[count_ypred], "with confidence of %.2f %%" % (max(i) * 100),
          "loss_percentage = %.2f %%" % (100 - (max(i) * 100)))
    print(y_matching)
    count_ypred += 1
    count_pic += 1

    cv2.destroyAllWindows()



