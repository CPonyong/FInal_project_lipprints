#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import mysql.connector
import pymysql
import io
from PIL import Image, ImageOps
from mysql.connector import errorcode
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA

mydb = pymysql.connect(
  host="localhost",
  user="root",
  password="root",
  database="lab_test"
)

mycursor = mydb.cursor()
mycursor.execute("SELECT image,image_name FROM source_image")

myresult = mycursor.fetchall()

class Preprocess:
    def __init__(self, file, FIG_NAME, pID):
        self.img = Image.open(file)
        self.img = np.asarray(self.img)
        self.original_img = self.img
        self.figname = FIG_NAME
        self.pID = pID

    def toGray(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # self.img = ImageOps.grayscale(self.img)
        self.original_img = self.img

    
    def resize(self, x, y):
        self.img = cv2.resize(self.img,(x, y))
        # self.img = self.img.resize((x,y))

    def crop(self, x, y, w, h):
        self.img = self.img[y:h, x:w]
        # self.img = self.img.crop((x,w,y,h))

    def gaussianBlur(self, times, window = (5,5)):
        img = self.img
        for i in range(times):
            img = cv2.GaussianBlur(img, window,0)
        self.img = img

    def sobel(self, axis):
        if axis == 'x':
            sobel_img = cv2.Sobel(self.img, cv2.CV_8U, 1, 0, ksize=5)
        elif axis == 'y':
            sobel_img = cv2.Sobel(self.img, cv2.CV_8U, 0, 1, ksize=5)
        elif axis == 'xy':
            sobelx= cv2.Sobel(self.img, cv2.CV_8U, 1, 0, ksize=5)
            sobely = cv2.Sobel(self.img, cv2.CV_8U, 0, 1, ksize=5)
            abs_grad_x = cv2.convertScaleAbs(sobelx)
            abs_grad_y = cv2.convertScaleAbs(sobely)
            sobel_img = cv2.addWeighted(abs_grad_x, 0.1, abs_grad_y, 0.1, 0)
        self.img = sobel_img

        # print type(sobel_img), type(self.img)

    def threshold(self):
        th, self.img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

    def savefig(self, OUTPUT_DIR):
        # cv2.imwrite(OUTPUT_DIR+self.figname, self.img)
        preproc_image = open(OUTPUT_DIR+"/"+self.figname, 'rb').read()
        sql = "INSERT INTO  preprocess_image (image, image_name, img_description, pID) VALUES (%s, %s, %s, %s)"
        val = (preproc_image, self.figname, "p", self.pID)
        mycursor.execute(sql, val)
        mydb.commit()


def prepare_inputs(OUTPUT_DIR):
    pID = 1
    fig_name = []
    j = 0
    for file in sorted(os.listdir(FIG_DIR)):
        file_name = file.split("(")[0]
        fig_name.append(file_name)

    fig_name = list(dict.fromkeys(fig_name))
    name = fig_name[j]
    for i in range(len(myresult)):
        file_like2 = io.BytesIO(myresult[i][0])
        FIG_NAME = myresult[i][1]

        name_Split = FIG_NAME.split("(")[0]
        if name_Split == name:
            pID = pID
        else:
            name = fig_name[j+1]
            pID += 1
            j += 1



        p = Preprocess(file_like2,FIG_NAME,pID)
        p.toGray()
        p.crop(600, 700, 3500, 3000)
        p.resize(600, 400)
        p.gaussianBlur(4)
        p.sobel('xy')
        p.threshold()
        # p.savefig(OUTPUT_DIR)
    
FIG_DIR = "../01_lipprint_original/03_20pp_3r_org/"
OUTPUT_DIR = "../02_lipprint_prepro/03_20pp_3r_org/"


prepare_inputs(OUTPUT_DIR)
mycursor.close()
mydb.close()
