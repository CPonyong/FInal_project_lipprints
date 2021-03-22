#!/usr/bin/env python
# coding: utf-8

# In[10]:
import pymysql
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tqdm import tqdm
from keras.preprocessing import image
import cv2
import numpy as np
import os
import io
from PIL import Image, ImageOps

mydb = pymysql.connect(
  host="localhost",
  user="root",
  password="root",
  database="lab_test"
)

mycursor = mydb.cursor()
mycursor.execute("SELECT image,image_name FROM preprocess_image where img_description ='p' ")

myresult = mycursor.fetchall()

datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='constant')

#fill mode constant", "nearest", "reflect" or "wrap"

def upload(UL_PATH):
    pID = 1
    fig_name = []
    i = 0
    for file in sorted(os.listdir(UL_PATH)):
        file_name = file.split("_")[1].split("(")[0]
        fig_name.append(file_name)
    fig_name = list(dict.fromkeys(fig_name))

    name = fig_name[i]
    for file in sorted(os.listdir(UL_PATH)):
        img = open(UL_PATH+file,'rb').read()
        name_Split = file.split("_")[1].split("(")[0]

        if name_Split == name:
            pID = pID
        else:
            name = fig_name[i+1]
            pID += 1
            i += 1

        sql = "INSERT INTO preprocess_image (image, image_name, img_description,pID) VALUES (%s, %s, %s, %s)"
        val = (img, file, "ag", pID)
        mycursor.execute(sql, val)
        mydb.commit()

#เช็คไฟล์ทั้งหมดในโฟลเดอร์นั้น
def prepare_input(LP_PATH):

    for i in range(len(myresult)):
        if i < 10:
            img_db = io.BytesIO(myresult[i][0])
            samplelp = myresult[i][1]
            sample = samplelp.split("C")[0].split("(")[1].split(")")[0]
            participant = samplelp.split("(")[0]

            img = Image.open(img_db)

            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            j = 0

        # เพิ่มรูปในแต่ละรอบจำนวนกี่รูป
            for batch in datagen.flow(x, batch_size=1,
                                        save_to_dir='../03_lipprint_ag/03_20pp3r_5004/',
                                        save_prefix='ls_'+str(participant)+'('+str(sample)+')',
                                        save_format='tiff'):


                j += 1

                #กำหนดรูปที่ต้องการเพิ่ม - 1
                if j > 49:
                    break
        else:
            break


LP_PATH = myresult
UL_PATH = "../03_lipprint_ag/03_20pp3r_2000/"
prepare_input(LP_PATH)
upload(UL_PATH)
mycursor.close()
mydb.close()
# In[ ]:




