#!/usr/bin/env python
# coding: utf-8


import pymysql
import os
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
import sklearn
import cv2
import os
import io
import warnings
import pickle
import joblib
import seaborn as sns; sns.set()

from sklearn.decomposition import PCA

mydb = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    database="lab_test"
)

mycursor = mydb.cursor()




def prepare_inputs(FIG_DIR, OUTPUT_DIR):
    pID = 1
    fig_name = []
    i = 0
    for file in sorted(os.listdir(FIG_DIR)):
        file_name = file.split("(")[0]
        fig_name.append(file_name)

    fig_name = list(dict.fromkeys(fig_name))
    name = fig_name[i]

    for file in sorted(os.listdir(FIG_DIR)):
        name_Split = file.split("(")[0]
        if name_Split == name:
            pID = pID
        else:
            name = fig_name[i+1]
            pID += 1
            i += 1


    # # insert into DB
        source = open(OUTPUT_DIR + file, 'rb').read()

        # sql = "INSERT INTO source_image (image, image_name, pID) VALUES (%s, %s,%s)"
        # val = (source, file, pID)
        # mycursor.execute(sql, val)
        # mydb.commit()



FIG_DIR = "../01_lipprint_original/03_20pp_3r_org/"

OUTPUT_DIR = "../01_lipprint_original/03_20pp_3r_org/"


prepare_inputs(FIG_DIR, OUTPUT_DIR)
mycursor.close()
mydb.close()
# cnx.close()