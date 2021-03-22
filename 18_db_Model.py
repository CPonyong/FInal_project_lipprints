import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import cv2
import os
import warnings
import seaborn as sns; sns.set()
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

class LipprintImage:
    def __init__(self,LP_PATH):
        self.train_image = []
        self.label = []
        for LP_NAME in os.listdir(LP_PATH):
            img = cv2.imread(LP_PATH+LP_NAME,0)
            self.train_image.append(img)
            
            label = LP_NAME.split("_")[1].split("(")[0]
            self.label.append(label)
            #print(LP_NAME,label)
    def image_array(self):
        x = np.asarray(self.train_image)
        return x
    def label_list(self):
        y = self.label
        y_label = []
        y_label = list(set(y))
        y_label.sort()
        return y, y_label

class TrainingSet:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def splitdata(self,test_size,shuffle,random_state):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, 
                                                            test_size=test_size,
                                                            shuffle = shuffle,
                                                            random_state = random_state)
        return x_train, x_test, y_train, y_test
    def img_reshape(self,x_reshape):
        x_reshape = x_reshape.reshape((x_reshape.shape[0], x_reshape.shape[1] * x_reshape.shape[2]))
        return x_reshape
        
class Model:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    def setParameter(self,C,gamma,kernel):
        parameters_grid = {'C': C,  
              'gamma': gamma, 
              'kernel': [kernel]}  
        self.grid = GridSearchCV(SVC(probability=True), param_grid=parameters_grid, refit=True, verbose=10, cv=5)
        return self.grid
    def fitModel(self):
        self.grid.fit(self.x_train, self.y_train)
        print(self.grid.best_params_)
    def predictModel(self, x_test, y_test):
        self.svc_model = self.grid.best_estimator_
        y_pred = self.svc_model.predict(x_test)
        print ("score = %3.2f" %(self.svc_model.score(x_test,y_test)))
        return y_pred 
    def saveModel(self, SAVEMD_PATH):
        joblib_file = SAVEMD_PATH  
        joblib.dump(self.svc_model, joblib_file)

class Report:
    def cfmt(self, y_test, y_pred, y_label):
        cmtx = pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            index=[y_label],
            columns=[y_label]
        )
        print(cmtx)
    def class_report(self, y_test, y_pred):
        print(classification_report(y_test, y_pred))
    def accuracy(self, y_test, y_pred):
        print("Accuracy = ", accuracy_score(y_test, y_pred) * 100)
    def cfmt_hm(self, SAVECM_PATH, grid, y_test, y_pred, y_label):
        plt.figure()
        mat = confusion_matrix(y_test, y_pred)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels= y_label,
                    yticklabels= y_label)

        plt.title('Best hyperparameters kernel = %s, C = %s, gamma = %s' 
                  %(grid.best_params_['kernel'], grid.best_params_['C'], grid.best_params_['gamma']))
        plt.xlabel('True label')
        plt.ylabel('Predicted label')
        plt.savefig(SAVECM_PATH)
        plt.show()
    def typeVariable(self,variable):
        print(type(variable))

########## training set ########        
LP_PATH = "database/03_lipprint_ag/test/"
SAVECM_PATH = "database/05_lipprint_model/02_savecm/01_5pp3r_750_0103.pdf"
SAVEMD_PATH ="database/05_lipprint_model/01_savemodel/01_model5pp3r_750_0103.pkl"

lp_img = LipprintImage(LP_PATH)
x = lp_img.image_array()
y , y_label = lp_img.label_list()

train_set = TrainingSet(x,y)
x_train, x_test, y_train, y_test = train_set.splitdata(test_size = 0.3,
                                             shuffle = True,random_state = 1)
x_train = train_set.img_reshape(x_train)
x_test = train_set.img_reshape(x_test)

model = Model(x_train, y_train)
grid = model.setParameter([1],[0.0001],'linear')
model.fitModel()
y_pred = model.predictModel(x_test, y_test)
#model.saveModel(SAVEMD_PATH)

report = Report()
#report.cfmt(y_test, y_pred, y_label)  
#report.class_report(y_test, y_pred)
#report.accuracy(y_test, y_pred)
#report.cfmt_hm(SAVECM_PATH, grid, y_test, y_pred, y_label)

print("LP_PATH")
report.typeVariable(LP_PATH)
print("\n")

print("SAVECM_PATH")
report.typeVariable(SAVECM_PATH)
print("\n")

print("SAVEMD_PATH")
report.typeVariable(SAVEMD_PATH)
print("\n")

print("x")
report.typeVariable(x)
print("\n")

print("y")
report.typeVariable(y)
print("\n")

print("x_train")
report.typeVariable(x_train)
print("\n")

print("y_train")
report.typeVariable(y_train)
print("\n")






