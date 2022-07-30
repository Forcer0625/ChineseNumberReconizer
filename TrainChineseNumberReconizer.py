# -*- coding: utf-8 -*-
"""
Created on Sat May 16 22:09:49 2020

@author: user
"""

#---載入模型---
from keras.models import load_model
model=load_model("MLP_Chinese_Number_Reconizer.h5")
#-------------

#--定義訓練方法--
model.compile(loss="categorical_crossentropy", #設定損失函式為categorical_crossentropy
              optimizer="adam", #設定優化方法為adam
              metrics=["accuracy"]) #設定評估模型的方式為accuracy準確率
#---------------

#---載入資料---
import glob,cv2
import numpy as np
from keras.utils import np_utils

train_feature=[]
train_label=[]
data_num=6
for i in range(data_num):
    now=str(i+1)
    data_path="ChineseTrainData0"
    data_path+=now
    data_path+="\*.jpg"
    print("\nLoading...",data_path)
    files = glob.glob(data_path) #改這裡
    for file in files:
        img=cv2.imread(file)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰階
        _,img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)#轉為反向黑白
        train_feature.append(img)
        label=file[19:20] #ChineseTrainData\1.jpg:將第17個字元1為label
        train_label.append(int(label)) #轉成int
        

train_feature=np.array(train_feature)
train_label=np.array(train_label) #list轉為矩陣
    
train_feature_vector=train_feature.reshape(len(train_feature),28*28).astype("float32")
train_feature_normalize=train_feature_vector/255
train_label_onehot=np_utils.to_categorical(train_label)
#------------

#-----訓練------
trained_model=model.fit(x=train_feature_normalize,
                        y=train_label_onehot,
                        validation_split=0.1,epochs=50,
                        batch_size=(len(train_feature)))
#--------------
#----查看訓練狀況----
import matplotlib.pyplot as plot
def show_train_history(train_history, title, train, validation):
    plot.plot(train_history.history[train])
    plot.plot(train_history.history[validation])
    plot.title(title)
    plot.ylabel(train)
    plot.xlabel('Epoch')
    plot.legend(['train', 'validation'], loc = 'upper left')
    plot.show()

show_train_history(trained_model, 'Accuracy', 'accuracy', 'val_accuracy')
show_train_history(trained_model, 'Loss', 'loss', 'val_loss')
#------------------

model.save("MLP_Chinese_Number_Reconizer.h5")#儲存檔名為Mnist_MLP_model01.h5的模型
print("模型儲存完畢")
del model #刪掉model

#predictions=model.predict_classes(train_feature_normalize)
#start=50
#n=5
#show_image_labels_predictions(train_feature, train_label, predictions, start, start+n)

