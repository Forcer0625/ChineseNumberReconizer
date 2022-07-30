# -*- coding: utf-8 -*-
"""
Created on Sat May 16 22:19:33 2020

@author: user
"""


from keras.models import load_model
model=load_model("MLP_Chinese_Number_Reconizer.h5")

#------------顯示圖片----------------------
import matplotlib.pyplot as plot
def show_image(image): #以黑白灰階顯示2*2大小的圖片
        fig=plot.gcf()
        fig.set_size_inches(2, 2)# 數字圖片大小
        plot.imshow(image, cmap="binary") # 'binary':黑白灰階顯示 0~1.0
        plot.show()

def show_image_labels_predictions(image, labels, predictions, start_id, num=10):
    #image是數字圖片label是真值,prdictions是預測值
    #start_id是開始顯示的索引編號 num是顯示的圖片個數 最剁可縣示num_max張
    plot.gcf().set_size_inches(60, 75)
    num_max=50
    if num>num_max: num=num_max
    for i in range(num):
        ax=plot.subplot(10, 10, i+1)# 顯示黑白圖片
        ax.imshow(image[start_id], cmap="binary")
        if( len(predictions)>0  ): # 有AI預測結果資料, 才在標題顯示結果
            title = "AI = "+str(predictions[start_id])
            title +=("(O)" if predictions[start_id]==labels[start_id] else "(X)") #正確顯示O 錯誤則X
            title +="\nLabel = "+str(labels[start_id])
        else:                      #沒有AI預測結果資料, 只在標題顯示真值
            title = "Label = "+str(labels[start_id])
        # X, Y軸不顯示刻度
        ax.set_title(title,fontsize=60)
        ax.set_xticks([])
        ax.set_yticks([])
        start_id+=1
    plot.show()        

#-----------------------------------------

import glob,cv2

files = glob.glob("testChineseData\*.jpg") #改這裡
test_feature=[]
test_label=[]
for file in files:
    img=cv2.imread(file)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰階
    _,img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)#轉為反向黑白
    test_feature.append(img)
    label=file[16:17] #ChineseTrainData\1.jpg:將第17個字元1為label #還有這裡
    test_label.append(int(label)) #轉成int

import numpy as np
test_feature=np.array(test_feature)
test_label=np.array(test_label) #list轉為矩陣

test_feature_vector=test_feature.reshape(len(test_feature),28*28).astype("float32")
test_feature_normalize=test_feature_vector/255

predictions=model.predict_classes(test_feature_normalize)
start=0
n=9
show_image_labels_predictions(test_feature, test_label, predictions, start, n)

from keras.utils import np_utils
test_label_onehot=np_utils.to_categorical(test_label)
score=model.evaluate(test_feature_normalize, test_label_onehot)
print("\n準確率 = ",score[1])