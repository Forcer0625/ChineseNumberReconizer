# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:33:23 2020

@author: user
"""

#建立模型
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()

model.add(Dense(units=256,
                input_dim=28*28, 
                kernel_initializer="normal",
                activation="relu"))

model.add(Dense(units=10, 
                kernel_initializer="normal",
                activation="softmax" ))

from keras.layers import DropOut
model.add(DropOut(0.5))



model.save("MLP_Chinese_Number_Reconizer.h5")#儲存模型
print("模型儲存完畢")
print("\n輸入層784\t隱藏層256\t輸出層10 \t無Dropout")
del model #刪掉model




