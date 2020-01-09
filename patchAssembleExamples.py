"""
<Ostraca pairwise matching examples to use with trained model>
    Copyright (C) <2019>  <Cecilia Ostertag>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU, Cropping2D, Input, Concatenate, Lambda, Flatten, Dense, Dropout
from keras import Sequential, Model
from keras import backend as K
import os
import keras
import gc
import cv2
from keras.utils.training_utils import multi_gpu_model
import random

net = keras.models.load_model('patchAssemble_ostraca.h5',custom_objects={"tf": tf})
assemb_net = keras.models.load_model('Assemblies.h5',custom_objects={"tf": tf})
net.summary()

patchFile = np.load("validData.npz")
p1 = patchFile['arr_0'][:40]
p2 = patchFile['arr_1'][:40]
l = patchFile['arr_2'][:40]
print("Truth")
#print(l)
preds = net.predict_on_batch([p1,p2])
print("Preds")
#print(np.argmax(preds))
for i in range(p1.shape[0]):
	T = l[i]
	P = np.argmax(preds[i])
	pr = np.amax(preds[i])
	pr = str(pr)[:4]
	P1 = p1[i,:,:,:3]
	P2 = p2[i,:,:,:3]
	P1 = P1.copy()
	P2 = P2.copy()
	#cv2.imwrite("./recons/p1_"+str(i)+"_T"+str(T)+"_P"+str(P)+".jpg",p1[i,:,:,:]*255.)
	#cv2.imwrite("./recons/p2_"+str(i)+"_T"+str(T)+"_P"+str(P)+".jpg",p2[i,:,:,:]*255.)
	print(P1.shape)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(P1,"orig",(50,50), font, 2,(0,0,255),2,cv2.LINE_AA)
	cv2.putText(P2,str(pr),(50,50), font, 2,(0,0,255),2,cv2.LINE_AA)
	if P == 1:
		rec = np.concatenate((P1,P2),axis=1)
	elif P == 2:
		rec = np.concatenate((P2,P1),axis=1)
	elif P == 3:
		rec = np.concatenate((P2,P1),axis=0)
	elif P == 4:
		rec = np.concatenate((P1,P2),axis=0)
	else:
		rec = np.concatenate((P1,P2),axis=1)
	cv2.line(rec,(200,0),(200,200),(0,255,0),1)
	print(rec.shape)
	cv2.imwrite("./recons/recon_"+str(i)+"_T"+str(T)+"_P"+str(P)+".jpg",rec*255.)

