"""
<Ostraca pairwise matching NN model architecture, data loading, and training>
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

os.environ["CUDA_VISIBLE_DEVICES"]="1,2" #select available gpus on the cluster

### TF Records utility functions
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
def _read_from_tfrecord(example_proto):
    feature = {
        'train/p1': tf.FixedLenFeature([], tf.string),
        'train/p2': tf.FixedLenFeature([], tf.string),
        'train/label': tf.FixedLenFeature([], tf.int64)
    }

    features = tf.parse_example([example_proto], features=feature)
    # one feature = one pair of patches (images stored as bytes string) + corresponding label
    # patches have 5 channels : R, G, B, luminance, and transparency
    # pixel values are normalized between 0 and 1

    label_1d = features['train/label']
    image_1d = tf.decode_raw(features['train/p1'], tf.float32)
    image2_1d = tf.decode_raw(features['train/p2'], tf.float32)
    label_restored = label_1d #label
    image_restored = tf.reshape(image_1d, [200,200,5],name="r1") # patch1
    image2_restored = tf.reshape(image2_1d, [200,200,5],name="r2") #patch2
    return label_restored, image_restored, image2_restored


class DataGenerator(keras.utils.Sequence):
	def __init__(self, dataset, next, sess, nbsamples, batchsize, augment=False):
		self.dataset = dataset
		self.next = next
		self.sess = sess
		self.nbsamples = nbsamples
		self.batchsize = batchsize
		self.augment = augment
		self.i = 1
		self.rot90 = {1:4,4:2,2:3,3:1,0:0} #correspondance between old labels and new labels after a 90 degree rotation 
		self.rot_90 = {1:3,3:2,2:4,4:1,0:0} #correspondance between old labels and new labels after a - 90 degree rotation
		self.flip1 = {1:2,2:1,0:0,3:3,4:4} #correspondance between old labels and new labels after an horizontal flip
		self.flip0 = {3:4,4:3,0:0,1:1,2:2} #correspondance between old labels and new labels after a vertical flip

	def __len__(self):
		#Number of batches per epoch
		return self.nbsamples // self.batchsize

	def __getitem__(self, index):
		#Generate one batch of data
		try:
			batch = self.sess.run(self.next)
		except tensorflow.python.framework.errors_impl.InvalidArgumentError:
			print("ERROR")
			return __getitem__(self, index)

		labels = batch[0].reshape((-1,))
		p1 = batch[1]
		p2 = batch[2]
		if self.augment == True:
			for i in range(p1.shape[0]):
				rot = np.random.randint(0,5)
				flip = np.random.randint(0,5)
				if rot == 1: # 1 in 4 chances of 90 degree rotation
					p1[i,:,:,:] = np.rot90(p1[i,:,:,:] ,k=1,axes=(1,0))
					p2[i,:,:,:]  = np.rot90(p2[i,:,:,:] ,k=1,axes=(1,0))
					labels[i] = self.rot90[labels[i]]
				if rot == 2: # 1 in 4 chances of - 90 degree rotation
					p1[i,:,:,:] = np.rot90(p1[i,:,:,:] ,k=3,axes=(1,0))
					p2[i,:,:,:]  = np.rot90(p2[i,:,:,:] ,k=3,axes=(1,0))
					labels[i] = self.rot_90[labels[i]]
				if flip == 1: # 1 in 4 chances of horizontal flip
					p1[i,:,:,:] = np.flip(p1[i,:,:,:] ,axis=1)
					p2[i,:,:,:]  = np.flip(p2[i,:,:,:] ,axis=1)
					labels[i] = self.flip1[labels[i]]
				if flip == 2: # 1 in 4 chances of vertical flip
					p1[i,:,:,:] = np.flip(p1[i,:,:,:] ,axis=0)
					p2[i,:,:,:]  = np.flip(p2[i,:,:,:] ,axis=0)
					labels[i] = self.flip0[labels[i]]
		else:
			#print(labels)
			pass	
		try:
			#reshape to feed to TF neural network
			p1=p1.reshape((-1,200,200,5))
			p2=p2.reshape((-1,200,200,5))
		return [p1,p2], keras.utils.to_categorical(labels, num_classes=5)

### TF and Keras options
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

### Load dataset
train_filename = './preprocessed/pairs_ostraca2.tfrecords'
epochs=100
batch_size = 40 
data_path = tf.placeholder(dtype=tf.string, name="tfrecord_file")


dataset = tf.data.TFRecordDataset(data_path)
dataset = dataset.map(_read_from_tfrecord)
dataset2 = dataset.take(7000).shuffle(7000,seed=None,reshuffle_each_iteration=False) # take the 7000 first elements of the dataset, and shuffle the elements
dataval = dataset2.take(1000) # take the first 1000 elements as validation dataset
dataval = dataval.repeat().batch(batch_size) 
datatrain = dataset2.skip(1000) # take the remaining 6000 elements as training dataset
datatrain = datatrain.shuffle(buffer_size=6000).repeat().batch(batch_size) #shuffle training dataset at each epoch

gc.collect()

offset = 10 #size of ROI for cropping the borders of the feature map after the last conv layer
reg = 0.1 # L2 regularization factor, to reduce overfitting
size = 200 # input image size

### Define NN architecture (net is stored on cpu to save gpu memory)
with tf.device('/cpu:0'):

	left_in = Input((size,size,5))
	right_in = Input((size,size,5))
	inputs = Input((size,size,5))

	x = BatchNormalization(axis=-1, momentum=0.99)(inputs)

	c2 = Conv2D(32, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)
	x = BatchNormalization(axis=-1, momentum=0.99)(c2)
	o2 = keras.layers.LeakyReLU(alpha=0.01)(x)
	x = MaxPooling2D(pool_size=3, strides=2, padding="same")(o2)

	c3 = Conv2D(64, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)
	x = BatchNormalization(axis=-1, momentum=0.99)(c3)
	o3 = keras.layers.LeakyReLU(alpha=0.01)(x)
	x = MaxPooling2D(pool_size=3, strides=2, padding="same")(o3)

	c4 = Conv2D(128, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)
	x = BatchNormalization(axis=-1, momentum=0.99)(c4)
	o4 = keras.layers.LeakyReLU(alpha=0.01)(x)
	x = MaxPooling2D(pool_size=3, strides=2, padding="same")(o4)

	enc = Conv2D(256, 3, padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)
	x = BatchNormalization(axis=-1, momentum=0.99)(enc)
	res = keras.layers.LeakyReLU(alpha=0.01)(x)

	model = Model(inputs=inputs, outputs=res)
	model.summary()

	encoded_l = model(left_in)
	encoded_r = model(right_in)

	print(encoded_l.shape)
	G1 = Cropping2D(((0,0),(0,25-offset)))(encoded_l)
	print(G1.shape)
	D1 = Cropping2D(((0,0),(25-offset,0)))(encoded_l)
	print(D1.shape)
	H1 = Cropping2D(((0,25-offset),(0,0)))(encoded_l)
	print(H1.shape)
	B1 = Cropping2D(((25-offset,0),(0,0)))(encoded_l)
	print(B1.shape)

	G2 = Cropping2D(((0,0),(0,25-offset)))(encoded_r)
	D2 = Cropping2D(((0,0),(25-offset,0)))(encoded_r)
	H2 = Cropping2D(((0,25-offset),(0,0)))(encoded_r)
	B2 = Cropping2D(((25-offset,0),(0,0)))(encoded_r)

	D1G2= keras.layers.subtract([D1, G2])
	D1G2 = Lambda (K.abs)(D1G2)
	D2G1= keras.layers.subtract([D2, G1])
	D2G1 = Lambda (K.abs)(D2G1)
	H1B2= keras.layers.subtract([H1, B2])
	H1B2 = Lambda (K.abs)(H1B2)
	H1B2 = Lambda (lambda x:K.permute_dimensions(x,(0,2,1,3)))(H1B2)
	H2B1= keras.layers.subtract([H2, B1])
	H2B1 = Lambda (K.abs)(H2B1)
	H2B1 = Lambda (lambda x:K.permute_dimensions(x,(0,2,1,3)))(H2B1)


	concat = Concatenate(axis=-1)([D1G2,D2G1,H1B2,H2B1])
	x = keras.layers.GlobalAveragePooling2D()(concat)
	x = Dense(100)(x)
	x = keras.layers.LeakyReLU(alpha=0.01)(x)
	x = Dropout(0.5)(x)
	res = Dense(5, activation="softmax")(x)

	net = Model(inputs=[left_in,right_in],outputs=res)
	
parallel_model = multi_gpu_model(net,gpus=2) # distribute model on two gpus		

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
parallel_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy','mean_squared_logarithmic_error'])
net.summary() # print description of the network layer by layer

### Initialize iterators for the training and validation datasets
iter1 = datatrain.make_initializable_iterator()
next1 = iter1.get_next()
iter2 = dataval.make_initializable_iterator()
next2 = iter2.get_next()
sess.run(iter1.initializer, feed_dict={data_path: train_filename})
sess.run(iter2.initializer, feed_dict={data_path: train_filename})
print("init iterateur ok")

### Plot an example of patch pair in the validation dataset
batch0 = sess.run(next2)
p1 = batch0[1].reshape((-1,size,size,5))
p2 = batch0[2].reshape((-1,size,size,5))
plt.figure()
plt.imshow(p1[1][:,:,0:3],"gray")
plt.figure()
plt.imshow(p2[1][:,:,0:3],"gray")
plt.show()
l = batch0[0]

### Save a batch of patch pairs and labels from the validation dataset
np.savez("validData.npz",p1,p2,l)

### Create generators for the training and validation datasets
train_gen = DataGenerator(datatrain, next1, sess, 6000, batch_size, True) # on-the-fly data augmentation for training set
val_gen = DataGenerator(dataval, next2, sess, 1000, batch_size, False)

### Run training and validation on the model	
history = parallel_model.fit_generator(train_gen, epochs=epochs, verbose=2, validation_data=val_gen, max_queue_size=40, workers=1, use_multiprocessing=False, callbacks=[keras.callbacks.CSVLogger("log_ostraca.csv", separator=',', append=False)])

### Save trained model
net.save("patchAssemble_ostraca.h5")	
	
### Plot the metrics
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['mean_squared_logarithmic_error'])
plt.plot(history.history['val_mean_squared_logarithmic_error'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Val loss','Train MSLE', 'Val MSLE'], loc='upper left')
plt.show()	
	
