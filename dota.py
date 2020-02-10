# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 07:04:07 2020

@author: FUJITSU
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

# Pandas baca CSV
sf_train = pd.read_csv('p5_training_data.csv')

# korelasi Matrix untuk target
corr_matrix = sf_train.corr()
print(corr_matrix['type'])

# hapus kolom yang tidak terpakai
sf_train.drop(sf_train.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)
print(sf_train.head())

# Pandas validasi CSV
sf_val = pd.read_csv('p5_val_data.csv')

# hapus kolom yang tidak terpakai(tidak saling bergantungan)
sf_val.drop(sf_val.columns[[5, 12, 14, 21, 22, 23]], axis=1, inplace=True)

# Mengambil nilai array Pandas (konversi menjadi NumPy array)
train_data = sf_train.values
val_data = sf_val.values

# gunakan kolom ke 2 hingga terakhir sebagai Inputan
train_x = train_data[:,2:]
val_x = val_data[:,2:]

# Gunakan kolom pertaama sebagai Output (One-Hot Encoding)
train_y = to_categorical( train_data[:,1] )
val_y = to_categorical( val_data[:,1] )

# Membangun jaringan
inputs = Input(shape=(16,))
h_layer = Dense(10, activation='sigmoid')(inputs)

# Aktivasi Softmax untuk Multiclass Classification
outputs = Dense(3, activation='softmax')(h_layer)

model = Model(inputs=inputs, outputs=outputs)

# Optimizer / Update Rule
sgd = SGD(lr=0.001)

# Compile model dengan Cross Entropy Loss
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Train model dan menggunakan data validasi
model.fit(train_x, train_y, batch_size=16, epochs=5000, verbose=1, validation_data=(val_x, val_y))
model.save_weights('weights.h5')

# prediksi data validasi
predict = model.predict(val_x)

# Visualisasi hasil Prediksi
df = pd.DataFrame(predict)
df.columns = [ 'Strength', 'Agility', 'Intelligent' ]
df.index = val_data[:,0]
print(df)