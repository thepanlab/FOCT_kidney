#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:26:00 2020

@author: paulcalle
"""
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0-preview is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os

# ------------------------------------
# Code for testing kidneys 1, 2, and 3
# ------------------------------------

# Reading inputs

import numpy as np

with open('/scratch/paulcalle/kidney_data/a_images_1D.npy', 'rb') as f:
    a_images_1D = np.load(f)
    
with open('/scratch/paulcalle/kidney_data/a_label.npy', 'rb') as f:
    a_label = np.load(f)

with open('/scratch/paulcalle/kidney_data/a_kidney_num.npy', 'rb') as f:
    a_kidney_num = np.load(f)
    
# Transforming labels to numerical categories

a_label_num = np.copy(a_label)

a_label_num[a_label_num == "medulla"] = 0
a_label_num[a_label_num == "cortex"] = 1
a_label_num[a_label_num == "pelvis_calyx"] = 2

a_label_num = a_label_num.astype(int)

a_images_1D_float64 = a_images_1D.astype(float)

## ResNet-34

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

from functools import partial

# Code for ResNet34 taken from repository of book
# "Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow" 
# By Aurelien Geron
# Chapter 14
# https://github.com/ageron/handson-ml2/blob/master/14_deep_computer_vision_with_cnns.ipynb

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
    
import numpy as np
import pickle

import numpy as np
import pickle
import pandas as pd 
df_results = pd.DataFrame() 

# a_kidneys_num = np.arange(1,11,1)
a_kidneys_num = np.unique(a_kidney_num)

a_selected_kidneys_test = np.array([1,2,3])
# if all use : a_kidneys_num
# a_selected_kidneys_test = a_kidneys_num

print("a_selected_kidneys_test = ", a_selected_kidneys_test)

# a_selected_kidneys_val = np.array([4,7,9])
# if all use : a_kidneys_num
a_selected_kidneys_val = a_kidneys_num
bool_save = True
# for index in np.arange(1,11,1):
for index in a_selected_kidneys_test:

    print("**Kidney test**: " + str(index) )

    a_kidneys_num_val = np.delete(a_selected_kidneys_val, np.where( a_selected_kidneys_val == index))
    print("a_kidneys_num_val = ", a_kidneys_num_val)   
    
    bool_kidney_num = a_kidney_num != index
    a_images_1D_9_kidneys = a_images_1D_float64[bool_kidney_num]
    a_label_num_9_kidneys = a_label_num[bool_kidney_num]
    a_kidney_num_9_kidneys = a_kidney_num[bool_kidney_num]
    
    
    print(len(a_images_1D_9_kidneys))
    print(len(a_label_num_9_kidneys))
    
    X_cv = a_images_1D_9_kidneys
    y_cv = a_label_num_9_kidneys
    
    n_epochs = 200

    # k fold validation  max:9 fold

    for index_val in a_kidneys_num_val:

        print("Kidney_val: " + str(index_val) )
        
        bool_val_kidney = ( a_kidney_num_9_kidneys == index_val )
        bool_train_kidney = ~bool_val_kidney
        
        X_train, X_val = X_cv[bool_train_kidney], X_cv[bool_val_kidney]
        y_train, y_val = y_cv[bool_train_kidney], y_cv[bool_val_kidney]        
       
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # RESNET model
        
        model = keras.models.Sequential()
        model.add(DefaultConv2D(64, kernel_size=7, strides=4,
                                input_shape=[301, 235, 1]))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
        prev_filters = 64
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == prev_filters else 2
            model.add(ResidualUnit(filters, strides=strides))
            prev_filters = filters
        model.add(keras.layers.GlobalAvgPool2D())
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(3, activation="softmax"))
        
#         checkpoint_cb = keras.callbacks.ModelCheckpoint("model_K%s_outer_k%s_val.h5"%(index, index_val),
#                                                         save_best_only=True)
        
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
        
        optimizer = keras.optimizers.Adam( learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                            name="Adam")

        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        
        history = model.fit(X_train.reshape(-1,301, 235, 1), y_train, epochs=n_epochs,
                            batch_size=512,
                            validation_data=(X_val.reshape(-1,301, 235, 1), y_val),
                            callbacks=[early_stopping_cb])

        y_proba = model.predict(X_val.reshape(-1,301, 235, 1))
        
#         print("y_proba = ", y_proba.dtype)
        
        with open('/scratch/paulcalle/kidney_results/archRESNET1_results/pred_val_K%s_outer_k%s_val.npy'%(index, index_val), 'wb') as f:
            np.save(f, y_proba)
        
        with open('/scratch/paulcalle/kidney_results/archRESNET1_results/history_K%s_outer_k%s_val'%(index, index_val), 'wb') as handle:
            pickle.dump(history.history, handle)