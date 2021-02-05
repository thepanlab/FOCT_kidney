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

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os

from functools import partial     
import pickle

from time import perf_counter 

from tensorflow.python.client.device_lib import list_local_devices

# tf.debugging.set_log_device_placement(True)

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = perf_counter()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(perf_counter()-self.starttime)

# only one value for paramameter
try:
    K_test = int(sys.argv[1])
    k_val = int(sys.argv[2])
except IndexError:
    raise SystemExit(f"Usage: {sys.argv[0]} <K_test>  <k_val>")

devices = list_local_devices()
print("devices = ", devices)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# tf.config.threading.set_inter_op_parallelism_threads(7)
# tf.config.threading.set_intra_op_parallelism_threads(4)

# inter_op = tf.config.threading.get_inter_op_parallelism_threads()
# intra_op = tf.config.threading.get_intra_op_parallelism_threads()

# print("inter_op = ", inter_op)
# print("intra_op = ", intra_op)

# Reading inputs

with open('/gpfs/alpine/bif121/proj-shared/kidney/kidney_data/a_images_1D.npy', 'rb') as f:
    a_images_1D = np.load(f)
    
with open('/gpfs/alpine/bif121/proj-shared/kidney/kidney_data/a_label.npy', 'rb') as f:
    a_label = np.load(f)

with open('/gpfs/alpine/bif121/proj-shared/kidney/kidney_data/a_kidney_num.npy', 'rb') as f:
    a_kidney_num = np.load(f)

# Transforming labels to numerical categories

a_label_num = np.copy(a_label)

a_label_num[a_label_num == "medulla"] = 0
a_label_num[a_label_num == "cortex"] = 1
a_label_num[a_label_num == "pelvis_calyx"] = 2

a_label_num = a_label_num.astype(int)

a_images_3D = np.repeat(a_images_1D[..., np.newaxis], 3, -1) 
del a_images_1D

print("images_resized")
with tf.device('/CPU:0'):
    images_resized = tf.image.resize_with_pad(a_images_3D, 224, 224, antialias=True)
    np_images_resized = images_resized.numpy()
    del images_resized
del a_images_3D

print("input_resnet50")
input_resnet50 = keras.applications.resnet50.preprocess_input(np_images_resized)

# ## ResNet-50 TF

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# a_kidneys_num = np.arange(1,11,1)
a_kidneys_num = np.unique(a_kidney_num)

a_selected_kidneys_test = np.array([K_test])
# if all use : a_kidneys_num
# a_selected_kidneys_test = a_kidneys_num

print("a_selected_kidneys_test = ", a_selected_kidneys_test)

# a_selected_kidneys_val = np.array([4,7,9])
a_selected_kidneys_val = np.array([k_val])
# if all use : a_kidneys_num
# a_selected_kidneys_val = a_kidneys_num

for index in a_selected_kidneys_test:

    print("**Kidney test**: " + str(index) )

    a_kidneys_num_val = np.delete(a_selected_kidneys_val, np.where( a_selected_kidneys_val == index))
    print("a_kidneys_num_val = ", a_kidneys_num_val)   
    
    bool_kidney_num = a_kidney_num != index

    print("a_images_1D_9_kidneys Start")
    a_images_1D_9_kidneys = input_resnet50[bool_kidney_num]
    
    a_label_num_9_kidneys = a_label_num[bool_kidney_num]
    a_kidney_num_9_kidneys = a_kidney_num[bool_kidney_num]

    print("a_images_1D_9_kidneys End")

    print(len(a_images_1D_9_kidneys))
    print(len(a_label_num_9_kidneys))
    
    print("CV Start")
    X_cv = a_images_1D_9_kidneys
    y_cv = a_label_num_9_kidneys
    print("CV End")
    
    n_epochs = 50
    batch_size = 32

#     # k fold validation  max:9 fold

    for index_val in a_kidneys_num_val:

        print("Kidney_val: " + str(index_val) )
        
        bool_val_kidney = ( a_kidney_num_9_kidneys == index_val )
        bool_train_kidney = ~bool_val_kidney

        print("CV train val start")
      
        X_train, X_val = X_cv[bool_train_kidney], X_cv[bool_val_kidney]
        y_train, y_val = y_cv[bool_train_kidney], y_cv[bool_val_kidney]        
    
        print("CV train val end")


        print("Model Start")
    
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # RESNET 50 TL

        # Using weights = "imagenet" doesn't work. It got stuck and cannot download
        # Better to download it from other machine
        
        base_model_resnet50 = keras.applications.resnet50.ResNet50( include_top=False,
                                              weights="/gpfs/alpine/bif121/proj-shared/kidney/resnet50_weights_schooner/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                              input_tensor=None,
                                              input_shape=(224,224,3),
                                              pooling=None)

        n_classes=3

        avg = keras.layers.GlobalAveragePooling2D()(base_model_resnet50.output)
        output = keras.layers.Dense(n_classes, activation="softmax")(avg)
        model_tf = keras.models.Model(inputs=base_model_resnet50.input, outputs=output)

        for layer in base_model_resnet50.layers:
            layer.trainable = False

        optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                          restore_best_weights=True)


        model_tf.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                      metrics=["accuracy"])

        print("Model End")

        time_cb = TimingCallback()

        t1_start = perf_counter() 

        history = model_tf.fit(X_train, y_train,
                               batch_size=batch_size,
                               validation_data=(X_val, y_val),
                               epochs=n_epochs,
                               callbacks=[early_stopping_cb,
                                          time_cb])
        
        y_proba = model_tf.predict(X_val)

        with open('/gpfs/alpine/bif121/proj-shared/kidney/kidney_results/archRESNET50_tf_results/time_epoch_a_K%s_outer_k%s_val.npy'%(index, index_val), 'wb') as f:
            np.save(f, np.array(time_cb.logs))

        with open('/gpfs/alpine/bif121/proj-shared/kidney/kidney_results/archRESNET50_tf_results/history_a_K%s_outer_k%s_val'%(index, index_val), 'wb') as handle:
            pickle.dump(history.history, handle)

        with open('/gpfs/alpine/bif121/proj-shared/kidney/kidney_results/archRESNET50_tf_results/pred_val_a_K%s_outer_k%s_val.npy'%(index, index_val), 'wb') as f:
            np.save(f, y_proba)

        # 2nd stage

        for layer in base_model_resnet50.layers:
            layer.trainable = True

        optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,
                                     nesterov=True, decay=0.001)
        
        time_cb = TimingCallback()

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
        
        model_tf.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
        history = model_tf.fit(X_train, y_train,
                               batch_size=batch_size,
                               validation_data=(X_val, y_val),
                               epochs=n_epochs,
                               callbacks=[early_stopping_cb,
                                          time_cb])

        t1_stop = perf_counter()        
        time_lapse = t1_stop-t1_start
                
        y_proba = model_tf.predict(X_val)

        print( "Elapsed time during the whole program in seconds for K%s_outer_k%s_val: "%(index, index_val), time_lapse)
        
        with open('/gpfs/alpine/bif121/proj-shared/kidney/kidney_results/archRESNET50_tf_results/time_total_K%s_outer_k%s_val.npy'%(index, index_val), 'wb') as f:
            np.save(f, np.array(time_lapse))

        with open('/gpfs/alpine/bif121/proj-shared/kidney/kidney_results/archRESNET50_tf_results/time_epoch_b_K%s_outer_k%s_val.npy'%(index, index_val), 'wb') as f:
            np.save(f, np.array(time_cb.logs))

        with open('/gpfs/alpine/bif121/proj-shared/kidney/kidney_results/archRESNET50_tf_results/history_b_K%s_outer_k%s_val'%(index, index_val), 'wb') as handle:
            pickle.dump(history.history, handle)

        with open('/gpfs/alpine/bif121/proj-shared/kidney/kidney_results/archRESNET50_tf_results/pred_val_b_K%s_outer_k%s_val.npy'%(index, index_val), 'wb') as f:
            np.save(f, y_proba)