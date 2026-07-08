import json
import tensorflow as tf 
from tensorflow import keras
# import requests
import numpy as np
# import concurrent.futures as _futures
# # from itertools import chain as _chain
# # import time as _time
# # import base64 as _base64
# import os
# from sklearn.preprocessing import OneHotEncoder
# import pickle
# import pandas as pd

def tf_version():
    return tf.__version__

# Definir la métrica manualmente para la carga
custom_objects = {'mse': tf.keras.metrics.MeanSquaredError()}

model = keras.models.load_model('model/mpg_regr_model_es.h5', custom_objects=custom_objects)

def predict(input_list):

    return model.predict(np.array(input_list)).flatten()

# print(model.predict(np.array([[ 1.48388718,  1.86598835,  2.23462028,  1.01878165, -2.53089071,-1.60464169,  0.77467638, -0.46514837, -0.49522541]])))
