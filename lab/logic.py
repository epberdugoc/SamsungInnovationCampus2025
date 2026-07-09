import json
import tensorflow as tf 
from tensorflow import keras
import numpy as np

def tf_version():
    return tf.__version__

# Definir la métrica manualmente para la carga
custom_objects = {'mse': tf.keras.metrics.MeanSquaredError()}

model = keras.models.load_model('model/mpg_regr_model_es.h5', custom_objects = custom_objects)

def predict(input_list):

    return model.predict(np.array(input_list)).flatten()