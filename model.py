import numpy as np
from tensorflow.keras.models import model_from_json

import tensorflow as tf

# handling the frames on the gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)

class FEM(object):
    exps = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
#   loading the model and its weights for the prediction  
    def __init__(self, model_json_file, model_weight_file):
        with open(model_json_file, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
            
        self.loaded_model.load_weights(model_weight_file)
        self.loaded_model._make_predict_function()
        
#   predicting the expression    
    def predict_exp(self, img):
        self.preds = self.loaded_model.predict(img)
        return FEM.exps[np.argmax(self.preds)]