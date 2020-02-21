import numpy as np
import os
import sys
import skimage.draw
import cv2

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True

session = InteractiveSession(config=config)
     
class custom_Mrcnn():
     
    def __init__(self,custom_config,socket):
     self.class_names = ['BG', 'arm', 'ring']
     PRETRAINED_MODEL_PATH = "mask_rcnn_surgery_0060.h5"
     self.custom_config = custom_config
     self.socket = socket
     
     self.model = modellib.MaskRCNN(mode="inference", config=self.custom_config, model_dir='')
     model_path = PRETRAINED_MODEL_PATH
     # or if you want to use the latest trained model, you can use :
     # model_path = model.find_last()[1]
     self.model.load_weights(model_path, by_name=True)

     self.colors = visualize.random_colors(len(self.class_names))
    def run(self,img=""):
     #img = cv2.imread(img)
     
     predictions = self.model.detect([img],verbose=1)  
                      
     p = predictions[0]
    
     output,attrs = visualize.display_instances(img, p['rois'], p['masks'], p['class_ids'],self.class_names, p['scores'], colors=self.colors, real_time=True)
     cv2.namedWindow("ImageWindow", cv2.WINDOW_AUTOSIZE);
     cv2.imshow('ImageWindow',output)
     print("attrs")
     print(attrs)
     
     #result, frame = cv2.imencode('.jpg', frame, encode_param)
    #data = zlib.compress(pickle.dumps(frame, 0))
     self.socket.send(output,attrs)   
     cv2.waitKey(10)
     
      
     
