
# coding: utf-8

# In[6]:

import sys
import os
import dlib
import glob
from skimage import io


# In[7]:

predictor_path = 'shape_predictor_68_face_landmarks.dat'
fr_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
dataset_path = 'train_equal_re/'


# In[8]:

detector = dlib.get_frontal_face_detector() #detecting the face
sp_predictor = dlib.shape_predictor(predictor_path) #predicting 68 point landmarks
facerec = dlib.face_recognition_model_v1(fr_model_path) #resnet model for 128D vector
win = dlib.image_window()


# In[9]:

def feature_gen(path, image_file, visualization = False):
    img = io.imread(path+image_file)
    bounding_box = detector(img,1)[0] #detecting
    D68_shape_vector = sp_predictor(img, bounding_box) #68D landmark
    D128_facial_features = list(facerec.compute_face_descriptor(img, D68_shape_vector)) #128D resnet features
    if visualization:
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(shape)
        win.add_overlay(bounding_box)
        dlib.hit_enter_to_continue()
    return [bounding_box.left(), bounding_box.top(), bounding_box.right(), bounding_box.bottom()], D128_facial_features


# In[10]:

#feature_gen(dataset_path, 'cla1.jpg')


# In[11]:

def main():
    for f in os.listdir(dataset_path):
        print("Processing file: {}".format(f))
        img = io.imread(dataset_path+f)

        win.clear_overlay()
        win.set_image(img)

        dets = detector(img, 1)
        print("dets", dets[0])
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            shape = predictor(img, d)
            win.add_overlay(shape)

        win.add_overlay(dets)
        dlib.hit_enter_to_continue()

