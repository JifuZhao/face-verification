#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" API for face detection and verification """

import cv2 as cv
from .utils import read_image, BGR2RGB, BGR2Gray

xml = './xmls/haarcascade_frontalface_default.xml'  # Haar Cascade detector
# xml = './xmls/lbpcascade_frontalface_improved.xml'  # LBP Cascade detector

def cascade_detector(image, xml=xml, scale_factor=1.3, min_neighbors=5):
    """ implement Haar or LBP Feature-based Cascade Classifiers from OpenCV
        change the xml to specify Haar or LBP Cascade detector
        note: the image format should be BGR, instead of RGB
    """
    face_detector = cv.CascadeClassifier(xml)
    gray_img = BGR2Gray(image)
    faces = face_detector.detectMultiScale(gray_img, scaleFactor=scale_factor,
                                           minNeighbors=min_neighbors)

    if len(faces) == 0:
        raise ValueError('Error, there is no faces.')
    elif len(faces) > 1:
        raise ValueError('Error, multiple faces are found.')

    return faces

def load_model():
    """ load pre-trained CNN models """
    pass

def prediction(model, image):
    """ make prediction using the pretrained model """
