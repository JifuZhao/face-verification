#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" unitity functions """

import cv2 as cv
import matplotlib.pyplot as plt

def read_image(path):
    """ function to read single image at the given path
        note: the loaded image is in B G R format
    """
    return cv.imread(path)

def BGR2RGB(image):
    """ function to transform image from BGR into RBG format """
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def BGR2Gray(image):
    """ function to transofrm image from BGR into Gray format """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def show_image(image, img_format='RGB', figsize=(8, 6)):
    """ function to show image """
    if img_format == 'RGB' or img_format == 'Gray':
        pass
    elif img_format == 'BGR':
        image = BGR2RGB(image)
    else:
        raise ValueError('format should be "RGB", "BGR" or "Gray"')

    fig, ax = plt.subplots(figsize=figsize)
    if format == 'Gray':
        ax.imshow(image, format='gray')
    else:
        ax.imshow(image)
    return fig

def denote_face(image, face):
    """ function to denote location of face on image """
    img = image.copy()
    for (x, y, w, h) in face:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img

def crop_face(image, face, scale_factor=1.0, target_size=(128, 128)):
    """ crop face at the given positons and resize to target size """
    rows, columns, channels = image.shape
    x, y, w, h = face[0]
    mid_x = x + w // 2
    mid_y = y + h // 2

    # calculate the new vertices
    x_new = mid_x - int(w // 2 * scale_factor)
    y_new = mid_y - int(h // 2 * scale_factor)
    w_new = int(w * scale_factor)
    h_new = int(h * scale_factor)

    # validate the new vertices
    left_x = max(0, x_new)
    left_y = max(0, y_new)
    right_x = min(columns, x_new + w_new)
    right_y = min(rows, y_new + h_new)

    # crop and resize the facial area
    cropped = image[left_y:right_y, left_x:right_x, :]
    resized = cv.resize(cropped, dsize=target_size, interpolation=cv.INTER_LINEAR)

    return resized
