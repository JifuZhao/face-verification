#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Code modified from https://github.com/iwantooxxoox/Keras-OpenFace
# -------------------------------------------------------------------------
""" CNN modules for face verification """

import  warnings
warnings.simplefilter('ignore')

import numpy as np
from keras.layers import Dense, ZeroPadding2D, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Lambda, Flatten, concatenate
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf


def triplet_net(base_model, input_shape=(96, 96, 3)):
    """ define triplet networks """
    # define input: anchor, positive, negative images
    anchor = Input(shape=input_shape, name='anchor_input')
    positive = Input(shape=input_shape, name='positive_input')
    negative = Input(shape=input_shape, name='negative_input')

    # extract vector represent using CNN base model
    anc_vec = base_model(anchor)
    pos_vec = base_model(positive)
    neg_vec = base_model(negative)

    # stack outputs
    stacks = Lambda(lambda x: K.stack(x, axis=1), name='output')([anc_vec, pos_vec, neg_vec])

    # define inputs and outputs
    inputs=[anchor, positive, negative]
    outputs = stacks

    # define the triplet model
    model = Model(inputs=inputs, outputs=outputs, name='triplet_net')

    return model


def triplet_loss(margin=0.2):
    """ wrapper function for triplet loss """
    def loss(y_true, y_pred):
        """ function to calculate the triplet loss"""
        # define triplet margin
        margin = K.constant(0.2)
        zero = K.constant(0.0)

        # get the prediction vector
        anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

        # compute distance
        pos_distance = K.sum(K.square(anchor - positive), axis=1)
        neg_distance = K.sum(K.square(anchor - negative), axis=1)

        # compute loss
        partial_loss = pos_distance - neg_distance + margin
        full_loss = K.sum(K.maximum(partial_loss, zero), axis=0)

        return full_loss
    return loss


def LRN2D(x):
    """ local response normalization """
    lrn = tf.nn.local_response_normalization(x, alpha=1e-4, beta=0.75)
    return lrn


def conv2d_bn(x, layer=None, cv1_out=None, cv1_filter=(1, 1), cv1_strides=(1, 1),
              cv2_out=None, cv2_filter=(3, 3), cv2_strides=(1, 1), padding=None):
    """ 2D convolution with batch normalization """
    if cv2_out is None:
        num = ''
    else:
        num = '1'

    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, name=layer+'_conv'+num)(x)
    tensor = BatchNormalization(axis=3, epsilon=0.00001, name=layer+'_bn'+num)(tensor)
    tensor = Activation('relu')(tensor)

    if padding is None:
        return tensor

    tensor = ZeroPadding2D(padding=padding)(tensor)
    if cv2_out is None:
        return tensor

    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, name=layer+'_conv2')(tensor)
    tensor = BatchNormalization(axis=3, epsilon=0.00001, name=layer+'_bn2')(tensor)
    tensor = Activation('relu')(tensor)

    return tensor


def basenet(output_shape=128):
    """ create base-net for face verification """
    input_shape = (96, 96, 3)
    inputs = Input(shape=input_shape, name='input')

    # convolution layers
    x = ZeroPadding2D(padding=(3, 3), input_shape=input_shape)(inputs)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Lambda(LRN2D, name='lrn_1')(x)
    x = Conv2D(64, (1, 1), name='conv2')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), name='conv3')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
    x = Activation('relu')(x)
    x = Lambda(LRN2D, name='lrn_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # inception_3a
    inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
    inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
    inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

    inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
    inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
    inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

    inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
    inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
    inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
    inception_3a_pool = Activation('relu')(inception_3a_pool)
    inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

    inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
    inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
    inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

    inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

    # inception_3b
    inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
    inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
    inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

    inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
    inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
    inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

    inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3a)
    inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
    inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
    inception_3b_pool = Activation('relu')(inception_3b_pool)
    inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

    inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
    inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
    inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

    inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

    # inception_3c
    inception_3c_3x3 = conv2d_bn(inception_3b, layer='inception_3c_3x3', cv1_out=128, cv1_filter=(1, 1),
                                 cv2_out=256, cv2_filter=(3, 3), cv2_strides=(2, 2), padding=(1, 1))

    inception_3c_5x5 = conv2d_bn(inception_3b, layer='inception_3c_5x5', cv1_out=32, cv1_filter=(1, 1),
                                 cv2_out=64, cv2_filter=(5, 5), cv2_strides=(2, 2), padding=(2, 2))

    inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
    inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

    inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

    # inception_4a
    inception_4a_3x3 = conv2d_bn(inception_3c, layer='inception_4a_3x3', cv1_out=96, cv1_filter=(1, 1),
                                 cv2_out=192, cv2_filter=(3, 3), cv2_strides=(1, 1), padding=(1, 1))

    inception_4a_5x5 = conv2d_bn(inception_3c, layer='inception_4a_5x5', cv1_out=32, cv1_filter=(1, 1),
                                 cv2_out=64, cv2_filter=(5, 5), cv2_strides=(1, 1), padding=(2, 2))

    inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3c)
    inception_4a_pool = conv2d_bn(inception_4a_pool, layer='inception_4a_pool', cv1_out=128,
                                  cv1_filter=(1, 1), padding=(2, 2))

    inception_4a_1x1 = conv2d_bn(inception_3c, layer='inception_4a_1x1', cv1_out=256, cv1_filter=(1, 1))

    inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

    # inception_4b
    inception_4b_3x3 = conv2d_bn(inception_4a, layer='inception_4b_3x3', cv1_out=160, cv1_filter=(1, 1),
                                 cv2_out=256, cv2_filter=(3, 3), cv2_strides=(2, 2), padding=(1, 1))

    inception_4b_5x5 = conv2d_bn(inception_4a, layer='inception_4b_5x5', cv1_out=64, cv1_filter=(1, 1),
                                 cv2_out=128, cv2_filter=(5, 5), cv2_strides=(2, 2), padding=(2, 2))

    inception_4b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
    inception_4b_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4b_pool)

    inception_4b = concatenate([inception_4b_3x3, inception_4b_5x5, inception_4b_pool], axis=3)

    # inception_5a
    inception_5a_3x3 = conv2d_bn(inception_4b, layer='inception_5a_3x3', cv1_out=96, cv1_filter=(1, 1),
                                 cv2_out=384, cv2_filter=(3, 3), cv2_strides=(1, 1), padding=(1, 1))

    inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4b)
    inception_5a_pool = conv2d_bn(inception_5a_pool, layer='inception_5a_pool', cv1_out=96,
                                  cv1_filter=(1, 1), padding=(1, 1))

    inception_5a_1x1 = conv2d_bn(inception_4b, layer='inception_5a_1x1', cv1_out=256, cv1_filter=(1, 1))

    inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

    # inception_5b
    inception_5b_3x3 = conv2d_bn(inception_5a, layer='inception_5b_3x3', cv1_out=96, cv1_filter=(1, 1),
                                 cv2_out=384, cv2_filter=(3, 3), cv2_strides=(1, 1), padding=(1, 1))

    inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
    inception_5b_pool = conv2d_bn(inception_5b_pool, layer='inception_5b_pool', cv1_out=96, cv1_filter=(1, 1))
    inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

    inception_5b_1x1 = conv2d_bn(inception_5a, layer='inception_5b_1x1', cv1_out=256, cv1_filter=(1, 1))

    inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

    # final output layers
    av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
    reshape_layer = Flatten()(av_pool)
    dense_layer = Dense(output_shape, name='dense_layer')(reshape_layer)
    norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

    return Model(inputs=inputs, outputs=norm_layer)


def train_triplet_generator(df, batch_size=128, img_size=(96, 96), seed=42):
    """ training set triplet images generator """
    np.random.seed(seed)
    names = list(df['name'].unique())
    labels = np.zeros((batch_size, 3, 1), dtype=K.floatx())

    while True:
        np.random.shuffle(names)
        anchor_img_path = []
        positive_img_path = []
        negative_img_path = []

        # get the image path list for all images
        for i in range(len(names)):
            pair_list = df[df['name'] == names[i]]['path'].values
            anchor, positive = np.random.choice(pair_list, size=2, replace=False)
            neg_name = np.random.choice(names[:i] + names[i+1:], size=1)[0]
            negative = np.random.choice(df[df['name'] == neg_name]['path'].values, size=1)[0]

            anchor_img_path.append(anchor)
            positive_img_path.append(positive)
            negative_img_path.append(negative)

        # generate batch images
        for j in range(len(anchor_img_path) // batch_size):
            batch_anchor_img_path = anchor_img_path[j*batch_size : (j + 1)*batch_size]
            batch_positive_img_path = positive_img_path[j*batch_size : (j + 1)*batch_size]
            batch_negative_img_path = negative_img_path[j*batch_size : (j + 1)*batch_size]

            anchor_imgs = []
            positive_imgs = []
            negative_imgs = []

            # iteratively read images
            for k in range(batch_size):
                tmp_anc_img = load_img(batch_anchor_img_path[k], target_size=img_size)
                anchor_imgs.append(img_to_array(tmp_anc_img))

                tmp_pos_img = load_img(batch_positive_img_path[k], target_size=img_size)
                positive_imgs.append(img_to_array(tmp_pos_img))

                tmp_neg_img = load_img(batch_negative_img_path[k], target_size=img_size)
                negative_imgs.append(img_to_array(tmp_neg_img))

            # transform image list into array
            anc_imgs = np.array(anchor_imgs, dtype=K.floatx()) / 255.0
            pos_imgs = np.array(positive_imgs, dtype=K.floatx()) / 255.0
            neg_imgs = np.array(negative_imgs, dtype=K.floatx()) / 255.0

            yield [anc_imgs, pos_imgs, neg_imgs], labels


def test_triplet_generator(df, batch_size=100, img_size=(96, 96), seed=42):
    """ test set triplet images generator, it will generate 1000 pairs """
    names = list(df['name'].unique())
    labels = np.zeros((batch_size, 3, 1), dtype=K.floatx())

    while True:
        np.random.seed(seed)
        anchor_img_path = []
        positive_img_path = []
        negative_img_path = []

        # get the image path list for all images
        for outer in range(2):
            for i in range(len(names)):
                pair_list = df[df['name'] == names[i]]['path'].values
                anchor, positive = np.random.choice(pair_list, size=2, replace=False)
                neg_name = np.random.choice(names[:i] + names[i+1:], size=1)[0]
                negative = np.random.choice(df[df['name'] == neg_name]['path'].values, size=1)[0]

                anchor_img_path.append(anchor)
                positive_img_path.append(positive)
                negative_img_path.append(negative)

        # generate batch images
        for j in range(len(anchor_img_path) // batch_size):
            batch_anchor_img_path = anchor_img_path[j*batch_size : (j + 1)*batch_size]
            batch_positive_img_path = positive_img_path[j*batch_size : (j + 1)*batch_size]
            batch_negative_img_path = negative_img_path[j*batch_size : (j + 1)*batch_size]

            anchor_imgs = []
            positive_imgs = []
            negative_imgs = []

            # iteratively read images
            for k in range(batch_size):
                tmp_anc_img = load_img(batch_anchor_img_path[k], target_size=img_size)
                anchor_imgs.append(img_to_array(tmp_anc_img))

                tmp_pos_img = load_img(batch_positive_img_path[k], target_size=img_size)
                positive_imgs.append(img_to_array(tmp_pos_img))

                tmp_neg_img = load_img(batch_negative_img_path[k], target_size=img_size)
                negative_imgs.append(img_to_array(tmp_neg_img))

            # transform image list into array
            anc_imgs = np.array(anchor_imgs, dtype=K.floatx()) / 255.0
            pos_imgs = np.array(positive_imgs, dtype=K.floatx()) / 255.0
            neg_imgs = np.array(negative_imgs, dtype=K.floatx()) / 255.0

            yield [anc_imgs, pos_imgs, neg_imgs], labels
