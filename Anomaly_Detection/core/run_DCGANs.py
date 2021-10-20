"#_*_ coding:utf-8 _*_"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from sklearn.preprocessing import MinMaxScaler
import json
import random
import math
import tensorflow as tf
from tqdm import tqdm
from keras.layers import Masking, Embedding
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, BatchNormalization, Activation,Reshape, UpSampling2D, Conv2D, MaxPooling2D
from keras import optimizers
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam
import warnings
import time
import argparse
warnings.filterwarnings('ignore')

# load local package
from home.poac.code.Time_Series_Analysis.Anomaly_Detection.util.utils import search_threshold, load_data
from home.poac.code.Time_Series_Analysis.Anomaly_Detection.core.options import Options


def generator_model(input_size, settings):
    """
    Build generator model
    :param input_size:
    :param settings:
    :return:
    """
    # read settings
    dense_1_dim_g = settings["dense_1_dim_g"]
    activation = settings["activation"]
    dense_2_dim_g = settings["dense_2_dim_g"]
    reshape_high_g = settings["reshape_high_g"]
    upsampling_1_g_size = settings["upsampling_1_g_size"]
    filter_1_g = settings["filter_1_g"]
    kernel_1_g_size = settings["kernel_1_g_size"]
    upsampling_2_g_size = settings["upsampling_2_g_size"]
    filter_2_g = settings["filter_2_g"]
    kernel_2_g_size = settings["kernel_2_g_size"]
    # build
    model = Sequential()
    model.add(Dense(units=dense_1_dim_g, input_dim=input_size))
    model.add(Activation(activation))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(dense_2_dim_g * reshape_high_g * reshape_high_g))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(Reshape((reshape_high_g, reshape_high_g, dense_2_dim_g), input_shape=(dense_2_dim_g * reshape_high_g * reshape_high_g,)))
    model.add(UpSampling2D(size=(upsampling_1_g_size, upsampling_1_g_size)))
    model.add(Conv2D(filters=filter_1_g, kernel_size=(kernel_1_g_size, kernel_1_g_size), padding='same'))
    model.add(Activation(activation))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(UpSampling2D(size=(upsampling_2_g_size, upsampling_2_g_size)))
    model.add(Conv2D(filters=filter_2_g, kernel_size=(kernel_2_g_size, kernel_2_g_size), padding='same'))
    model.add(Activation('tanh'))
    model.summary()
    return model


def discriminator_model(input_size, settings):
    """
    Build discriminator model
    :param input_size:
    :param settings:
    :return:
    """
    # read settings
    input_channel_d = settings["input_channel_d"]
    filter_1_d = settings["filter_1_d"]
    kernel_1_d_sie = settings["kernel_1_d_sie"]
    activation = settings["activation"]
    maxpool_1_size = settings["maxpool_1_size"]
    filter_2_d = settings["filter_2_d"]
    kernel_2_d_sie = settings["kernel_2_d_sie"]
    maxpool_2_size = settings["maxpool_2_size"]
    dense_1_dim_d = settings["dense_1_dim_d"]
    last_dense_dim_d = settings["last_dense_dim_d"]
    last_activation_d = settings["last_activation_d"]
    # build
    model = Sequential()
    model.add(Conv2D(filter_1_d, (kernel_1_d_sie, kernel_1_d_sie), padding='same', input_shape=(input_size, input_size, input_channel_d)))
    model.add(Activation(activation))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(maxpool_1_size, maxpool_1_size)))
    model.add(Conv2D(filter_2_d, (kernel_2_d_sie, kernel_2_d_sie)))
    model.add(Activation(activation))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(maxpool_2_size, maxpool_2_size)))
    model.add(Flatten())
    model.add(Dense(dense_1_dim_d))
    model.add(Activation(activation))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(last_dense_dim_d))
    model.add(Activation(last_activation_d))
    # model.add(LeakyReLU(alpha=0.05))
    model.summary()
    return model


def generator_containing_discriminator(g, d):
    """
    Contain generator and discriminator
    :param g:
    :param d:
    :return:
    """
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    model.summary()
    return model


def combine_images(generated_images):
    """
    Combine images
    :param generated_images:
    :return:
    """
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]
    return image


def train(opt, x_train):
    """
    train model
    :param opt:
    :param x_train:
    :return:
    """
    # read settings
    path = '/home/poac/code/Time_Series_Analysis/Anomaly_Detection/settings/setting.json'
    settings = json.load(open(path))[opt.dataset+"_settings"]
    dim = settings['dim']

    d = discriminator_model(input_size=dim, settings=settings)
    g = generator_model(input_size=opt.nz, settings=settings)
    d_on_g = generator_containing_discriminator(g, d)

    # SGD
    d_optim = SGD(lr=opt.lr_d, momentum=0.9, nesterov=True, clipvalue=0.5)
    g_optim = SGD(lr=opt.lr_g, momentum=0.9, nesterov=True, clipvalue=0.5)

    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    # real_label = 1
    # fake_label = 0

    for epoch in tqdm(range(opt.niter)):
        print("Epoch is", epoch)
        print("Number of batches", int(x_train.shape[0] / opt.batchsize))
        for index in range(int(x_train.shape[0] / opt.batchsize)):
            noise = np.random.uniform(-1, 1, size=(opt.batchsize, 100))
            image_batch = x_train[index * opt.batchsize:(index + 1) * opt.batchsize]
            generated_images = g.predict(noise, verbose=0)
            # train discriminator
            real_label = random.uniform(0.9, 1.0)
            fake_label = random.uniform(0.0, 0.1)
            # train with real sample
            X = image_batch
            y = [real_label] * opt.batchsize
            d_loss_r = d.train_on_batch(X, y)
            # train with fake sample
            X = generated_images
            y = [fake_label] * opt.batchsize
            d_loss_f = d.train_on_batch(X, y)
            # train loss
            d_loss = d_loss_r + d_loss_f
            # train generator
            real_label = 1
            fake_label = 0
            noise = np.random.uniform(-1, 1, (opt.batchsize, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [real_label] * opt.batchsize)
            d.trainable = True
        if epoch % 30 == 0 and epoch != 0:
            g.save_weights('/home/poac/code/Time_Series_Analysis/Anomaly_Detection/model_file/netG_' + opt.dataset + '_dcgan.h5', True)
            d.save_weights('/home/poac/code/Time_Series_Analysis/Anomaly_Detection/model_file/netD_' + opt.dataset + '_dcgan.h5', True)
        print('epoch:{} | epochs:{}   g loss:{}   d loss:{}'.format(epoch, opt.niter, g_loss, d_loss))
        # print(f"[Epoch {epoch:{opt.niter}}] "
        #       f"[Batch {{index}}] "
        #       f"[D loss: {d_loss.item():3f}] "
        #       f"[G loss: {g_loss.item():3f}]")

    # save model files
    g.save_weights('/home/poac/code/Time_Series_Analysis/Anomaly_Detection/model_file/netG_' + opt.dataset + '_dcgan.h5', True)
    d.save_weights('/home/poac/code/Time_Series_Analysis/Anomaly_Detection/model_file/netD_' + opt.dataset + '_dcgan.h5', True)
    print('netD had been saved to ../model_file/netD_' + opt.dataset + '_dcgan.h5')


def anomaly_detection(opt, x_test, label):
    """
    Anomaly detection
    :param opt:
    :param label:
    :return:
    """
    # read settings
    path = '/home/poac/code/Time_Series_Analysis/Anomaly_Detection/settings/setting.json'
    settings = json.load(open(path))[opt.dataset + "_settings"]
    dim = settings['dim']

    # load discriminator model parameters
    d = discriminator_model(input_size=dim, settings=settings)
    d_optim = SGD(lr=opt.lr_d, momentum=0.9, nesterov=True)
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    d.load_weights('/home/poac/code/Time_Series_Analysis/Anomaly_Detection/model_file/netD_' + opt.dataset + '_dcgan.h5')
    d.trainable = False

    # load generator model parameters
    g = generator_model(input_size=opt.nz, settings=settings)
    g_optim = SGD(lr=opt.lr_g, momentum=0.9, nesterov=True, clipvalue=0.5)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('/home/poac/code/Time_Series_Analysis/Anomaly_Detection/model_file/netG_' + opt.dataset + '_dcgan.h5')
    g.trainable = False

    # predict
    label = label.reshape(-1, 1)
    pred = d.predict(x_test, verbose=1)
    pred = np.array(pred)
    ll = max(2 * dim, settings['window_2'], settings['window_3'])
    label = label[ll-1:]
    pred = np.log(1 / (np.abs(pred)))
    sc = MinMaxScaler()
    pred = sc.fit_transform(pred)

    # search the best threshold based on matrix
    p, r, f, _, _, _ = search_threshold(pred=pred, label=label, methods="matrix", verbose=1)

    # search the best threshold based on ROC
    p, r, f, _, _, _ = search_threshold(pred=pred, label=label, methods="ROC", verbose=1)


def main(opt, model):
    """
    main
    :param opt:
    :param model:
    :return:
    """
    if opt.dataset == 'gens':
        train_path = '/home/poac/AnomalyDetectionDataset/Genesis_demonstrator_data/features_train.csv'
        test_path = '/home/poac/AnomalyDetectionDataset/Genesis_demonstrator_data/features_test.csv'
        label_path = '/home/poac/AnomalyDetectionDataset/Genesis_demonstrator_data/labels_test.csv'

    elif opt.dataset == 'satellite':
        train_path = "/home/poac/AnomalyDetectionDataset/satellite/train.csv"
        test_path = "/home/poac/AnomalyDetectionDataset/satellite/test.csv"
        label_path = "/home/poac/AnomalyDetectionDataset/satellite/label.csv"

    elif opt.dataset == 'shuttle':
        train_path = "/home/poac/AnomalyDetectionDataset/shuttle/train_features.csv"
        test_path = "/home/poac/AnomalyDetectionDataset/shuttle/test_features.csv"
        label_path = "/home/poac/AnomalyDetectionDataset/shuttle/test_label.csv"

    elif opt.dataset == 'gamaray':
        train_path = "/home/poac/AnomalyDetectionDataset/GamaRay/cleaned_train_65_sub.csv"
        test_path = "/home/poac/AnomalyDetectionDataset/GamaRay/cleaned_test_65_sub.csv"
        label_path = "/home/poac/AnomalyDetectionDataset/GamaRay/label_sub.csv"

    path = '/home/poac/code/Time_Series_Analysis/Anomaly_Detection/settings/setting.json'
    settings = json.load(open(path))[opt.dataset + "_settings"]
    window_2 = settings['window_2']
    window_3 = settings['window_3']
    dim = settings['dim']
    print('dim:', dim)

    if model == 'train':
        # train
        train_data = load_data(train_path=train_path, test_path=test_path, label_path=label_path,
                               window_1=2 * dim, window_2=window_2, window_3=window_3,
                                     dimension=dim, data_type=model, fm='p')
        train(opt=opt, x_train=train_data)
    else:
        test_data, label = load_data(train_path=train_path, test_path=test_path, label_path=label_path,
                                        window_1=2 * dim, window_2=window_2, window_3=window_3,
                                        dimension=dim, data_type=model, fm='p')
        print('test data shape:', test_data.shape)
        anomaly_detection(opt=opt, x_test=test_data, label=label)


if __name__ == "__main__":
    opt = Options().parse()
    if opt.model == 'DCGANs':
        if opt.phase == 'train':
            main(opt=opt, model='train')
        elif opt.phase == 'test':
            main(opt=opt, model='test')
        else:
            raise Exception('phase is wrong')
    else:
        raise Exception('model is wrong')



