#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import chainer
import cv2
import numpy as np
import os

import alex
import googlenet
import googlenetbn
import nin
import resnet50
import resnet101

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--img', type=str, default='data/cat.png',
                        help='Path to image file.')
    parser.add_argument('--mean', type=str, default='data/mean.npy')
    parser.add_argument('--arch', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet152', 'resnext50', 'resnext101'])
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    archs = {
        'alex': alex.Alex,
        'alex_fp16': alex.AlexFp16,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'googlenetbn_fp16': googlenetbn.GoogLeNetBNFp16,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50,
        'resnext50': resnet50.ResNeXt50,
        'resnet101': resnet101.ResNet101,
        'resnext101': resnet101.ResNeXt101,
    }

    model = archs[args.arch]()
    chainer.serializers.load_npz(args.model, model)
    mean = np.load(args.mean)

    x_data = img.transpose(2, 0, 1).astype(np.float32)[np.newaxis, :, :, :]
    x_data -= mean

    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        model.to_gpu()
        x_data = chainer.cuda.to_gpu(x_data)

    x_data = chainer.Variable(x_data)
    with chainer.using_config('train', False):
        pred = model.predict(x_data)
    pred = chainer.cuda.to_cpu(pred.data)

    with open('data/synset_words.txt') as f:
        synset = f.read().split('\n')[:-1]

    for i in np.argsort(pred)[0][-1::-1][:5]:
        print(synset[i])
