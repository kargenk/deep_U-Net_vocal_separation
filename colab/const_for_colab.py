#!usr/bin/env python3
# -*- coding:utf-8 -*-

SAMPLING_RATE = 16000              # サンプリング周波数:16kHz
SAMPLING_STRIDE = 10               # 学習データの移動幅
HOP_LENGTH = 768                   # 窓関数を掛ける時の移動(ずらし)幅
FFT_SIZE = 1024                    # 高速フーリエ変換(FFT)を行うときの幅
IMAGE_HEIGHT = 512                 # 入力画像の縦サイズ
IMAGE_WIDTH = 128                  # 入力画像の横サイズ
BATCH_SIZE = 64
PATCH_LENGTH = 128

PATH_FFT = './spectrograms'        # spectrogramを保存するディレクトリ
PATH_AUDIO = './src/audio_check/'  # データセットを保存するディレクトリ