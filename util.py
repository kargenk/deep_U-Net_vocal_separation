from librosa.core import load, resample, stft, istft
from librosa.output import write_wav
from librosa.util import find_files
import numpy as np
import const as C
import network
import os

# # スペクトログラムを生成する関数
# def make_spectrograms(y_mixture, y_instrumental):
#     # 短時間フーリエ変換(STFT)による各スペクトログラムの生成
#     mix_spec = np.abs(
#                     stft(y_mixture, n_fft=C.FFT_SIZE, hop_length=C.HOP_LENGTH))\
#                 .astype(np.float32)
#     inst_spec = np.abs(
#                     stft(y_instrumental, n_fft=C.FFT_SIZE, hop_length=C.HOP_LENGTH))\
#                 .astype(np.float32)
        
#     # ボーカル部分は原曲とインストのスペクトログラムの減算によって生成
#     vocal_spec = np.maximum(0, mix_spec - inst_spec)
    
#     return mix_spec, inst_spec, vocal_spec

# # 生成したボーカル音源を保存(wav形式)する関数
# def save_vocal(y_mixture, y_instrumental, file_name, opt='vocal'):
#     mix_spec, inst_spec, vocal_spec = make_spectrograms(y_mixture, y_instrumental)
    
#     phase = np.exp(1.j*np.angle(vocal_spec))  # 位相情報
    
#     # 逆短時間フーリエ変換により，位相情報を含むスペクトログラムから音声信号を復元
#     y_vocal = istft(vocal_spec*phase,
#                     hop_length=C.HOP_LENGTH, win_length=C.FFT_SIZE)
    
#     # ファイル名を'曲名_vocal.wav'にして保存
#     write_wav(C.PATH_AUDIO + file_name + '_' + opt + '.wav', y_vocal, C.SAMPLING_RATE)
#     print('Saving: ' + PATH_AUDIO + file_name + '_' + opt + '.wav')

# スペクトログラムを正規化して保存(npz形式)する関数
def save_spectrogram(y_mixture, y_instrumental, y_vocal, file_name, original_sr=44100):
    # 16kHzにダウンサンプリング
    y_mix = resample(y_mixture, original_sr, C.SAMPLING_RATE)
    y_inst = resample(y_instrumental, original_sr, C.SAMPLING_RATE)
#     y_vocal = resample(y_vocal, original_sr, C.SAMPLING_RATE)
    
    # 各スペクトログラムを生成
    mix_spec = np.abs(
                    stft(y_mixture, n_fft=C.FFT_SIZE, hop_length=C.HOP_LENGTH))\
                .astype(np.float32)
    inst_spec = np.abs(
                    stft(y_instrumental, n_fft=C.FFT_SIZE, hop_length=C.HOP_LENGTH))\
                .astype(np.float32)
    vocal_spec = np.abs(
                    stft(y_vocal, n_fft=C.FFT_SIZE, hop_length=C.HOP_LENGTH))\
                .astype(np.float32)
    
    # 各スペクトログラムを正規化
    norm = mix_spec.max()
    mix_spec /= norm
    inst_spec /= norm
    vocal_spec /= norm
    
    print(mix_spec.shape)
    print(vocal_spec.shape)
    print(inst_spec.shape)
    
    # 保存
    np.savez(os.path.join(C.PATH_FFT, file_name + '.npz'),
             mix=mix_spec, vocal=vocal_spec, inst=inst_spec)
#     print('Saving: ' + C.PATH_FFT + file_name + '.npz')

# データセット(.npz形式)を読み込む関数
def load_dataset(target='vocal'):
#     file_list_fft = find_files(C.PATH_FFT, ext='npz')[:200]
    X_list = []
    y_list = []
    file_list_fft = find_files(C.PATH_FFT, ext='npz')
    
    for file_fft in file_list_fft:
        data = np.load(file_fft)
        X_list.append(data['mix'])
        
        if target == 'vocal':
            assert(data['mix'].shape == data['vocal'].shape)
            y_list.append(data['vocal'])
        else:
            assert(data['mix'].shape == data['inst'].shape)
            y_list.append(data['inst'])
    return X_list, y_list

# 音源を読み込んで，スペクトログラムと位相情報を返す関数
def load_audio(file_name):
    y, sr = load(file_name, sr=C.SAMPLING_RATE)
    spectrum = stft(y, n_fft=C.FFT_SIZE, hop_length=C.HOP_LENGTH, win_length=C.FFT_SIZE)
    magnitude = np.abs(spectrum)
    magnitude /= np.max(magnitude)
    phase = np.exp(1.j*np.angle(spectrum))
    return magnitude, phase

# スペクトログラムと位相情報から音源を復元して保存する(wav形式)関数
def save_audio(file_name, magnitude, phase):
    y = istft(magnitude*phase, hop_length=C.HOP_LENGTH, win_length=C.FFT_SIZE)
    write_wav(file_name, y, C.SAMPLING_RATE, norm=True)

# マスクを計算する関数
def compute_mask(input_magnitude, unet_model="unet.model", hard=True):
    unet = network.UNet()
    unet.load(unet_model)
    config.train = False
    config.enable_backprop = False
    mask = unet(input_magnitude[np.newaxis, np.newaxis, 1:, :]).data[0, 0, :, :]
    mask = np.vstack((np.zeros(mask.shape[1], dtype="float32"), mask))
    if hard:
        hard_mask = np.zeros(mask.shape, dtype="float32")
        hard_mask[mask > 0.5] = 1
        return hard_mask
    else:
        return mask