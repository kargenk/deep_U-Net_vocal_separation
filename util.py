from librosa.core import load, resample, stft, istft
from librosa.output import write_wav
from librosa.util import find_files
import numpy as np
import const as C
import os

# スペクトログラムを生成する関数
def make_spectrograms(y_mixture, y_instrumental):
    # 短時間フーリエ変換(STFT)による各スペクトログラムの生成
    mix_spec = np.abs(
                    stft(y_mixture, n_fft=C.FFT_SIZE, hop_length=C.HOP_LENGTH))\
                .astype(np.float32)
    inst_spec = np.abs(
                    stft(y_instrumental, n_fft=C.FFT_SIZE, hop_length=C.HOP_LENGTH))\
                .astype(np.float32)
        
    # ボーカル部分は原曲とインストのスペクトログラムの減算によって生成
    vocal_spec = np.maximum(0, mix_spec - inst_spec)
    
    return mix_spec, inst_spec, vocal_spec

# 生成したボーカル音源を保存(wav形式)する関数
def save_vocal(y_mixture, y_instrumental, file_name):
    mix_spec, inst_spec, vocal_spec = make_spectrograms(y_mixture, y_instrumental)
    
    phase = np.exp(1.j*np.angle(vocal_spec))  # 位相情報
    
    # 逆短時間フーリエ変換により，位相情報を含むスペクトログラムから音声信号を復元
    y_vocal = istft(vocal_spec*phase,
                    hop_length=C.HOP_LENGTH, win_length=C.FFT_SIZE)
    
    # ファイル名を'曲名_vocal.wav'にして保存
    write_wav('./src/audio_check/' + file_name + '_vocal' + '.wav', y_vocal, C.SAMPLING_RATE)
    print('Saving: ./src/audio_check/' + file_name + '.wav')

# スペクトログラムを正規化して保存(npz形式)する関数
def save_spectrogram(y_mixture, y_instrumental, file_name):
    mix_spec, inst_spec, vocal_spec = make_spectrograms(y_mixture, y_instrumental)
    
    # 各スペクトログラムを正規化
    norm = mix_spec.max()
    mix_spec /= norm
    inst_spec /= norm
    vocal_spec /= norm
    
    # 保存
    np.savez(os.path.join(C.PATH_FFT, file_name + '.npz'),
             vocal=vocal_spec, mix=mix_spec, inst=inst_spec)
    print('Saving: ' + C.PATH_FFT + file_name + '.npz')