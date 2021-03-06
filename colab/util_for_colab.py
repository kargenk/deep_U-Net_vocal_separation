from librosa.core import load, resample, stft, istft, magphase
from librosa.output import write_wav
from librosa.util import find_files
import numpy as np
import const_for_colab as C
# import network_for_colab as network
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
def save_vocal(y_mixture, y_instrumental, file_name, opt='vocal'):
    mix_spec, inst_spec, vocal_spec = make_spectrograms(y_mixture, y_instrumental)
    
    phase = np.exp(1.j*np.angle(vocal_spec))  # 位相情報
    
    # 逆短時間フーリエ変換により，位相情報を含むスペクトログラムから音声信号を復元
    y_vocal = istft(vocal_spec*phase,
                    hop_length=C.HOP_LENGTH, win_length=C.FFT_SIZE)
    
    # ファイル名を'曲名_vocal.wav'にして保存
    write_wav(C.PATH_AUDIO + file_name + '_' + opt + '.wav', y_vocal, C.SAMPLING_RATE)
    print('Saving: ' + PATH_AUDIO + file_name + '_' + opt + '.wav')

# スペクトログラムを正規化して保存(npz形式)する関数
def save_spectrogram(y_mixture, y_instrumental, y_vocal, file_name, original_sr=44100):
    # 16kHzにダウンサンプリング
    y_mix = resample(y_mixture, original_sr, C.SAMPLING_RATE)
    y_inst = resample(y_instrumental, original_sr, C.SAMPLING_RATE)
    # y_vocal = resample(y_vocal, original_sr, C.SAMPLING_RATE)
    
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
             mix=mix_spec, inst=inst_spec, vocal=vocal_spec)
    print('Saving: ' + C.PATH_FFT + file_name + '.npz')

# データセット(.npz形式)を読み込む関数
def load_dataset(path, target='vocal'):
    X_list = []
    y_list = []
    filelist_fft = find_files(path, ext='npz')[:80]
    
    for file_fft in filelist_fft:
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
    magnitude, phase = magphase(spectrum)
    return magnitude.astype(np.float32), phase

# スペクトログラムと位相情報から音源を復元して保存する(wav形式)関数
def save_audio(file_name, magnitude, phase):
    y = istft(magnitude*phase, hop_length=C.HOP_LENGTH, win_length=C.FFT_SIZE)
    write_wav(file_name, y, C.SAMPLING_RATE, norm=True)
    print('audio saved:' + file_name)

# # マスクを計算する関数
# def compute_mask(input_magnitude, epoch, path='', hard=True):
#     unet = network.UNet()
#     unet.load(epoch=epoch, path=path)
#     mask = unet(input_magnitude[np.newaxis, np.newaxis, 1:, :]).data[0, 0, :, :]
#     mask = np.vstack((np.zeros(mask.shape[1], dtype="float32"), mask))
#     if hard:
#         hard_mask = np.zeros(mask.shape, dtype="float32")
#         hard_mask[mask > 0.5] = 1
#         return hard_mask
#     else:
#         return mask

# スペクトル情報と位相情報のリストを返す関数
def magnitude_phase(spectrograms):
	magnitude_list = []
	phase_list = []
	for i in spectrograms:
		magnitude, phase = magphase(i)
		magnitude_list.append(magnitude)
		phase_list.append(phase)
	return magnitude_list, phase_list

# 学習時にランダムにサンプルを取る関数
def sampling(X_magnitude, y_magnitude):
	X = []
	y = []
	for mixture, target in zip(X_magnitude, y_magnitude):
		starts = np.random.randint(0, mixture.shape[1] - C.PATCH_LENGTH,
									 (mixture.shape[1] - C.PATCH_LENGTH) // C.SAMPLING_STRIDE)
		for start in starts:
			end = start + C.PATCH_LENGTH
			X.append(mixture[1:, start:end, np.newaxis])
			y.append(target[1:, start:end, np.newaxis])
	return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
	