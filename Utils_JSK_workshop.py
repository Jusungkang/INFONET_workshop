from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
torch.set_printoptions(precision=10)

import timeit
from scipy import signal
from scipy.fft import fftshift

import numpy as np
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
   'cpu'
if device == 'cuda':
    torch.cuda.manual_seed(7777)
else:
    torch.manual_seed(7777)

class Extraction_Features:
    def __init__(self):
        self.Fs = 96e6
        self.Deci_f = 20
        self.Fss = self.Fs / self.Deci_f
        self.FreqL = 500e3
        self.thr = 0.1

    def RTextraction(self, sig, Flag='Test'):
        if sig.shape[0] >= 10e4:
            samples = 1
            Feature_size = int(np.floor(0.1 * sig.shape[0]))
            MV_size = int(np.floor(0.01 * sig.shape[0]))
        else:
            samples = sig.shape[0]
            Feature_size = int(np.floor(0.1 * sig.shape[1]))
            MV_size = int(np.floor(0.01 * sig.shape[1]))

        sig_features = np.zeros((samples, Feature_size), dtype="complex_")
        sig_features_spectrogram = []
        sig_features_time = 0;
        sig_features_transform_time = 0;

        for i in range(samples):

            if sig.shape[0] >= 10e4:
                tmp_sig = sig
            else:
                tmp_sig = sig[i, :]

            tmp_energy = []
            tmp_energy_time = []

            RTr_st = []
            RTr_fin= 0

            tmp_sig_features_time_st = timeit.default_timer()

            for j in range(0, int(np.floor(tmp_sig.shape[0]/2) - MV_size), int(np.floor(MV_size/2))):

                tmp_energy_val = np.linalg.norm(tmp_sig[j:j+MV_size+1])

                tmp_energy.append(tmp_energy_val)
                tmp_energy_time.append(j)

                tmp_col = len(tmp_energy)

                if j > 0:
                    tmp_norm = (tmp_energy[tmp_col-1]-tmp_energy[tmp_col-2])/tmp_energy[tmp_col-2]

                    if (tmp_norm >= self.thr):
                        RTr_st.append(j)
                        RTr_fin = (j+MV_size)

            if RTr_st != []:
                RTr_time_st = RTr_st[0]
            else:
                RTr_time_st = 0;

            if RTr_fin != 0:
                RTr_time_fin = RTr_fin
            else:
                RTr_time_fin = int(np.floor(tmp_sig.shape[0]/2))

            if (RTr_time_fin - RTr_time_st > 0) and (RTr_time_fin - RTr_time_st <= Feature_size) :
                sig_features[i, 0: RTr_time_fin - RTr_time_st] = tmp_sig[RTr_time_st : RTr_time_fin]
            else:
                ## Case not happen. hmm....
                print('[Breaks happens] with RTr_time_st :', RTr_time_st)
                print(
                    '[Breaks happens] with RTr_time_fin : ', RTr_time_fin)
                sig_features[i, :] = tmp_sig[RTr_time_st: RTr_time_st + Feature_size]

            tmp_sig_features_time_fi = timeit.default_timer()
            tmp_sig_features_time = -1 * (tmp_sig_features_time_st - tmp_sig_features_time_fi)
            sig_features_time += tmp_sig_features_time

            ##
            ## Feature extraction is endeed
            ##

            tmp_sig_2 = sig_features[i, :]

            tmp_sig_features_transform_time_st = timeit.default_timer()

            Sxx_feature_DB = self.STFT(tmp_sig_2)

            sig_features_spectrogram.append(Sxx_feature_DB)

            tmp_sig_features_transform_time_fi = timeit.default_timer()
            tmp_sig_features_transform_time = -1 * (
                    tmp_sig_features_transform_time_st - tmp_sig_features_transform_time_fi)
            sig_features_transform_time += tmp_sig_features_transform_time

        sig_features_spectrogram = np.asarray(sig_features_spectrogram)

        return sig_features, sig_features_spectrogram, sig_features_time, sig_features_transform_time

    def FTextraction(self, sig, Flag='Test'):
        if sig.shape[0] >= 10e4:
            samples = 1
            Feature_size = int(np.floor(0.1 * sig.shape[0]))
            MV_size = int(np.floor(0.01 * sig.shape[0]))
        else:
            samples = sig.shape[0]
            Feature_size = int(np.floor(0.1 * sig.shape[1]))
            MV_size = int(np.floor(0.01 * sig.shape[1]))

        sig_features = np.zeros((samples, Feature_size), dtype="complex_")
        sig_features_spectrogram = []
        sig_features_time = 0;
        sig_features_transform_time = 0;

        for i in range(samples):
            # print(i)

            if sig.shape[0] >= 10e4:
                tmp_sig = sig
            else:
                tmp_sig = sig[i, :]


            tmp_energy = []
            tmp_energy_time = []

            # RTr_fin = 0

            FTr_st = []
            FTr_fin = 0

            tmp_sig_features_time_st = timeit.default_timer()

            for j in range(0, int(tmp_sig.shape[0]-MV_size), int(np.floor(MV_size / 2))):

                tmp_energy_val = np.linalg.norm(tmp_sig[j:j + MV_size + 1])

                tmp_energy.append(tmp_energy_val)
                tmp_energy_time.append(j)

                tmp_col = len(tmp_energy)

                if j > 0:
                    tmp_norm = (tmp_energy[tmp_col - 1] - tmp_energy[tmp_col - 2]) / tmp_energy[tmp_col - 2]

                    if (tmp_norm <= -1 * self.thr) and (j >= tmp_sig.shape[0]/2):
                        FTr_st.append(j)
                        FTr_fin = (j + MV_size)


            if FTr_st != []:
                FTr_st_time = FTr_st[0]
            else:
                FTr_st_time = int(np.floor(tmp_sig.shape[0]/2))

            if FTr_fin != 0:
                FTr_fin_time = FTr_fin
            else:
                FTr_fin_time = int(tmp_sig.shape[0])

            if (FTr_fin_time - FTr_st_time > 0) and (FTr_fin_time - FTr_st_time <= Feature_size):
                sig_features[i, 0: FTr_fin_time - FTr_st_time] = tmp_sig[FTr_st_time: FTr_fin_time]
            else:

                ## Case not happen. hmm....
                print('[Breaks happens] with FT_time_st :', FTr_st_time)
                print(
                    '[Breaks happens] with FT_time_fin : ', FTr_fin_time)
                sig_features[i, : ] = tmp_sig[FTr_st_time: FTr_st_time + Feature_size]

            tmp_sig_features_time_fi = timeit.default_timer()
            tmp_sig_features_time = -1 * (tmp_sig_features_time_st - tmp_sig_features_time_fi)
            sig_features_time += tmp_sig_features_time

            tmp_sig_2 = sig_features[i, :]

            tmp_sig_features_transform_time_st = timeit.default_timer()

            Sxx_feature_DB = self.STFT(tmp_sig_2)

            sig_features_spectrogram.append(Sxx_feature_DB)

            tmp_sig_features_transform_time_fi = timeit.default_timer()
            tmp_sig_features_transform_time = -1 * (
                    tmp_sig_features_transform_time_st - tmp_sig_features_transform_time_fi)
            sig_features_transform_time += tmp_sig_features_transform_time

        sig_features_spectrogram = np.asarray(sig_features_spectrogram)

        return sig_features, sig_features_spectrogram, sig_features_time, sig_features_transform_time

    def SSextraction(self, sig, Flag='Test'):
        if sig.shape[0] >= 10e4:
            samples = 1
            Feature_size = int(np.floor(0.95 * sig.shape[0]))
            MV_size = int(np.floor(0.01 * sig.shape[0]))
        else:
            samples = sig.shape[0]
            Feature_size = int(np.floor(0.95 * sig.shape[1]))
            MV_size = int(np.floor(0.01 * sig.shape[1]))

        sig_features = np.zeros((samples, Feature_size), dtype="complex_")
        sig_features_spectrogram = []
        sig_features_time = 0;
        sig_features_transform_time = 0;

        for i in range(samples):
            if sig.shape[0] >= 10e4:
                tmp_sig = sig
            else:
                tmp_sig = sig[i, :]

            tmp_energy = []
            tmp_energy_time = []

            RTr_fin = 0
            FTr_st = []

            tmp_sig_features_time_st = timeit.default_timer()

            for j in range(0, int(tmp_sig.shape[0] - MV_size), int(np.floor(MV_size / 2))):

                tmp_energy_val = np.linalg.norm(tmp_sig[j:j + MV_size + 1])

                tmp_energy.append(tmp_energy_val)
                tmp_energy_time.append(j)

                tmp_col = len(tmp_energy)

                if j > 0:
                    tmp_norm = (tmp_energy[tmp_col - 1] - tmp_energy[tmp_col - 2]) / tmp_energy[tmp_col - 2]
                    if (tmp_norm >= self.thr) and (j <= tmp_sig.shape[0] / 2):
                        RTr_fin = (j + MV_size)
                    elif (tmp_norm <= -1 * self.thr) and (j >= tmp_sig.shape[0] / 2):
                        FTr_st.append(j)

            if RTr_fin != 0:
                RTr_time_fin = RTr_fin
            else:
                RTr_time_fin = int(np.floor(tmp_sig.shape[0]/2))

            if FTr_st != []:
                FTr_st_time = FTr_st[0]
            else:
                FTr_st_time = int(np.floor(tmp_sig.shape[0]/2))

            if (FTr_st_time - RTr_time_fin > 0) and (FTr_st_time - RTr_time_fin <= Feature_size):
                sig_features[i, 0: FTr_st_time - RTr_time_fin] = tmp_sig[RTr_time_fin: FTr_st_time]
            else:
                ## Case not happen. hmm....
                print('[Breaks happens] with SS_time_st :', RTr_time_fin)
                print(
                    '[Breaks happens] with SS_time_fin : ', FTr_st_time)
                if RTr_time_fin + Feature_size > tmp_sig.shape[0] :
                    sig_features[i, 0: tmp_sig.shape[0]-RTr_time_fin] = tmp_sig[RTr_time_fin: tmp_sig.shape[0]]
                else:
                    sig_features[i, :] = tmp_sig[RTr_time_fin: RTr_time_fin + Feature_size]

            tmp_sig_features_time_fi = timeit.default_timer()
            tmp_sig_features_time = -1 * (tmp_sig_features_time_st - tmp_sig_features_time_fi)
            sig_features_time += tmp_sig_features_time

            tmp_sig_2 = sig_features[i, :]

            tmp_sig_features_transform_time_st = timeit.default_timer()

            Sxx_feature_DB = self.STFT(tmp_sig_2)

            sig_features_spectrogram.append(Sxx_feature_DB)

            tmp_sig_features_transform_time_fi = timeit.default_timer()
            tmp_sig_features_transform_time = -1 * (
                    tmp_sig_features_transform_time_st - tmp_sig_features_transform_time_fi)
            sig_features_transform_time += tmp_sig_features_transform_time

        sig_features_spectrogram = np.asarray(sig_features_spectrogram)

        return sig_features, sig_features_spectrogram, sig_features_time, sig_features_transform_time

    def Hopextraction(self, sig, Flag='Test'):
        if sig.shape[0] >= 10e4:
            samples = 1
            Feature_size = int(np.floor(1.0 * sig.shape[0]))
            MV_size = int(np.floor(0.01 * sig.shape[0]))
        else:
            samples = sig.shape[0]
            Feature_size = int(np.floor(1.0 * sig.shape[1]))
            MV_size = int(np.floor(0.01 * sig.shape[1]))

        sig_features = np.zeros((samples, Feature_size), dtype="complex_")
        sig_features_spectrogram = []
        sig_features_time = 0;
        sig_features_transform_time = 0;

        for i in range(samples):
            # print(i)
            if sig.shape[0] >= 10e4:
                tmp_sig = sig
            else:
                tmp_sig = sig[i, :]

            tmp_sig_features_time_st = timeit.default_timer()

            ##
            ## None of process are needed for hop extraction.
            ##

            tmp_sig_features_time_fi = timeit.default_timer()
            tmp_sig_features_time = -1 * (tmp_sig_features_time_st - tmp_sig_features_time_fi)
            sig_features_time += tmp_sig_features_time

            tmp_sig_features_transform_time_st = timeit.default_timer()

            Sxx_feature_DB = self.STFT(tmp_sig)

            sig_features_spectrogram.append(Sxx_feature_DB)

            tmp_sig_features_transform_time_fi = timeit.default_timer()
            tmp_sig_features_transform_time = -1 * (
                    tmp_sig_features_transform_time_st - tmp_sig_features_transform_time_fi)
            sig_features_transform_time += tmp_sig_features_transform_time

        sig_features_spectrogram = np.asarray(sig_features_spectrogram)

        return sig_features, sig_features_spectrogram, sig_features_time, sig_features_transform_time

    def STFT(self, sig, Flag='Test'):
        f, t, Sxx = signal.spectrogram(sig, self.Fss, return_onesided=False,
                                       window=signal.get_window('boxcar', 1024),
                                       nperseg=1024, noverlap=512, nfft=1024 * 4,
                                       scaling='spectrum', mode='magnitude')

        f_feature = f[np.where(np.abs(f) <= self.FreqL)]
        Sxx_feature = Sxx[np.where(np.abs(f) <= self.FreqL)]

        # Sxx_feature_DB = 10 * np.log10(Sxx_feature)
        # Sxx_feature_DB[np.isinf(Sxx_feature_DB)] = -100
        # Sxx_feature_DB = Sxx_feature_DB + 100

        Sxx_feature_DB = fftshift(Sxx_feature, axes = 0)

        tmp_mean = np.mean(Sxx_feature_DB)
        tmp_std = np.std(Sxx_feature_DB)
        Sxx_feature_DB = (Sxx_feature_DB - tmp_mean) / tmp_std

        # plt.figure()
        # plt.pcolormesh(t, f_feature, np.abs(Sxx_feature))
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.ylim(ymax=500e3, ymin = -500e3)

        return Sxx_feature_DB

def awgn(sig_real, sig_image, target_SNR):

    sig_real_SNR = np.zeros((sig_real.shape[0],sig_real.shape[1]))
    sig_image_SNR = np.zeros((sig_image.shape[0],sig_image.shape[1]))

    for i in range(sig_real_SNR.shape[0]):

        P_sig_r = np.linalg.norm(sig_real[i,:],2) ** 2
        P_noise_r = P_sig_r / (10 ** (target_SNR/10))
        tmp_noise_r = np.random.normal(0,1,sig_real.shape[1]) * np.sqrt(P_noise_r/sig_real.shape[1])

        P_sig_i = np.linalg.norm(sig_image[i,:], 2) ** 2
        P_noise_i = P_sig_i / (10 ** (target_SNR / 10))
        tmp_noise_i = np.random.normal(0, 1, sig_image.shape[1]) * np.sqrt(P_noise_i / sig_image.shape[1])

        sig_real_SNR[i,:] = sig_real[i,:] + tmp_noise_r
        sig_image_SNR[i,:] = sig_image[i,:] + tmp_noise_i

    return sig_real_SNR , sig_image_SNR

##
##  Define your OWN clasifier models
##

class CNN_workshop(nn.Module):
    def __init__(self):
        super(CNN_workshop, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.fc1 = torch.nn.Linear(32*14*1, 256, bias=False)
        nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
        )

        self.fc2 = torch.nn.Linear(256, 128, bias=False)
        nn.init.xavier_uniform(self.fc2.weight)
        self.layer5 = nn.Sequential(
            self.fc2,
            nn.ReLU(),
        )
        self.fc3 = torch.nn.Linear(128, 8, bias=False)
        nn.init.xavier_uniform(self.fc3.weight)

# 10월 14일 수정
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # print(out.shape)
        out = out.view(out.size(0),-1)
        # print(out.shape)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.fc3(out)
        return out



class CNN_workshop_SS(nn.Module):
    def __init__(self):
        super(CNN_workshop_SS, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.fc1 = torch.nn.Linear(32*14*5, 256, bias=False)
        nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
        )

        self.fc2 = torch.nn.Linear(256, 128, bias=False)
        nn.init.xavier_uniform(self.fc2.weight)
        self.layer5 = nn.Sequential(
            self.fc2,
            nn.ReLU(),
        )
        self.fc3 = torch.nn.Linear(128, 8, bias=False)
        nn.init.xavier_uniform(self.fc3.weight)

# 10월 14일 수정
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # print(out.shape)
        out = out.view(out.size(0),-1)
        # print(out.shape)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.fc3(out)
        return out