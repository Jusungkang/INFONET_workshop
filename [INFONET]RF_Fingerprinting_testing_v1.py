from __future__ import print_function

import torch.nn as nn
import torch.nn.init

from torch.utils.data import Dataset, DataLoader
torch.set_printoptions(precision=10)

import timeit
import numpy as np
import torch

from Utils_JSK_workshop import Extraction_Features, awgn, CNN_workshop_SS, CNN_workshop

if torch.cuda.is_available():
    device = 'cuda'
else:
   'cpu'
if device == 'cuda':
    torch.cuda.manual_seed(7777)
else:
    torch.manual_seed(7777)

#############################################################################################################
## [시작] Phase 2. Classifier Test 부
## 목적 : 구성된 Dataset 에 대한 RF Fingerprinting model 학습 및 성능을 평가하는 단계
#############################################################################################################

##
## Dataset loader
##

##
## Reference Database number : 0
##
DB_num = 0
print('The ' + str(DB_num) + 'th database is loaded')
test_DB_raw = np.load('./Models/DB/[' + str(DB_num) + '][8C]Test_DB_raw.npy')
test_label_DB = np.load('./Models/DB/[' + str(DB_num) + '][8C]Test_label_DB.npy')

##
## Model TEST를 위한 Dataloader class 정의
##

class RF_test(Dataset):
    def __init__(self):
        self.len = test_DB_raw.shape[0]
        self.x_data_real = torch.from_numpy(np.real(test_DB_raw)).float()
        self.x_data_image = torch.from_numpy(np.imag(test_DB_raw)).float()

        self.y_data = torch.from_numpy(test_label_DB).long()

    def __getitem__(self, index):
        return self.x_data_real[index], self.x_data_image[index], self.y_data[index]

    def __len__(self):
        return self.len

##
## TEST 시작, 편의상 Average 는 1번만 수행
##
Avg_times = 1

for zzz in range(Avg_times):
    zzz = zzz + 1

    print('The ' + str(zzz) + 'th avg_times are ongoing')

    #######################################################
    ## [시작] Phase 2-1. Model Testing
    ## 목적 : 앞서 학습된 Model 의 성능을 평가하기 위함.
    #######################################################

    Extractors = Extraction_Features()

    ##
    ## Target SNR 정의 (AWGN)
    ## SNR < 1000 -> NO AWGN noise case
    ## SNR := [1000(No SNR), ... (Target SNR)]
    ##

    # SNR_range = list(range(20, 20 + 1, 5))
    SNR_range = []
    SNR_range.insert(0,1000)

    SNR_result = np.zeros((6, len(SNR_range)))

    for k in range(len(SNR_range)):
        check_point = 0

        SNR = SNR_range[k]
        # print(SNR)


        # ##
        # ## Classifier model definition.
        # ## For version 1, Define your own classifier models
        # ## Phase 1-1 에서 학습 및 저장된 model 을 불러오는 과정.
        # ##

        model_version = 'CNN'
        new_model = CNN_workshop().to(device)
        # new_model = CNN_workshop_SS().to(device)

        ##
        ## Classifier model definition.
        ## For version 1, including Ethernet connection and googlenet structure
        ## Phase 1-1 에서 학습 및 저장된 model 을 불러오는 과정.

        ##
        ## URL : https://pytorch.org/hub/pytorch_vision_googlenet/
        ##

        # model_version = 'GoogLeNet'
        # new_model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True).to(device)
        # new_model.transform_input = False
        #
        # num_out_chs = new_model.conv1.conv.out_channels
        # new_model.conv1.conv = nn.Conv2d(1, num_out_chs, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #
        # num_ftrs = new_model.fc.in_features
        # new_model.fc = nn.Linear(num_ftrs,8,bias=True)
        #
        # new_model = new_model.to(device)


        ##
        ## Training 과정에서 저장해둔 model structure 및 parameter 를, 불러오는 과정
        ##

        new_model.load_state_dict(torch.load("./Models/RF_fingerprinting/[" + str(zzz) + "][" + model_version + "]Classifier_" + str(SNR) + ".pth"))

        print("[Avg : {:>3}] [SNR : {:>3}] Load the model from *.pth files".format(zzz, SNR))

        ##
        ## Pytorch 정의에 따른 Dataset loader
        ##
        dataset_test = RF_test()
        test_loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=True, num_workers=0)

        with torch.no_grad():
            new_model.eval()

            avg_acc = 0
            avg_test_time = 0
            avg_Feature_time = 0
            avg_STFT_time = 0

            total_batch_test = len(test_loader)

            Num_class = 8
            Conf_mat = np.zeros((Num_class,Num_class))

            for i, data in enumerate(test_loader):

                inputs_raw_real, imputs_raw_image, labels = data

                if SNR < 1000:
                    inputs_raw_real, imputs_raw_image = awgn(inputs_raw_real.numpy(), imputs_raw_image.numpy(), SNR)
                    inputs_sig_test = inputs_raw_real + 1j * imputs_raw_image
                else:
                    inputs_sig_test = inputs_raw_real.numpy() + 1j * imputs_raw_image.numpy()

                _ , features_spectrum, features_time, features_spectrum_time = Extractors.RTextraction(inputs_sig_test)

                # _ , features_spectrum, features_time, features_spectrum_time = Extractors.SSextraction(inputs_sig_test)

                tmp = np.reshape(features_spectrum,
                                 (
                                     features_spectrum.shape[0], 1, features_spectrum.shape[1], features_spectrum.shape[2]))
                features_spectrum = torch.from_numpy(tmp).float()

                inputs = features_spectrum.to(device)

                labels = labels.to(device)

                ##
                ## Tensor 구조의 TEST 데이터를 Model 에 입력하는 부분.
                ##
                testing_st = timeit.default_timer()
                prediction_Ts = new_model(inputs)

                testing_fi = timeit.default_timer()
                test_time = -1*(testing_st-testing_fi)

                ##
                ## Output 결과를 Labels와 비교하는 부분
                ##
                correct_prediction_Ts = torch.argmax(prediction_Ts, 1) == labels
                accuracy_Ts = correct_prediction_Ts.float().mean()

                ##
                ## Sample test 및 Top 3 prediction 결과 확인을 위함.
                ##
                ex_acc, ex_label  = torch.sort(torch.softmax(prediction_Ts, 1)[0, :], descending=True)

                print('                                            ')
                print('Prediction Test examples [Top-3 predictions]')
                print('The model predict the test inputs as : ')
                print('1st predictions : [Device ID : {:>=3}] with confidence score [{:>=3}]'.format(ex_label[0] + 1, ex_acc[0]))
                print('2nd predictions : [Device ID : {:>=3}] with confidence score [{:>=3}]'.format(ex_label[1] + 1, ex_acc[1]))
                print('3rd predictions : [Device ID : {:>=3}] with confidence score [{:>=3}]'.format(ex_label[2] + 1, ex_acc[2]))
                print('                                            ')

                avg_test_time += test_time
                avg_acc += accuracy_Ts
                avg_Feature_time += features_time
                avg_STFT_time += features_spectrum_time

                for jj in range(len(labels)):
                    Act_val = labels[jj]
                    Pre_val = torch.argmax(prediction_Ts, 1).cpu().numpy()[jj]
                    Conf_mat[Act_val,Pre_val] += 1

                np.savetxt("./Models/RF_fingerprinting/[" + str(zzz) + "][" + model_version + "]Classifier_" + str(SNR) + "_conf_mat.csv", Conf_mat, delimiter=",")

            print("[Avg : {:>3}] Test for [SNR : {:>3}] is finished".format(zzz, SNR))

            avg_acc = avg_acc / total_batch_test
            avg_test_time = avg_test_time / total_batch_test
            avg_Feature_time = avg_Feature_time / dataset_test.len
            avg_STFT_time = avg_STFT_time / dataset_test.len

            # print('Accuracy_Ts:', avg_acc.item())
            tmp_val = avg_acc.item()
            print('[Avg : {:>=3}] [SNR : {:>=3}] Accuracy_Ts: {:>=4}'.format(zzz, SNR, tmp_val))
            print('[Avg : {:>=3}] [SNR : {:>=3}] Avg_Feature_time: {:>=4}'.format(zzz, SNR, avg_Feature_time))
            print('[Avg : {:>=3}] [SNR : {:>=3}] Avg_STFT_time: {:>=4}'.format(zzz, SNR, avg_STFT_time))
            print('[Avg : {:>=3}] [SNR : {:>=3}] Avg_test_time : {:>=4}'.format(zzz, SNR, avg_test_time))
            print('[Avg : {:>=3}] Testing_finished'.format(zzz))

        SNR_result[0, k] = SNR
        SNR_result[1, k] = avg_acc
        SNR_result[2, k] = avg_Feature_time
        SNR_result[3, k] = avg_STFT_time
        SNR_result[4, k] = avg_test_time

        np.savetxt('./Models/RF_fingerprinting/[' + str(zzz) + '][' + model_version + ']Testing_result.csv', SNR_result)

        #######################################################
        ## [종료] Phase 2-1. (Offline) Testing procedure finished
        #######################################################



