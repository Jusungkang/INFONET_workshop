from __future__ import print_function

import torch.nn as nn
import torch.optim as opt
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

# parameter
learning_rate_init = 0.001
training_epoch = 100

#############################################################################################################
## [시작] Phase 1. Classifier 학습부
## 목적 : 앞서 구성된 Radio signal dataset 에 대한 Classifier model 학습 단계
#############################################################################################################

##
## Dataset loader
##

##
## Reference Database number : 0
##
DB_num = 0
print('The ' + str(DB_num) + 'th database is loaded')
train_DB_raw = np.load('./Models/DB/[' + str(DB_num) + '][8C]Train_DB_raw.npy')
train_label_DB = np.load('./Models/DB/[' + str(DB_num) + '][8C]Train_label_DB.npy')

##
## for class unvalanced problem.
##
Total_index = np.asarray([range(0,int(50*0.7*8),1)])
Val_index = np.asarray([range(0,int(50*0.7*8),7)])
Val_index = Val_index.reshape((Val_index.shape[1]))
Train_index = np.delete(Total_index,Val_index)
# Rand_index = np.random.randint(0,len(train_DB_raw),int(len(train_DB_raw)/7*1))


validation_DB_raw = train_DB_raw[Val_index]
validation_label_DB = train_label_DB[Val_index]

train_DB_raw = train_DB_raw[Train_index]
train_label_DB = train_label_DB[Train_index]

##
## Model 학습을 위한 Dataloader class 정의
##
class RF_train(Dataset):
    def __init__(self):
        self.len = train_DB_raw.shape[0]
        # self.x_data = train_DB_raw
        self.x_data_real = torch.from_numpy(np.real(train_DB_raw)).float()
        self.x_data_image = torch.from_numpy(np.imag(train_DB_raw)).float()

        self.y_data = torch.from_numpy(train_label_DB).long()

    def __getitem__(self, index):
        # return self.x_data[index], self.y_data[index]
        return self.x_data_real[index], self.x_data_image[index], self.y_data[index]

    def __len__(self):
        return self.len

class RF_val(Dataset):
    def __init__(self):
        self.len = validation_DB_raw.shape[0]
        # self.x_data = train_DB_raw
        self.x_data_real = torch.from_numpy(np.real(validation_DB_raw)).float()
        self.x_data_image = torch.from_numpy(np.imag(validation_DB_raw)).float()

        self.y_data = torch.from_numpy(validation_label_DB).long()

    def __getitem__(self, index):
        # return self.x_data[index], self.y_data[index]
        return self.x_data_real[index], self.x_data_image[index], self.y_data[index]

    def __len__(self):
        return self.len

##
## 학습 시작, 편의상 Average 는 1번만 수행
##

Avg_times = 1

for zzz in range(Avg_times):
    zzz = zzz + 1

    print('The ' + str(zzz) + 'th avg_times are ongoing')

    #######################################################
    ## [시작] Phase 1-1. Model training
    ## 목적 : 기 구성한 Datset 을 기반으로, Classifier model 을 학습하는 과정
    #######################################################

    ##
    ## 필요한 function call
    ##

    Extractors = Extraction_Features()

    dataset = RF_train()
    dataset_val = RF_val()

    ##
    ## Target SNR 정의 (AWGN)
    ## SNR < 1000 -> NO AWGN noise case
    ## SNR := [1000(No SNR), ... (Target SNR)]
    ##

    # SNR_range = list(range(20, 20 + 1, 5))
    SNR_range = []
    SNR_range.insert(0,1000)

    SNR_result = np.zeros((4, len(SNR_range)))

    for k in range(len(SNR_range)):
        check_point = 0

        SNR = SNR_range[k]

        ##
        ## Pytorch 정의에 따른 Dataset loader
        ##
        
        train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset=dataset_val, batch_size=32, shuffle=True, num_workers=0)

        total_batch = len(train_loader)
        total_batch_val = len(val_loader)

        ##
        ## Classifier model definition.
        ## For version 1, Define your own classifier models
        ##

        model_version = 'CNN'
        model = CNN_workshop().to(device)
        # model = CNN_workshop_SS().to(device)

        ##
        ## Classifier model definition.
        ## For version 1, including Ethernet connection and googlenet structure
        ##

        ##
        ## URL : https://pytorch.org/hub/pytorch_vision_googlenet/
        ##

        # model_version = 'GoogLeNet'
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True).to(device)
        # model.transform_input = False
        #
        # num_out_chs = model.conv1.conv.out_channels
        # model.conv1.conv = nn.Conv2d(1, num_out_chs, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs,8,bias=True)
        #
        # model = model.to(device)

        ##
        ## Define your Loss functionwith optimizer & scheduler 
        ##

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = opt.Adam(model.parameters(), lr=learning_rate_init)
        scheduler = opt.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        train_cost = []
        val_cost = []

        avg_train_time = 0

        ##
        ## Model training 시작 
        ##

        for epoch in range(training_epoch):

            ##
            ## Model 을 Training mode 로 정의
            ## Training mode 에서는, Gradient 및 Backward propagation 까지 기억하여, Weight update 가 이뤄짐. 
            ## 반대로 model.eval() 명령어는, feedforward propoagation value 만 기억하는 Test(evaluation) mode 임.
            ## Coding 시 자주 실수하는 부분. 
            ##
            
            model.train()
            avg_cost = 0

            training_st = timeit.default_timer()

            for i, data in enumerate(train_loader):
                inputs_raw_real, imputs_raw_image , labels = data

                if SNR < 1000:
                    inputs_raw_real, imputs_raw_image = awgn(inputs_raw_real.numpy(), imputs_raw_image.numpy(), SNR)
                    inputs_sig = inputs_raw_real + 1j * imputs_raw_image
                else:
                    inputs_sig = inputs_raw_real.numpy() + 1j * imputs_raw_image.numpy()
                                
                features_sig, features_spectrum, _ , _ = Extractors.RTextraction(inputs_sig)

                # features_sig, features_spectrum, _ , _ = Extractors.SSextraction(inputs_sig)

                tmp = np.reshape(features_spectrum,
                                 (features_spectrum.shape[0], 1, features_spectrum.shape[1], features_spectrum.shape[2]))
                features_spectrum = torch.from_numpy(tmp).float()

                inputs = features_spectrum.to(device)

                labels = labels.to(device)

                ##
                ## 기존에 남아있을 수 있는 혹시 모를 Gradient 를 zero 로 Initialization 시켜주고
                ## Tensor 구조의 입력 데이터를 Model 에 입력하는 부분.
                ##
                optimizer.zero_grad()
                hypothesis = model(inputs)

                ##
                ## 정의한 Loss 에 따라 Cost 계산
                ##
                cost = criterion(hypothesis, labels)

                ##
                ## 계산한 backward gradient 값을 'Update' 하는 부분
                ##
                cost.backward()
                optimizer.step()

                avg_cost += cost

            avg_cost = avg_cost / total_batch
            train_cost.append(avg_cost)

            training_fi = timeit.default_timer()
            tmp_train_time = -1 * (training_st - training_fi)

            avg_train_time += tmp_train_time

            avg_cost_val = 0

            ##
            ## Validation test 시작
            ##

            with torch.no_grad():
                model.eval()
                for i, data in enumerate(val_loader):

                    inputs_raw_real, imputs_raw_image , labels = data

                    if SNR < 1000:
                        inputs_raw_real, imputs_raw_image = awgn(inputs_raw_real.numpy(), imputs_raw_image.numpy(), SNR)
                        inputs_sig_val = inputs_raw_real + 1j * imputs_raw_image
                    else:
                        inputs_sig_val = inputs_raw_real.numpy() + 1j * imputs_raw_image.numpy()

                    # features_sig, features_spectrum, _ , _ = Extractors.RTextraction(inputs_sig_val)

                    features_sig, features_spectrum, _ , _ = Extractors.SSextraction(inputs_sig_val)

                    tmp = np.reshape(features_spectrum,
                                     (
                                     features_spectrum.shape[0], 1, features_spectrum.shape[1], features_spectrum.shape[2]))
                    features_spectrum = torch.from_numpy(tmp).float()

                    inputs = features_spectrum.to(device)

                    labels = labels.to(device)

                    optimizer.zero_grad()
                    hypothesis = model(inputs)

                    cost = criterion(hypothesis, labels)

                    avg_cost_val += cost

                avg_cost_val = avg_cost_val / total_batch_val
                val_cost.append(avg_cost_val)

            print(
                '[Avg : {:>=2}][SNR : {:>=4}] (Epoch: {:>3}) Train cost = {:>.6} Val cost = {:>.6} with learning rate = {:>.4}'.format(
                    zzz , SNR, epoch + 1, avg_cost,
                    avg_cost_val, optimizer.param_groups[0]['lr']))

            ##
            ## Validation test 결과에 따라, 가장 낮은 Validation cost 를 보이는 Model 을 저장하는 내용.
            ##

            if (np.min(val_cost) == avg_cost_val) == True:
                # save
                check_point = epoch + 1
                savePath = "./Models/RF_fingerprinting/[" + str(zzz) + "][" + model_version + "][VAL]Classifier_" + str(SNR) + ".pth"
                torch.save(model.state_dict(), savePath)

            scheduler.step()

            if avg_cost_val <= 0.1:
                break


        train_time = avg_train_time

        print('[Avg : {:>=3}] Train time : {:>5}'.format(zzz, train_time))

        print("[Avg : {:>3}] Learning for [SNR : {:>3}] is finished".format(zzz, SNR))

        SNR_result[0, k] = SNR
        SNR_result[1, k] = train_time
        SNR_result[2, k] = avg_cost
        SNR_result[3, k] = avg_cost_val

        np.savetxt('./Models/RF_fingerprinting/[' + str(zzz) + '][' + model_version + '][VAL]Training_result.csv', SNR_result)

        #######################################################
        ## [종료] Phase 1-1. Model training is finished
        #######################################################



