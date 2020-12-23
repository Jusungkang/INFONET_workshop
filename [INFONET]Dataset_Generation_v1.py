from sklearn.utils import shuffle
import numpy as np
import torch.nn.init
torch.set_printoptions(precision=10)
import os
import scipy.io as sio  # .mat 파일 불러오는 방법

# Data
train = []
train_raw = []
test = []
test_raw = []
validation = []
validation_raw = []
train_label = []
test_label = []
validation_label = []

MK_DB_num = 1

for kk in range(MK_DB_num):
    i = kk + 1

    print('The ' + str(i) + 'th dataset generation task are ongoiing')

    #############################################################################################################
    ## [시작] Phase 0. Dataset Generation
    ## 목적 : Radio signal 을 비율에 맞춰 Training, Validation 및 Test dataset 으로 나누는 목적.
    #############################################################################################################

    ##
    ## Loading the samples from '*.mat' sample files
    ##

    MatFileName = os.path.join('./DB/', 'Wakie_Talkie_Dataset_3')
    Database = sio.loadmat(MatFileName)
    data_class_1_raw = Database.get('H_DV1')
    data_class_2_raw = Database.get('H_DV2')
    data_class_3_raw = Database.get('H_DV3')
    data_class_4_raw = Database.get('H_DV4')
    data_class_5_raw = Database.get('M_DV1')
    data_class_6_raw = Database.get('M_DV2')
    data_class_7_raw = Database.get('M_DV3')
    data_class_8_raw = Database.get('M_DV4')

    ##
    ## For shuffling the samples within datset
    ##

    ##
    ## For Class 1
    ##
    data_class_1_raw = data_class_1_raw
    data_class_1_raw_L = list(data_class_1_raw)
    data_class_1_raw_L = shuffle(data_class_1_raw_L)
    data_class_1_raw = np.array(data_class_1_raw_L)

    ##
    ## For Class 2
    ##
    data_class_2_raw = data_class_2_raw
    data_class_2_raw_L = list(data_class_2_raw)
    data_class_2_raw_L = shuffle(data_class_2_raw_L)
    data_class_2_raw = np.array(data_class_2_raw_L)

    ##
    ## For Class 3
    ##
    data_class_3_raw = data_class_3_raw
    data_class_3_raw_L = list(data_class_3_raw)
    data_class_3_raw_L = shuffle(data_class_3_raw_L)
    data_class_3_raw = np.array(data_class_3_raw_L)

    ##
    ## For Class 4
    ##
    data_class_4_raw = data_class_4_raw
    data_class_4_raw_L = list(data_class_4_raw)
    data_class_4_raw_L = shuffle(data_class_4_raw_L)
    data_class_4_raw = np.array(data_class_4_raw_L)

    ##
    ## For Class 5
    ##
    data_class_5_raw = data_class_5_raw
    data_class_5_raw_L = list(data_class_5_raw)
    data_class_5_raw_L = shuffle(data_class_5_raw_L)
    data_class_5_raw = np.array(data_class_5_raw_L)

    ##
    ## For Class 6
    ##
    data_class_6_raw = data_class_6_raw
    data_class_6_raw_L = list(data_class_6_raw)
    data_class_6_raw_L = shuffle(data_class_6_raw_L)
    data_class_6_raw = np.array(data_class_6_raw_L)

    ##
    ## For Class 7
    ##
    data_class_7_raw = data_class_7_raw
    data_class_7_raw_L = list(data_class_7_raw)
    data_class_7_raw_L = shuffle(data_class_7_raw_L)
    data_class_7_raw = np.array(data_class_7_raw_L)

    ##
    ## For Class 8
    ##
    data_class_8_raw = data_class_8_raw
    data_class_8_raw_L = list(data_class_8_raw)
    data_class_8_raw_L = shuffle(data_class_8_raw_L)
    data_class_8_raw = np.array(data_class_8_raw_L)

    ##
    ## For dividing the samples to 'Training', 'Validation', and 'Testing' datasets
    ##

    ##
    ## For Class 1
    ##
    sample_len= data_class_1_raw.shape[0]
    for ii in range(sample_len):
        temp_raw = data_class_1_raw[ii, :]
        if ii < 0.7*sample_len:
            train_raw.append(temp_raw)
            train_label.append(0)
        else:
            test_raw.append(temp_raw)
            test_label.append(0)

    ##
    ## For Class 2
    ##
    sample_len= data_class_2_raw.shape[0]
    for ii in range(sample_len):
        temp_raw = data_class_2_raw[ii, :]
        if ii < 0.7*sample_len:
            train_raw.append(temp_raw)
            train_label.append(1)
        else:
            test_raw.append(temp_raw)
            test_label.append(1)

    ##
    ## For Class 3
    ##
    sample_len= data_class_3_raw.shape[0]
    for ii in range(sample_len):
        temp_raw = data_class_3_raw[ii, :]
        if ii < 0.7*sample_len:
            train_raw.append(temp_raw)
            train_label.append(2)
        else:
            test_raw.append(temp_raw)
            test_label.append(2)

    ##
    ## For Class 4
    ##
    sample_len= data_class_4_raw.shape[0]
    for ii in range(sample_len):
        temp_raw = data_class_4_raw[ii, :]
        if ii < 0.7*sample_len:
            train_raw.append(temp_raw)
            train_label.append(3)
        else:
            test_raw.append(temp_raw)
            test_label.append(3)

    ##
    ## For Class 5
    ##
    sample_len = data_class_5_raw.shape[0]
    for ii in range(sample_len):
        temp_raw = data_class_5_raw[ii, :]
        if ii < 0.7 * sample_len:
            train_raw.append(temp_raw)
            train_label.append(4)
        else:
            test_raw.append(temp_raw)
            test_label.append(4)

    ##
    ## For Class 6
    ##
    sample_len = data_class_6_raw.shape[0]
    for ii in range(sample_len):
        temp_raw = data_class_6_raw[ii, :]
        if ii < 0.7 * sample_len:
            train_raw.append(temp_raw)
            train_label.append(5)
        else:
            test_raw.append(temp_raw)
            test_label.append(5)

    ##
    ## For Class 7
    ##
    sample_len = data_class_7_raw.shape[0]
    for ii in range(sample_len):
        temp_raw = data_class_7_raw[ii, :]
        if ii < 0.7 * sample_len:
            train_raw.append(temp_raw)
            train_label.append(6)
        else:
            test_raw.append(temp_raw)
            test_label.append(6)

    ##
    ## For Class 8
    ##
    sample_len = data_class_8_raw.shape[0]
    for ii in range(sample_len):
        temp_raw = data_class_8_raw[ii, :]
        if ii < 0.7 * sample_len:
            train_raw.append(temp_raw)
            train_label.append(7)
        else:
            test_raw.append(temp_raw)
            test_label.append(7)

    ##
    ## Data structure : LIST -> NUMPY
    ##
    train_DB_raw = np.asarray(train_raw)
    train_label_DB = np.asarray(train_label)

    test_DB_raw = np.asarray(test_raw)
    test_label_DB = np.asarray(test_label)

    ##
    ## Save the datasets
    ##

    np.save('./Models/DB/[' + str(i) + '][8C]Train_DB_raw.npy', train_DB_raw)
    np.save('./Models/DB/[' + str(i) + '][8C]Train_label_DB.npy', train_label_DB)

    np.save('./Models/DB/[' + str(i) + '][8C]Test_DB_raw.npy', test_DB_raw)
    np.save('./Models/DB/[' + str(i) + '][8C]Test_label_DB.npy', test_label_DB)

    ############################################################################################################
    ## [종료] Phase 0. Dataset Generation
    ############################################################################################################
