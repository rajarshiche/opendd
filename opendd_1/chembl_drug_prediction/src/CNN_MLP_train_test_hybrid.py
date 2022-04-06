""" This file contains training functions to predict inhibition constant (Ki) of drugs to a particular protein (Coagulation factor X, chembl id: CHEMBL244) """
###------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import the necessary packages
from imutils import paths
import argparse
import random
import shutil
import os
from os.path import exists
import cv2
import numpy as np
import pandas as pd
import scipy


from chembl_drug_prediction.preprocessing.image_and_table_processing import load_table
from chembl_drug_prediction.preprocessing.image_and_table_processing import image_data_extract
from chembl_drug_prediction.preprocessing.image_and_table_processing import preprocess_img, preprocess_all_imgs

from chembl_drug_prediction.models.KiNet_mlp import KiNet_mlp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import base64
import datetime
import io
import logging
import random
from pathlib import Path
import tensorflow as tf


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

tf.get_logger().setLevel(logging.ERROR)
LOG_DIR = datetime.datetime.now().strftime("rev1_%Y%m%d_%H%M%S")


def CNN_MLP_train_test(lr, decay, bs, epochs, momentum, dropout, data_path):
    
    ### Defining the BS/ epoch_number values as int is important in order to avoid tensorflow issues
    BS = int(bs)
    epoch_number= int(epochs)

    # grab the image dataset path and preprocess the images in parallel
    mol_image_path = Path("chembl_drug_prediction/input_data/target_images").absolute()
    images = preprocess_all_imgs(mol_image_path)
    
    ### grab the image dataset path and preprocess the images one by one
    # basePath = os.path.join("chembl_drug_prediction/input_data/target_images")
    
    # ## This command will extract out all the image paths in dataset
    # imagePaths = list(paths.list_images(basePath))
    
    # # randomly sample the image paths to shuffle
    # random.seed(42)
    # random.shuffle(imagePaths)
    # images = image_data_extract(imagePaths)
    
    ## Getting the list of molecules from 
    df = pd.read_csv('chembl_drug_prediction/input_data/target_descriptors.csv', encoding='cp1252') 
    df_target = df['pChEMBL Value'].values ## This quantity is actually LOG10(pChEMBL Value)
    
    ## Getting the max & min of the target column
    maxPrice = df.iloc[:,-1].max() # grab the maximum val in the training set's last column
    minPrice = df.iloc[:,-1].min() # grab the minimum val in the training set's last column
    
    
    print("[INFO] loading images...")
    images = np.asarray(images/255.0)
    images = images.astype("float")
    
    
    print("[INFO] constructing training/ testing split")
    split = train_test_split(images, df, test_size = 0.2, random_state = 42) 
    
    # # Distribute the image values & descriptor values between train & test X between numerical and image sources
    (XImagetrain, XImagetest, XtrainTotalData, XtestTotalData) = split  # split format always is in (data_train, data_test, label_train, label_test)
     
    ## Normalizing the test or Labels
    XtrainLabels = (XtrainTotalData.iloc[:,-1])/ (maxPrice)
    XtestLabels = (XtestTotalData.iloc[:,-1])/ (maxPrice)  
    
    ## Getting the data columns except the last column which is the target column
    XtrainData = (XtrainTotalData.iloc[:,0:-2])
    XtestData = (XtestTotalData.iloc[:,0:-2])
    
    # perform min-max scaling of each continuous feature column (data columns) in the range [0 1]
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(XtrainTotalData.iloc[:,0:XtrainTotalData.shape[1]-1])
    testContinuous = cs.transform(XtestTotalData.iloc[:,0:XtestTotalData.shape[1]-1])
    
    print("[INFO] processing input data after normalization....")
    XtrainData, XtestData = trainContinuous,testContinuous
    
    
    # # create the MLP and CNN models
    mlp = KiNet_mlp.create_mlp(XtrainData.shape[1], regress = False) # the input dimension to mlp would be shape[1] of the matrix i.e. number of column features
    cnn = KiNet_mlp.create_cnn(224, 224, 3, filters = (2,4,8), regress = False)
    
    
    ### The final input to our last layer will be concatenated output from both MLP and CNN lyers
    combinedInput = concatenate([mlp.output, cnn.output])
    
    ### Final FC layers towards regression
    x = Dense(100, activation = "relu") (combinedInput)
    x = Dense(10, activation = "relu") (combinedInput)
    x = Dense(1, activation = "linear") (x)
    
    
    ### FINAL MODEL: taking into account feature data from MLP and images from CNN
    model = Model(inputs = [mlp.input, cnn.input], outputs = x)
    
    
    ## initialize the optimizer model
    print("[INFO] compiling model...")
    ### opt = SGD(lr= 1.05e-6, decay = 1.05e-6/200)
    opt = SGD(lr= lr, decay = decay)
    model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)
    
    ## train the network
    print("[INFO] training network...")
    trainX = XImagetrain
    trainY = XtrainLabels
    testX = XImagetest
    testY = XtestLabels
    
    
    ## defining some essential hyperparameters: # of epochs & batch size
    # epoch_number = 1000
    # BS = 6;
    
    ## Defining the early stop to monitor the validation loss to avoid overfitting
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=250, verbose=1, mode='auto')

    
    ## shuffle = False to reduce randomness and increase reproducibility
    H = model.fit( x = [XtrainData , trainX], y = trainY, validation_data = ( [XtestData, testX], testY), batch_size = BS, epochs = epoch_number, verbose=1, shuffle=False, callbacks = [early_stop]) 
    # H = model.fit( x = [XtrainData , trainX], y = trainY, validation_data = ( [XtestData, testX], testY), batch_size = BS, epochs = epoch_number, verbose=1, shuffle=False, callbacks = [early_stop]) 
    
    
    # save the trained neural network to disk
    print("[INFO] serializing network...")
    if os.path.exists('chembl_drug_prediction/models/trained_models/cnn_mlp_hybrid_Ki') == False:
        os.mkdir('chembl_drug_prediction/models/trained_models/cnn_mlp_hybrid_Ki')
    model.save("chembl_drug_prediction/models/trained_models/cnn_mlp_hybrid_Ki", overwrite = True)
    
    
    # evaluate the network
    print("[INFO] evaluating network...")
    preds = model.predict([XtestData, testX],batch_size=BS)
    
    
    
    ### compute the difference between the predicted and actual target values, then compute the % difference and absolute % difference
    diff = preds.flatten() - testY
    PercentDiff = (diff/testY)*100
    absPercentDiff = (np.abs(PercentDiff)).values
    
    ### compute the mean and standard deviation of absolute percentage difference
    mean = np.mean(absPercentDiff)
    std = np.std(absPercentDiff)
    
    print("[INF0] mean {: .2f},  std {: .2f}".format(mean, std) )
    
    ## Plotting Predicted  vs Actual pCHEMBL values
    N = len(testY)
    colors = np.random.rand(N)
    x = testY * maxPrice
    y =  (preds.flatten()) * maxPrice
    plt.scatter(x, y, c=colors)
    plt.plot( [0,10],[0,10] )
    plt.xlabel('Actual Ki: pCHEMBL Value', fontsize=18)
    plt.ylabel('Predicted Ki: pCHEMBL Value', fontsize=18)
    plt.savefig('chembl_drug_prediction/output_results/Ki_pred.png')
    plt.show()
    
    
    # for i in range(len(preds)):
    #     print("The percent difference in prediction: {}".format(absPercentDiff[i]))
        
    ### The MSE
    print("The mean squared error: {}".format( mean_squared_error(testY, diff, squared = True) ))  
    
    ### The RMSE
    print("The root mean squared error: {}".format( mean_squared_error(testY, diff, squared = False) ))    
    
    ### General stats
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    print("The R^2 value between actual and predicted target:", r_value**2)
    
    
    # plot the training loss and validation loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epoch_number ), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epoch_number ), H.history["val_loss"], label="val_loss")
    plt.title("Training loss and Validation loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('chembl_drug_prediction/output_results/loss@epochs')
    
    return_status = "Training completed"

    return  model, maxPrice