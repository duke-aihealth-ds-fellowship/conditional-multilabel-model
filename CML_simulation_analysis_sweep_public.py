# This script is used to generate classification results and compaing performances between several models (1L, ML, C1L, CML) using simulation data at different training sizes. 
# Created by Angel Huang 7/2024

# 1. Load library and data
useGPU = 1
%matplotlib inline

import sys, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import eli5 (show weights)

import math as m
import shap
import torch
import torch.optim as optim
import pickle # to save and load dictionary
torch.manual_seed(42) # 4/3 all change from 42 to 23

from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import notebook
from Function import TransData
from Function import MTLnet, LogisticRegression, valid_loop, train_loop, plot_roc_all_classes
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.special import expit # numerically stable sigmoid function
from numpy import genfromtxt # read y_target
from datetime import datetime
from tabulate import tabulate

today = datetime.today().strftime('%Y%m%d') # for filename

if useGPU == 1:
    BASE_DIR = 'xxx'
else: 
    BASE_DIR = 'xxx'

MAIN_DIR = os.path.join(BASE_DIR, 'Projects/Multistep_Model')
DATAMAIN_DIR = os.path.join(BASE_DIR, 'Study Data/EHR Data')

# choose settings for simulation data
similarity = 0.8 # 0.5 similarity between 1st label and other labels
signal_strength = 2 # 10 how easy to classify an event
linear_flag = 'linear_' # linear or nonlinear
suffix = linear_flag + 'similarity='+str(similarity)+'_signal='+str(signal_strength)

DATAOUT_DIR = os.path.join(MAIN_DIR, 'Data/Data Output/5label_simulation_400k', suffix) 
RESULT_DIR = os.path.join(MAIN_DIR, 'Output/5label_simulation_400k_' + suffix)
DATA_SCRIPT_DIR = os.path.join(DATAMAIN_DIR, 'Scripts/2_prepare_features/')
sys.path.insert(0,DATA_SCRIPT_DIR) # insert to 1st

# Load data
x = pickle.load(open(DATAOUT_DIR + '/x.pickle','rb'))
prob = pickle.load(open(DATAOUT_DIR + '/prob.pickle','rb'))
event = pickle.load(open(DATAOUT_DIR + '/event.pickle','rb'))

X_df = pd.DataFrame(x)
y_df = pd.DataFrame(event)
# add column names and NDD column at the end
NDD_labels = ['ASD', 'ADHD', 'MD', 'DD', 'LD']
y_df.columns = [f'{label}_flag' for label in NDD_labels]
y_df['NDD_flag'] = y_df.any(axis=1).astype(int)

# Adding anxiety_flag will break stratification, will have class less than 2 cases
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_df, test_size=0.33, 
    stratify=y_df[['NDD_flag','ASD_flag','DD_flag','LD_flag']], 
    random_state=42) # stratify for y labels, same every time


# initiate the standard scaler
ss = StandardScaler() # it has a mean of 0 and a standard deviation of 1

# Transform the test data using the same scaler
all_pred = list(X_df.columns)
num_pred = [x for x in all_pred]
X_train.loc[:, num_pred] = ss.fit_transform(X_train[num_pred])
X_test.loc[:, num_pred] = ss.transform(X_test[num_pred])

# seperate each flag vs. any flag (the last column) for y
y_train_each = y_train.drop(columns=['NDD_flag'])
y_train_any = y_train[['NDD_flag']] # needs double bracket to keep dataframe format
y_train_asd = y_train[['ASD_flag']]
y_test_each = y_test.drop(columns=['NDD_flag'])
y_test_any = y_test[['NDD_flag']]
y_test_asd = y_test[['ASD_flag']]
# For testing model11n
y_train_asd_any = pd.concat([y_train_asd, y_train_any], axis=1)
y_test_asd_any = pd.concat([y_test_asd, y_test_any], axis=1)

print(y_train_each.shape)
print(y_train_any.shape)
print(y_test_each.shape)
print(y_test_any.shape)
print(X_train.shape)
print(y_test_asd_any.shape)

# 2. Modeling functions
# Model Definition
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2000 # number samples in a batch
print(f'Using device: {device}')

def hyperparameter_tuning(model, model_name, X_train, y_train, model_flag, lams=np.logspace(1, 3, 10), lr=np.logspace(-2, 0, 10)):
    N_val_folds = 5
    val_loss = []
    for learning_rate in notebook.tqdm(learning_rates, desc='learning rate loop'):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for lam in notebook.tqdm(lams, desc='regularization lambda loop'):
            for fold_idx in range(N_val_folds):
                loss_list = []
                fold_ref = (N_val_folds * np.arange(len(X_train))) // len(X_train)
                Xtr = X_train[fold_ref != fold_idx]
                ytr = y_train[fold_ref != fold_idx]

                Xval = X_train[fold_ref == fold_idx]
                yval = y_train[fold_ref == fold_idx]

                # load into DataLoader
                # convert pandas.dataframe to numpy.array (needed for TransData to find index)
                Xtr = Xtr.to_numpy().astype(np.float32)
                ytr = ytr.to_numpy().astype(np.float32)
                Xval = Xval.to_numpy().astype(np.float32)
                yval = yval.to_numpy().astype(np.float32)

                # dataloader is the required form of data in pytorch
                # Create datasets
                set_train = TransData(xx=Xtr, yy=ytr)
                set_valid = TransData(xx=Xval, yy=yval)

                # Create dataloader
                #batch_size = 2000 # number samples in a batch
                train_dlr = DataLoader(set_train, batch_size=batch_size, shuffle=False) # True for later
                valid_dlr = DataLoader(set_valid, batch_size=batch_size, shuffle=False)

                loss = np.nan
                #try:
                if 1:
                    train_loop(train_dlr, model, optimizer, [], lam = lam, model_flag = model_flag)
                    valid_loop(valid_dlr, model, [], loss_list, model_flag = model_flag)

                    loss = loss_list[0] # should be 1 number anyway (not including l1-loss)
                    #print(loss)
                    current_result = {
                        'learning_rate': learning_rate,
                        'lambda': lam,
                        'fold_idx': fold_idx,
                        'loss': loss
                    }

                    val_loss.append(current_result)

                #except:
                #    print('Failed with alpha = %.3e' % alpha)

    val_loss = pd.DataFrame(val_loss)
    val_loss_ordered = val_loss.groupby(['lambda', 'learning_rate'])['loss'].median().reset_index().sort_values(
                by=['loss', 'lambda', 'learning_rate'], ascending=[True, True, True])

    # Save valication loss
    savename = 'tuning_bs=' + str(batch_size) + '_' + today
    savename_ordered = 'tuning_ordered_' + today
    # create folder if not exist
    if not os.path.exists(RESULT_DIR +'/' + model_name + '/hyperparameter_tuning/'):
        os.makedirs(RESULT_DIR +'/' + model_name + '/hyperparameter_tuning/')
        print("The new directory is created!")
        
    val_loss.to_csv(RESULT_DIR + '/' + model_name + '/hyperparameter_tuning/' + savename + '.csv')
    val_loss_ordered.to_csv(RESULT_DIR + '/' + model_name + '/hyperparameter_tuning/' + savename_ordered + '.csv')

    # Plot 2d heatmap
    fig, axes = plt.subplots(3,1, figsize=(5, 10), sharex=False)
    sns.lineplot(ax=axes[0], data=val_loss, x='lambda', y='loss')
    sns.lineplot(ax=axes[1], data=val_loss, x='learning_rate', y='loss')
    sns.kdeplot(ax=axes[2], data=val_loss, x='lambda', y='learning_rate', cmap="Reds", shade=True, cbar=True,
                cbar_kws={'label': 'Validation loss density'})
    
    plt.savefig(RESULT_DIR +'/' + model_name + '/hyperparameter_tuning/'  + savename + '.png')
    # plt.clf()
    best_param = val_loss_ordered.values[0] # pick the first with lowest loss
    print('Best lambda, learning_rate, loss is ' + np.array2string(best_param))
    return plt


def split_trainvaltest(X_train, y_train, stratify, X_test=pd.DataFrame(), y_test=pd.DataFrame()):
    
    Xtr, Xval, ytr, yval = train_test_split(
         X_train, y_train, test_size=0.2, stratify=stratify, random_state=42) # stratify for y labels
        
    # convert pandas.dataframe to numpy.array (needed for TransData to find index)
    Xtr = Xtr.to_numpy().astype(np.float32)
    ytr = ytr.to_numpy().astype(np.float32)
    Xval = Xval.to_numpy().astype(np.float32)
    yval = yval.to_numpy().astype(np.float32)
    X_test = X_test.to_numpy().astype(np.float32)
    y_test = y_test.to_numpy().astype(np.float32)

    # dataloader is the required form of data in pytorch
    # Create datasets
    set_train = TransData(xx=Xtr, yy=ytr)
    set_valid = TransData(xx=Xval, yy=yval)
    set_test  = TransData(xx=X_test, yy=y_test)
    
    # Create dataloader
    batch_size = 2000
    train_dlr = DataLoader(set_train, batch_size=batch_size, shuffle=False) # True for later
    valid_dlr = DataLoader(set_valid, batch_size=batch_size, shuffle=False) # True for later
    test_dlr  = DataLoader(set_test, batch_size=batch_size, shuffle=False) # True for later

    print('Dataset dimensions are', Xtr.shape, ytr.shape, Xval.shape, yval.shape, X_test.shape, y_test.shape)
   
    return train_dlr, valid_dlr, test_dlr


# Train model and plot loss
def train_model(model, model_name, model_flag, model_output_dir, train_dlr, valid_dlr, iter, epochs, loss_list, min_valid_loss, optimizer, scheduler, lam, NDD_labels):
    # create folder if not exist
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
        print("The new directory is created!")
    learning_rate = optimizer.param_groups[0]['lr']
    
    for epoch in notebook.tqdm(range(int(epochs)),desc='Training Epochs'):
        #print(f"Epoch {iter+1}\n-------------------------------")
        train_loop(train_dlr, model, optimizer = optimizer, scheduler = scheduler, lam = lam, model_flag=model_flag)
        valid_loss, loss_list = valid_loop(valid_dlr, model, [], loss_list, model_flag=model_flag)

        # Track the minimum valid loss and store state
        #print(valid_loss)
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss

            # save model
            suffix = 'valloss=' + str(round(min_valid_loss,2)) + '_bs=' + str(batch_size) + '_lr=' + str(learning_rate) + '_lambda='  + str(round(lam,2))
            savename = 'params_' + suffix
            savepath = model_output_dir + '/' + savename + '.pt'
            if epoch==1 and not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
                print("The new directory is created!")
            torch.save(model.state_dict(), savepath)

        iter+=1

    # load the last checkpoint with the best model    
    #print(min_valid_loss)
    model.load_state_dict(torch.load(savepath))
    
        
    # Plot validation loss
    number = list(range(1, len(loss_list)+1))

    for j in range(len(NDD_labels)): # 6 labels
        NDD_label = NDD_labels[j]
        savename = 'valLoss_' + NDD_label + '_' + suffix

        fig = plt.figure()
        plt.scatter(number, loss_list)
        plt.xlabel('Epochs')
        plt.ylabel('Average validation loss')
        plt.title(model_name + ': ' + NDD_label + ' - Validation Loss - Epochs Scatter Plot')        
        plt.savefig(model_output_dir + '/' + savename + '.png', bbox_inches='tight')
        plt.show()
    return model, loss_list, min_valid_loss, iter, suffix, plt


def get_y_pred(model, model_name, model_output_dir, test_dlr, prob_or_logit, suffix, NDD_labels):
    # output the prediction for test set performance
    y_target = []
    y_pred = []
    with torch.no_grad():
        for X, y in test_dlr:
            if prob_or_logit== 'prob': # if model output probability, use directly
                pred = model(X)
            elif prob_or_logit== 'logit': # if model output logit, then apply sigmoid to convert to probability
                l = model(X)
                pred = expit(l)
            y_pred += pred.tolist()
            y_target += y.tolist()
    len(y_pred) # 2588 x 6 list in list
    np.savetxt(model_output_dir + '/test_ytarget_' + suffix + '.csv', 
               y_target,
               delimiter =", ", 
               fmt ='% s')
    np.savetxt(model_output_dir + '/test_ypred_' + suffix + '.csv', 
               y_pred,
               delimiter =", ", 
               fmt ='% s')
    
    # Use function to plot roc
    print(NDD_labels)
    plt1=plot_roc_all_classes(y_target, y_pred, NDD_labels, model_name)
    plt1.savefig(model_output_dir + '/roc_pr_NDD_' + suffix + '.png', bbox_inches='tight')
    plt1.show()
    
    # Calibration Plot
    plt2 = calib_plot(y_target, y_pred, model_name, NDD_labels)
    plt2.savefig(model_output_dir + '/calib_NDD_' + suffix + '.png', bbox_inches='tight')
    return y_target, y_pred, plt1, plt2

# Bootstrap for confidence interval
def get_metrics(y_target, y_pred, flag=''):
    # compute AUC and average precision for each label and for "any label" metric
    metrics = {}
    if flag == 'single_label': # for ASD subpopulation, all any are true, just predict ASD
        if len(y_target.shape) == 1: # 1 dim, only 1 label
            i=0
            metrics['auc_label{}'.format(i)] = roc_auc_score(y_target, y_pred)
            metrics['precision_label{}'.format(i)] = average_precision_score(y_target, y_pred)
        else: # calculate for all labels
            for i in range(y_target.shape[1]): 
                metrics['auc_label{}'.format(i)] = roc_auc_score(y_target[:, i], y_pred[:, i])
                metrics['precision_label{}'.format(i)] = average_precision_score(y_target[:, i], y_pred[:, i])
        y_target_any = y_target
        y_pred_any = y_pred
        metrics['auc_any'] = roc_auc_score(y_target_any, y_pred_any)
        metrics['precision_any'] = average_precision_score(y_target_any, y_pred_any)
        
    else:
        y_target_any = y_target.max(axis=1)
        y_pred_any = 1 - np.prod(1 - y_pred, axis=1)

        metrics['auc_any'] = roc_auc_score(y_target_any, y_pred_any)
        metrics['precision_any'] = average_precision_score(y_target_any, y_pred_any)

        for i in range(y_target.shape[1]): # calculate for all labels
            metrics['auc_label{}'.format(i)] = roc_auc_score(y_target[:, i], y_pred[:, i])
            metrics['precision_label{}'.format(i)] = average_precision_score(y_target[:, i], y_pred[:, i])
        
    return metrics


def bootstrap_samples(y_target, y_pred, n_samples=1000, flag=''):
    # initialize array to store metric values
    metrics = {}
    #print(flag)
    for key in get_metrics(y_target, y_pred, flag).keys():
        metrics[key] = np.zeros(n_samples)
    
    # perform bootstrap sampling
    for i in range(n_samples):
        # generate random indices with replacement
        idx = np.random.choice(len(y_target), len(y_target), replace=True)
        
        # compute performance metrics on sample
        sample_metrics = get_metrics(y_target[idx], y_pred[idx], flag=flag)
        
        # store metric values in dictionary
        for key in sample_metrics.keys():
            metrics[key][i] = sample_metrics[key]
    
    # compute confidence intervals for all metrics
    ci = {}
    for key in metrics.keys():
        ci[key] = np.percentile(metrics[key], [2.5, 97.5])
    
    return ci, get_metrics(y_target, y_pred, flag)

# Define your models (model0n and model9n)
def train_and_evaluate_model0n(X_train, y_train_asd, X_test, y_test_asd, train_size):
    model0n_name = 'NN_1label'
    input_dim = X_train.shape[1]
    output_dim = 1  # only use asd column
    model0n = MTLnet(input_dim, output_dim)
    model0n.to(device)
    model0n_output_dir = RESULT_DIR + '/' + model0n_name
    model_flag = 0
    print(model0n)
    lam = 0.0001  # L1 regularization, get from cross-validation
    learning_rate = 0.001
    epochs = int(40000 / np.sqrt(train_size)) # 10k-> 400 epochs, 100k->126 epochs
    #epochs = 50 # need to change based on sample size
    optimizer = torch.optim.Adam(model0n.parameters(), lr=learning_rate)
    stratify = y_train_asd[['ASD_flag']]
    train_dlr, valid_dlr, test_dlr = split_trainvaltest(X_train, y_train_asd, stratify, X_test, y_test_asd)
    numbatches = len(train_dlr)
    scheduler = []
    
    # initialize validation loss during training
    loss_list = []  # track the loss change when training
    iter = 0
    min_valid_loss = 10000  # initialize as a large number
    model0n, loss_list, min_valid_loss, iter, suffix, plt = train_model(model0n, model0n_name, model_flag, model0n_output_dir, train_dlr, valid_dlr, iter, epochs, loss_list, min_valid_loss, optimizer, scheduler, lam, [NDD_labels[0]])
    y_target, y_pred, plt1, plt2 = get_y_pred(model0n, model0n_name, model0n_output_dir, test_dlr, 'logit', suffix, [NDD_labels[0]])
    return y_target, y_pred

def train_and_evaluate_model1n(X_train, y_train_each, X_test, y_test_each, train_size):
    model1n_name = 'NN_1step'
    input_dim = len(X_train.columns)  # number of input features
    output_dim = len(y_train_each.columns)  # number of output labels
    model1n = MTLnet(input_dim, output_dim)
    model1n.to(device)
    model1n_output_dir = RESULT_DIR + '/' + model1n_name
    model_flag = 'L21'
    print(model1n)

    # Use best hyperparameters for logistic regression
    lam = 0.01  # L21 regularization, get from cross-validation
    learning_rate = 0.0001
    epochs = int(60000 / np.sqrt(train_size)) # 10k-> 400 epochs, 100k->126 epochs
    optimizer = torch.optim.Adam(model1n.parameters(), lr=learning_rate)
    stratify = y_train_each[['ASD_flag', 'LD_flag']]
    train_dlr, valid_dlr, test_dlr = split_trainvaltest(X_train, y_train_each, stratify, X_test, y_test_each)
    numbatches = len(train_dlr)
    scheduler = []

    # Initialize validation loss during training
    loss_list = []  # track the loss change when training
    iter = 0
    min_valid_loss = 10000  # initialize as a large number

    # Train model
    model1n, loss_list, min_valid_loss, iter, suffix, plt = train_model(
        model1n, model1n_name, model_flag, model1n_output_dir, train_dlr, valid_dlr, iter, epochs, loss_list, min_valid_loss, optimizer, scheduler, lam, NDD_labels
    )

    # Test results
    print(min_valid_loss)
    y_target, y_pred, plt1, plt2 = get_y_pred(model1n, model1n_name, model1n_output_dir, test_dlr, 'logit', suffix, NDD_labels)
    
    return y_target, y_pred


def train_and_evaluate_model3n(X_train, y_train_any, y_train_each, X_test, y_test_any, y_test_each, train_size):
    model3n_name = 'NN_2weighted'
    input_dim = len(X_train.columns)  # number of input features 
    output_dim = len(y_train_any.columns)  # 1/0
    model3na = MTLnet(input_dim, output_dim)
    model3na.to(device)
    print(model3na)
    model3na_output_dir = RESULT_DIR + '/' + model3n_name + '/a/'
    model3n_output_dir = RESULT_DIR + '/' + model3n_name
    model_flag = 0
    lam = 0.00004
    learning_rate = 0.0001
    epochs = int(80000 / np.sqrt(train_size))
    #epochs = 50
    optimizer = torch.optim.Adam(model3na.parameters(), lr=learning_rate)

    stratify = y_train_any
    train_dlr, valid_dlr, test_dlr = split_trainvaltest(X_train, y_train_any, stratify, X_test, y_test_any)
    numbatches = len(train_dlr)
    scheduler = []

    loss_list = []  # track the loss change when training
    iter = 0
    min_valid_loss = 10000  # initialize as a large number

    model3na, loss_list, min_valid_loss, iter, suffix, plt = train_model(model3na, model3n_name, model_flag, model3na_output_dir, train_dlr, valid_dlr, iter, epochs, loss_list, min_valid_loss, optimizer, scheduler, lam, NDD_labels)
    
    # Step b: use prediction from step a as a criteria to go into step b (predict each)
    # Combined train and validation set to get prediction
    tmp_X_train = X_train.to_numpy().astype(np.float32)
    tmp_y_train = y_train_any.to_numpy().astype(np.float32)
    tmp_set_train = TransData(xx=tmp_X_train, yy=tmp_y_train)
    tmp_train_dlr = DataLoader(tmp_set_train, batch_size=batch_size, shuffle=False)

    # Output the prediction for training and validation set
    y_train_target, y_train_pred, plt1, plt2 = get_y_pred(model3na, model3n_name, model3na_output_dir, tmp_train_dlr, 'logit', suffix, ['anyNDD'])

    # Threshold prediction data to determine which samples goes into step b
    y_train_pred = np.array(y_train_pred)
    model3nb = MTLnet(input_dim, len(y_train_each.columns))
    model3nb.to(device)
    print(model3nb)
    model3nb_output_dir = RESULT_DIR + '/' + model3n_name + '/b/'
    model_flag = 'L21'
    W_train = y_train_pred
    
    if not os.path.exists(model3nb_output_dir):
        os.makedirs(model3nb_output_dir)
        
    pickle.dump(y_train_pred, open(model3nb_output_dir + 'y_train_pred.p', 'wb'))
    
    lam = 0.0001
    learning_rate = 0.0005
    epochs = int(80000 / np.sqrt(train_size))
    #epochs = 100
    optimizer = torch.optim.Adam(model3nb.parameters(), lr=learning_rate)

    stratify = y_train_each[['ASD_flag', 'LD_flag']]
    train_dlr, valid_dlr, test_dlr = split_trainvaltest(X_train * W_train, y_train_each, stratify, X_test, y_test_each)

    loss_list = []  # track the loss change when training
    iter = 0
    min_valid_loss = 10000  # initialize as a large number

    model3nb, loss_list, min_valid_loss, iter, suffix, plt = train_model(model3nb, model3n_name, model_flag, model3nb_output_dir, train_dlr, valid_dlr, iter, epochs, loss_list, min_valid_loss, optimizer, scheduler, lam, NDD_labels)
    
    # Test results
    print(min_valid_loss)
    y_target, y_pred, plt1, plt2 = get_y_pred(model3nb, model3n_name, model3nb_output_dir, test_dlr, 'logit', suffix, NDD_labels)
    
    return y_target, y_pred

# Update model9n training function
def train_and_evaluate_model9n(X_train, y_train, X_test, y_test, train_size):
    model9n_name = 'NN_2in1weighted'
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]  # number of output labels: each + NDD
    model9n = MTLnet(input_dim, output_dim)
    model9n.to(device)
    model_flag = '2in1weighted'
    model9n_output_dir = RESULT_DIR + '/' + model9n_name
    print(model9n)
    # Use best hyperparameters
    lam = 0.001  # 0.0002
    learning_rate = 0.01  # 0.008
    epochs = int(80000 / np.sqrt(train_size))
    #epochs = 50  # has the lowest validation loss
    optimizer = torch.optim.Adam(model9n.parameters(), lr=learning_rate)
    stratify = y_train_sample[['ASD_flag', 'LD_flag']]
    train_dlr, valid_dlr, test_dlr = split_trainvaltest(X_train, y_train, stratify, X_test, y_test)
    numbatches = len(train_dlr)
    scheduler = []
    # plot validation loss during training
    loss_list = []  # track the loss change when training
    iter = 0
    min_valid_loss = 10000  # initialize as a large number
    # Train model
    model9n, loss_list, min_valid_loss, iter, suffix, plt = train_model(model9n, model9n_name, model_flag, model9n_output_dir, train_dlr, valid_dlr, iter, epochs, loss_list, min_valid_loss, optimizer, scheduler, lam, NDD_labels)
    # Test results
    print(min_valid_loss)
    y_target, y_pred, plt1, plt2 = get_y_pred(model9n, model9n_name, model9n_output_dir, test_dlr, 'logit', suffix, NDD_labels)
    return y_target, y_pred

def train_and_evaluate_model10n(X_train, y_train_any, y_train_asd, X_test, y_test_any, y_test_asd, train_size):
    model10n_name = 'NN_2weighted_1L'
    input_dim = len(X_train.columns)  # number of input features
    output_dim = len(y_train_any.columns)  # 1/0
    model10na = MTLnet(input_dim, output_dim)
    model10na.to(device)
    print(model10na)
    model10na_output_dir = RESULT_DIR + '/' + model10n_name + '/a/'
    model10n_output_dir = RESULT_DIR + '/' + model10n_name
    model_flag = 0

    # Use best hyperparameters for logistic regression
    lam = 0.0001
    learning_rate = 0.0001
    epochs = int(80000 / np.sqrt(train_size))
    optimizer = torch.optim.Adam(model10na.parameters(), lr=learning_rate)

    stratify = y_train_any
    train_dlr, valid_dlr, test_dlr = split_trainvaltest(X_train, y_train_any, stratify, X_test, y_test_any)
    numbatches = len(train_dlr)
    scheduler = []

    # Initialize validation loss during training
    loss_list = []  # track the loss change when training
    iter = 0
    min_valid_loss = 10000  # initialize as a large number

    # Train model
    model10na, loss_list, min_valid_loss, iter, suffix, plt = train_model(
        model10na, model10n_name, model_flag, model10na_output_dir, train_dlr, valid_dlr, iter, epochs, loss_list, min_valid_loss, optimizer, scheduler, lam, ['anyNDD']
)
    print(min_valid_loss)

    # Performance on test set (not used)
    #y_target, y_pred, plt1, plt2 = get_y_pred(model10na, model10n_name, model10na_output_dir, test_dlr, 'logit', suffix, ['anyNDD'])
    
    # Step b: use prediction from step a as a criterion to go into step b (predict each)
    # Combine train and validation set to get prediction
    tmp_X_train = X_train.to_numpy().astype(np.float32)
    tmp_y_train = y_train_any.to_numpy().astype(np.float32)
    tmp_set_train = TransData(xx=tmp_X_train, yy=tmp_y_train)
    tmp_train_dlr = DataLoader(tmp_set_train, batch_size=batch_size, shuffle=False)

    # Output the prediction for training and validation set
    y_train_target, y_train_pred, plt1, plt2 = get_y_pred(model10na, model10n_name, model10na_output_dir, tmp_train_dlr, 'logit', suffix, ['anyNDD'])

    # Threshold prediction data to determine which samples go into step b
    y_train_pred = np.array(y_train_pred)
    model10nb = MTLnet(input_dim, len(y_train_asd.columns))
    model10nb.to(device)
    print(model10nb)
    model10nb_output_dir = RESULT_DIR + '/' + model10n_name + '/b/'
    model_flag = 0
    W_train = y_train_pred
    if not os.path.exists(model10nb_output_dir):
        os.makedirs(model10nb_output_dir)
    pickle.dump(y_train_pred, open(model10nb_output_dir + 'y_train_pred.p', 'wb'))

    # Use best hyperparameters for logistic regression
    lam = 0.0001  # L1 regularization, get from cross-validation
    learning_rate = 0.05
    epochs = int(80000 / np.sqrt(train_size))
    optimizer = torch.optim.Adam(model10nb.parameters(), lr=learning_rate)

    stratify = y_train_asd
    train_dlr, valid_dlr, test_dlr = split_trainvaltest(X_train * W_train, y_train_asd, stratify, X_test, y_test_asd)

    # Initialize validation loss during training
    loss_list = []  # track the loss change when training
    iter = 0
    min_valid_loss = 10000  # initialize as a large number

    # Train model
    model10nb, loss_list, min_valid_loss, iter, suffix, plt = train_model(
        model10nb, model10n_name, model_flag, model10nb_output_dir, train_dlr, valid_dlr, iter, epochs, loss_list, min_valid_loss, optimizer, scheduler, lam, [NDD_labels[0]]
    )
    print(min_valid_loss)

    # Test results
    y_target, y_pred, plt1, plt2 = get_y_pred(model10nb, model10n_name, model10nb_output_dir, test_dlr, 'logit', suffix, [NDD_labels[0]])
    return y_target, y_pred

def train_and_evaluate_model11n(X_train, y_train_asd_any, X_test, y_test_asd_any, train_size):
    model11n_name = 'NN_2in1weighted_1L'
    input_dim = len(X_train.columns)  # number of input features
    output_dim = len(y_train_asd_any.columns)  # number of output labels: each + NDD
    model11n = MTLnet(input_dim, output_dim)
    model11n.to(device)
    model_flag = '2in1weighted'
    model11n_output_dir = RESULT_DIR + '/' + model11n_name
    print(model11n)

    # Use best hyperparameters
    lam = 0.001  # 0.0002
    learning_rate = 0.01  # 0.008
    epochs = int(80000 / np.sqrt(train_size))
    optimizer = optim.Adam(model11n.parameters(), lr=learning_rate)  # switch to Adam 2/21/2023

    stratify = y_train_asd_any
    train_dlr, valid_dlr, test_dlr = split_trainvaltest(X_train, y_train_asd_any, stratify, X_test, y_test_asd_any)
    numbatches = len(train_dlr)
    scheduler = []

    # Initialize validation loss during training
    loss_list = []  # track the loss change when training
    iter = 0
    min_valid_loss = 10000  # initialize as a large number

    # Train model
    model11n, loss_list, min_valid_loss, iter, suffix, plt = train_model(
        model11n, model11n_name, model_flag, model11n_output_dir, train_dlr, valid_dlr, iter, epochs, loss_list, min_valid_loss, optimizer, scheduler, lam, NDD_labels
    )
    print(min_valid_loss)

    # Test results
    y_target, y_pred, plt1, plt2 = get_y_pred(model11n, model11n_name, model11n_output_dir, test_dlr, 'logit', suffix, [NDD_labels[0]])
    
    return y_target, y_pred


def generate_bootstrap_indices(n_samples, n_bootstrap, random_state=None):
# To ensure sample indices were used for bootstrap
    rng = np.random.RandomState(random_state)
    indices = [rng.choice(np.arange(n_samples), size=n_samples, replace=True) for _ in range(n_bootstrap)]
    return indices

# Function to calculate AUC and AP with bootstrapping
def bootstrap_auc_ap_ci(model_name, y_target, y_pred, size, results, indices):
    # size is only for storage reason
    auc_values = []
    ap_values = []
    
    # Just select the first column (Autism)
    # Check if y_target and y_pred are lists of lists or just lists
    y_target_asd = [sublist[0] for sublist in y_target]
    y_pred_asd = [sublist[0] for sublist in y_pred]
    
    # Calculate true AUC and AP (be the bar length)
    auc_true = roc_auc_score(y_target_asd, y_pred_asd)
    ap_true = average_precision_score(y_target_asd, y_pred_asd)        
    
    for idx in indices:
        y_resampled = [y_target_asd[i] for i in idx]
        y_pred_resampled = [y_pred_asd[i] for i in idx]
        auc = roc_auc_score(y_resampled, y_pred_resampled)
        ap = average_precision_score(y_resampled, y_pred_resampled)
        auc_values.append(auc)
        ap_values.append(ap)
        
    #auc_mean = np.mean(auc_values)
    auc_ci_lower = np.percentile(auc_values, 2.5)
    auc_ci_upper = np.percentile(auc_values, 97.5)
    
    #ap_mean = np.mean(ap_values)
    ap_ci_lower = np.percentile(ap_values, 2.5)
    ap_ci_upper = np.percentile(ap_values, 97.5)
    
    results[model_name]['train_size'].append(size)
    results[model_name]['auc'].append(auc_true)
    results[model_name]['auc_ci_lower'].append(auc_ci_lower)
    results[model_name]['auc_ci_upper'].append(auc_ci_upper)
    results[model_name]['ap'].append(ap_true)
    results[model_name]['ap_ci_lower'].append(ap_ci_lower)
    results[model_name]['ap_ci_upper'].append(ap_ci_upper)
    results[model_name]['aucs'].append(auc_values)
    results[model_name]['aps'].append(ap_values)
    results[model_name]['y_target'].append(y_target)
    results[model_name]['y_pred'].append(y_pred)
    
    return results



# 3 Train model with different size of training data, evaluate on the same test set
# Iterate over different training set sizes
# Set parameters
n_bootstrap = 1000
random_state = 42  # Seed for reproducibility of bootstrap

# Define training set sizes
train_sizes = [1000, 5000, 10000, 20000, 50000, 100000, 200000] #max len(x) =268000

# Dictionary to store results
results = {'model0n': {'train_size': [], 
                       'auc': [], 'auc_ci_lower': [], 'auc_ci_upper': [], 
                       'ap': [], 'ap_ci_lower': [], 'ap_ci_upper': [], 
                       'aucs': [], 'aps': [], 'y_target': [], 'y_pred': []},
           'model1n': {'train_size': [],
                       'auc': [], 'auc_ci_lower': [], 'auc_ci_upper': [], 
                       'ap': [], 'ap_ci_lower': [], 'ap_ci_upper': [], 
                       'aucs': [], 'aps': [], 'y_target': [], 'y_pred': []},
           'model3n': {'train_size': [],
                       'auc': [], 'auc_ci_lower': [], 'auc_ci_upper': [], 
                       'ap': [], 'ap_ci_lower': [], 'ap_ci_upper': [], 
                       'aucs': [], 'aps': [], 'y_target': [], 'y_pred': []},
           'model9n': {'train_size': [],
                       'auc': [], 'auc_ci_lower': [], 'auc_ci_upper': [], 
                       'ap': [], 'ap_ci_lower': [], 'ap_ci_upper': [], 
                       'aucs': [], 'aps': [], 'y_target': [], 'y_pred': []},
           'model10n': {'train_size': [],
                       'auc': [], 'auc_ci_lower': [], 'auc_ci_upper': [], 
                       'ap': [], 'ap_ci_lower': [], 'ap_ci_upper': [], 
                       'aucs': [], 'aps': [], 'y_target': [], 'y_pred': []},
           'model11n': {'train_size': [],
                       'auc': [], 'auc_ci_lower': [], 'auc_ci_upper': [], 
                       'ap': [], 'ap_ci_lower': [], 'ap_ci_upper': [], 
                       'aucs': [], 'aps': [], 'y_target': [], 'y_pred': []},
           'model12n': {'train_size': [],
                       'auc': [], 'auc_ci_lower': [], 'auc_ci_upper': [], 
                       'ap': [], 'ap_ci_lower': [], 'ap_ci_upper': [], 
                       'aucs': [], 'aps': [], 'y_target': [], 'y_pred': []}}

# Generate bootstrap indices
n_test = len(y_test)
indices = generate_bootstrap_indices(n_test, n_bootstrap, random_state) # indices are applied to the test set with the same size

for size in train_sizes:
    # Subsample the training set
    X_train_sample = X_train[:size]
    y_train_sample = y_train[:size]
    y_train_sample_each = y_train_sample.drop(columns=['NDD_flag'])
    y_train_sample_any = y_train_sample[['NDD_flag']] # needs double bracket to keep dataframe format
    y_train_sample_asd = y_train_sample[['ASD_flag']]
    # For testing model11n
    y_train_sample_asd_any = pd.concat([y_train_sample_asd, y_train_sample_any], axis=1)

    # Train and evaluate model0n
    y_target, y_pred = train_and_evaluate_model0n(X_train_sample, y_train_sample_asd, X_test, y_test_asd, size)
    results = bootstrap_auc_ap_ci('model0n', y_target, y_pred, size, results, indices)
    auc_true = results['model0n']['auc'][-1]
    print(f"Train size: {size}, model0n, AUC mean: {auc_true:.3f}")    
    
    # Train and evaluate model1n
    y_target, y_pred = train_and_evaluate_model1n(X_train_sample, y_train_sample_each, X_test, y_test_each, size)
    results = bootstrap_auc_ap_ci('model1n', y_target, y_pred, size, results, indices)
    auc_true = results['model1n']['auc'][-1]
    print(f"Train size: {size}, model1n, AUC mean: {auc_true:.3f}")    
    
    # Train and evaluate model3n
    y_target, y_pred = train_and_evaluate_model3n(X_train_sample, y_train_sample_any, y_train_sample_each, X_test, y_test_any, y_test_each, size)
    results = bootstrap_auc_ap_ci('model3n', y_target, y_pred, size, results, indices)
    auc_true = results['model3n']['auc'][-1]
    print(f"Train size: {size}, model3n, AUC mean: {auc_true:.3f}")
    
    # Train and evaluate model9n
    y_target, y_pred = train_and_evaluate_model9n(X_train_sample, y_train_sample, X_test, y_test, size)
    results = bootstrap_auc_ap_ci('model9n', y_target, y_pred, size, results, indices)
    auc_true = results['model9n']['auc'][-1]
    print(f"Train size: {size}, model9n, AUC mean: {auc_true:.3f}")
    
    # Train and evaluate model10n
    y_target, y_pred = train_and_evaluate_model10n(X_train_sample, y_train_sample_any, y_train_sample_asd, X_test, y_test_any, y_test_asd, size)
    results = bootstrap_auc_ap_ci('model10n', y_target, y_pred, size, results, indices)
    auc_true = results['model10n']['auc'][-1]
    print(f"Train size: {size}, model10n, AUC mean: {auc_true:.3f}")
    
    # Train and evaluate model11n
    y_target, y_pred = train_and_evaluate_model11n(X_train_sample, y_train_sample_asd_any, X_test, y_test_asd_any, size)
    results = bootstrap_auc_ap_ci('model11n', y_target, y_pred, size, results, indices)
    auc_true = results['model11n']['auc'][-1]
    print(f"Train size: {size}, model11n, AUC mean: {auc_true:.3f}")
    
    # Train and evaluate model12n
    y_target, y_pred = train_and_evaluate_model12n(X_train_sample, y_train_sample_asd, X_test, y_test_asd, size)
    results = bootstrap_auc_ap_ci('model12n', y_target, y_pred, size, results, indices)
    auc_true = results['model12n']['auc'][-1]
    print(f"Train size: {size}, model12n, AUC mean: {auc_true:.3f}")
    
results_output_dir = RESULT_DIR + '/model_comparisons/'
if not os.path.exists(results_output_dir):
        os.makedirs(results_output_dir)
pickle.dump(y_target, open(results_output_dir + 'y_target.p', 'wb'))
pickle.dump(results, open(results_output_dir + 'auc_ap_by_train_size.p', 'wb'))

# Convert results to DataFrame for easier plotting
df_model0n = pd.DataFrame(results['model0n'])
df_model1n = pd.DataFrame(results['model1n'])
df_model3n = pd.DataFrame(results['model3n'])
df_model9n = pd.DataFrame(results['model9n'])
df_model10n = pd.DataFrame(results['model10n'])
df_model11n = pd.DataFrame(results['model11n'])
df_model12n = pd.DataFrame(results['model12n'])

# Figure
results_output_dir = RESULT_DIR + '/model_comparisons/'
results = pickle.load(open(results_output_dir + 'auc_ap_by_train_size.p', 'rb'))   
df_model0n = pd.DataFrame(results['model0n'])
df_model1n = pd.DataFrame(results['model1n'])
df_model3n = pd.DataFrame(results['model3n'])
df_model9n = pd.DataFrame(results['model9n'])
df_model10n = pd.DataFrame(results['model10n'])
df_model11n = pd.DataFrame(results['model11n'])
df_model12n = pd.DataFrame(results['model12n'])

# Plot lines by train set size and significant level
from scipy.stats import ttest_rel
import matplotlib.patches as mpatches
from sklearn.utils import shuffle

# Function to perform paired t-test and add significance stars
# Add bar based on permutation test
def add_perm_significance_bars(ax, x_values, y1_values, y2_values, y_plot, color, metric = 'auc', n_permutations=1000, tail='two'):
    significance_results = []  # List to store the significance results
    for x in range(len(x_values)):
        diff_true, p_value, _ = permutation_test(
            pd.DataFrame({'y_target': y1_values['y_target'][x],
                          'y_pred': y1_values['y_pred'][x]}),
            pd.DataFrame({'y_target': y2_values['y_target'][x],
                          'y_pred': y2_values['y_pred'][x]}),
            metric = metric,
            n_permutations=n_permutations,
            tail=tail
        )
        
        # Store the results
        significance_results.append({
            'x_value': x_values[x],
            'diff_true': diff_true,
            'p_value': p_value
        })
        
        # Plot the significance bars if significant
        if p_value < 0.05 and diff_true < 0:
            xmin = x_values[0] if x == 0 else x_values[x-1]
            xmax = x_values[x]
            ax.hlines(y=y_plot, xmin=xmin, xmax=xmax, colors=color, linewidth=5)
    return significance_results

def permutation_test(df_model1, df_model2, metric = 'auc', n_permutations=1000, tail='two'):
    # Extract target and predicted values (y_target1==y_target2)
    y_target1 = [sublist[0] for sublist in df_model1['y_target']]
    y_pred1 = [sublist[0] for sublist in df_model1['y_pred']]
    y_target2 = [sublist[0] for sublist in df_model2['y_target']]
    y_pred2 = [sublist[0] for sublist in df_model2['y_pred']]

    if metric == 'auc':
        score_func = roc_auc_score
    elif metric == 'ap':
        score_func = average_precision_score
    else:
        raise ValueError("metric must be 'auc' or 'ap'")
     
    # Calculate the true AUCs
    auc_true1 = score_func(y_target1, y_pred1)
    auc_true2 = score_func(y_target2, y_pred2)
    auc_diff_true = auc_true1 - auc_true2

    # Initialize list to store permutation differences
    perm_diffs = []

    for _ in range(n_permutations):
        # Randomly choose to keep or switch y_pred1 and y_pred2 for each y_target
        switch_mask = np.random.choice([True, False], size=len(y_target1))
        y_pred1_perm = np.where(switch_mask, y_pred2, y_pred1)
        y_pred2_perm = np.where(switch_mask, y_pred1, y_pred2)

        # Calculate AUCs for permuted predictions
        auc1_perm = score_func(y_target1, y_pred1_perm)
        auc2_perm = score_func(y_target2, y_pred2_perm)
        perm_diffs.append(auc1_perm - auc2_perm)

    # Convert to numpy array for easier manipulation
    perm_diffs = np.array(perm_diffs)

    # Calculate p-value
    if tail == 'two':
        p_value = (np.sum(np.abs(perm_diffs) >= np.abs(auc_diff_true)) + 1) / (n_permutations + 1)
    elif tail == 'one':
        p_value = (np.sum(perm_diffs >= auc_diff_true) + 1) / (n_permutations + 1)
    else:
        raise ValueError("tail must be 'one' or 'two'")

    return auc_diff_true, p_value, perm_diffs


# AUC and AP plot
def plot_model_auc(ax, df_model, model_name, label, color):
    sns.lineplot(x='train_size', y='auc', data=df_model, label=label, marker='o', color=color, ax=ax)
    ax.fill_between(df_model['train_size'], df_model['auc_ci_lower'], df_model['auc_ci_upper'], alpha=0.2, color=color)
    
def plot_model_ap(ax, df_model, model_name, label, color):
    sns.lineplot(x='train_size', y='ap', data=df_model, label=label, marker='o', color=color, ax=ax)
    ax.fill_between(df_model['train_size'], df_model['ap_ci_lower'], df_model['ap_ci_upper'], alpha=0.2, color=color)

# Create side-by-side plots for AUC and AP with permutation bar
fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
plt.style.use('seaborn')

# Define colors
colors = {
    'model0n': 'black',
    'model1n': 'orange',
    'model9n': 'darkred',
    'model10n': 'purple',
    'model11n': 'lightgreen',
    'model12n': 'lightblue',
    'model1_9n': 'darkblue',
    'model11_9n': 'pink',
    'model12_9n': 'lightblue',
}

# AUC Plot
# Plot models
plot_model_auc(ax[0], df_model0n, 'model0n', '1L NN', colors['model0n'])
plot_model_auc(ax[0], df_model1n, 'model1n', 'ML NN', colors['model1n'])
plot_model_auc(ax[0], df_model9n, 'model9n', 'CML NN', colors['model9n'])
plot_model_auc(ax[0], df_model11n, 'model11n', 'C1L NN', colors['model11n'])

# Add significance stars for AUC (if have all auc values)
#add_significance_stars(ax[0], df_model1n['train_size'], df_model1n['aucs'], df_model9n['aucs'],
#                       df_model1n['auc_ci_lower'], df_model1n['auc_ci_upper'],
#                       df_model9n['auc_ci_lower'], df_model9n['auc_ci_upper'], 0.87)
# Initialize the dictionary to store significance results
significance_results_auc = {}
significance_results_ap = {}

y_plot = 0.93
significance_results_auc[('model0n', 'model9n')] = add_perm_significance_bars(ax[0], df_model0n['train_size'], df_model0n, df_model9n, y_plot, colors['model9n'], 'auc')
y_plot = 0.92
significance_results_auc[('model1n', 'model9n')] = add_perm_significance_bars(ax[0], df_model0n['train_size'], df_model1n, df_model9n, y_plot, colors['model1_9n'], 'auc')
y_plot = 0.91
significance_results_auc[('model11n', 'model9n')] = add_perm_significance_bars(ax[0], df_model0n['train_size'], df_model11n, df_model9n, y_plot, colors['model11_9n'], 'auc')
y_plot = 0.90
significance_results_auc[('model0n', 'model1n')] = add_perm_significance_bars(ax[0], df_model0n['train_size'], df_model0n, df_model1n, y_plot, colors['model1n'], 'auc')
y_plot = 0.89
significance_results_auc[('model0n', 'model11n')] = add_perm_significance_bars(ax[0], df_model0n['train_size'], df_model0n, df_model11n, y_plot, colors['model11n'], 'auc')
pickle.dump(significance_results_auc, open(results_output_dir + 'significance_results_auc.p', 'wb'))


ax[0].set_xlabel('Training Set Size')
ax[0].set_ylabel('AUC Value')
ax[0].set_title('AUC with Different Training Set Sizes')
ax[0].legend()

# AP Plot
# Plot models
plot_model_ap(ax[1], df_model0n, 'model0n', '1L NN', colors['model0n'])
plot_model_ap(ax[1], df_model1n, 'model1n', 'ML NN', colors['model1n'])
plot_model_ap(ax[1], df_model9n, 'model9n', 'CML NN', colors['model9n'])
plot_model_ap(ax[1], df_model11n, 'model11n', 'C1L NN', colors['model11n'])

# Add significance stars for AUC (if have all auc values)
#add_significance_stars(ax[1], df_model1n['train_size'], df_model1n['aps'], df_model9n['aps'], 
#                       df_model1n['ap_ci_lower'], df_model1n['ap_ci_upper'],
#                       df_model9n['ap_ci_lower'], df_model9n['ap_ci_upper'], 0.2)
y_plot = 0.23
significance_results_ap[('model0n', 'model9n')] = add_perm_significance_bars(ax[1], df_model0n['train_size'], df_model0n, df_model9n, y_plot, colors['model9n'], 'ap')
y_plot = 0.225
significance_results_ap[('model1n', 'model9n')] = add_perm_significance_bars(ax[1], df_model0n['train_size'], df_model1n, df_model9n, y_plot, colors['model1_9n'], 'ap')
y_plot = 0.22
significance_results_ap[('model11n', 'model9n')] = add_perm_significance_bars(ax[1], df_model0n['train_size'], df_model11n, df_model9n, y_plot, colors['model11_9n'], 'ap')
y_plot = 0.215
significance_results_ap[('model0n', 'model1n')] = add_perm_significance_bars(ax[1], df_model0n['train_size'], df_model0n, df_model1n, y_plot, colors['model1n'], 'ap')
y_plot = 0.21
significance_results_ap[('model0n', 'model11n')] = add_perm_significance_bars(ax[1], df_model0n['train_size'], df_model0n, df_model11n, y_plot, colors['model11n'], 'ap')

# Create custom legend entries for the significance bars
significance_patch_0n_9n = mpatches.Patch(color=colors['model9n'], label='1L < CML (p<0.05)')
significance_patch_1n_9n = mpatches.Patch(color=colors['model1_9n'], label='ML < CML (p<0.05)')
significance_patch_11n_9n = mpatches.Patch(color=colors['model11_9n'], label='C1L < CML (p<0.05)')
significance_patch_0n_1n = mpatches.Patch(color=colors['model1n'], label='1L < ML (p<0.05)')
significance_patch_0n_11n = mpatches.Patch(color=colors['model11n'], label='1L < C1L (p<0.05)')


ax[1].set_xlabel('Training Set Size')
ax[1].set_ylabel('AP Value')
ax[1].set_title('AP with Different Training Set Sizes')
ax[1].legend()

# Move legend to the bottom right
ax[0].legend(loc='lower right', bbox_to_anchor=(1, 0))
ax[1].legend(loc='lower right', bbox_to_anchor=(1, 0))

# Add the custom patches to the legend in both subplots
sig_patch_all = [significance_patch_0n_9n,
                 significance_patch_1n_9n,
                 significance_patch_11n_9n,
                 significance_patch_0n_1n, 
                 significance_patch_0n_11n
                 ]
ax[0].legend(handles=ax[0].get_legend_handles_labels()[0] + sig_patch_all, loc='lower right', bbox_to_anchor=(1, 0))
ax[1].legend(handles=ax[1].get_legend_handles_labels()[0] + sig_patch_all, loc='lower right', bbox_to_anchor=(1, 0))

plt.tight_layout()
plt.savefig(results_output_dir + '/CVbootstrap1000_auc_ap_by_train_size_0v1v9v11_perm.png', bbox_inches='tight')
plt.show()

pickle.dump(significance_results_ap, open(results_output_dir + 'significance_results_ap.p', 'wb'))
