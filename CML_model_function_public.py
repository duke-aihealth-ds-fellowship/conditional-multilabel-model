################################
#!usrbinenv python
# coding utf-8

# This script defines major models, training, and performance matrix functions 
# (Note last layer is linear because we use BCEWithLogitsLoss later)
#  MTLnet
#  LogisticRegression (not used)
#  train_loop
#  valid_loop
#  TransData
#  plot_roc_all_classes
#  calib_plot

# Angel Huang Created 2013/2
################################

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve

device = cuda if torch.cuda.is_available() else cpu # this is needed for .predict to(device)

# construct the model
class MTLnet(nn.Module)
    def __init__(self, input_dim, output_dim)
        nn.Module.__init__(self)
        self.flatten = nn.Flatten() # needed
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(input_dim, 100),   # size of input layer ~280
            nn.ReLU(),
            nn.Linear(100, output_dim)   # size of output layer = number of outcomes
            #nn.Sigmoid() # use BCEWithLogitsLoss instead
        )
    def forward(self, x)
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits
    
    def predict(self,x) # for shap
        # x is numpy not tensor, return numpy
        xx = torch.tensor(x, dtype = torch.float32).to(device)
        with torch.no_grad()
            probs = torch.exp(self.forward(xx))
        return probs.numpy()

class LogisticRegression(nn.Module)
    def __init__(self, input_dim, output_dim)
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x)
        #outputs = torch.sigmoid(self.linear(x))
        outputs = self.linear(x)
        return outputs

def train_loop(dataloader, model, optimizer, scheduler, lam, model_flag=0)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    nlabels = len(dataloader.dataset[0][1]) # i.e. y.size(dim=1) eg. 6 labels
    total_loss, overall_loss = 0, 0
    #for batch_index, (X, y) in enumerate(dataloader)
    for X, y in dataloader    
        # Compute prediction and loss
        l = model(X) # new model output logits
        pred = torch.sigmoid(l)
        #print(pred)
        pred_any = 1-torch.prod(1-pred,dim=1)
        y_any = torch.max(y,dim=1).values
        loss = [nn.BCEWithLogitsLoss()(l[,ilabel],y[,ilabel]) for ilabel in range(nlabels)]
        #print(loss)
       
        if model_flag == '2in1weighted' # 2-step loss 
            pred_any = pred[,-1]
            joint_loss = torch.tensor(0)
            sample_weight = pred_any.detach() # loss is not differentiable with respect to weight, need to detach
            loss = [nn.BCEWithLogitsLoss(weight=sample_weight)(l[,ilabel],y[,ilabel]) for ilabel in range(nlabels-1)]
            
        else
            joint_loss = torch.tensor(0) 
            
        total_loss = sum(loss) + joint_loss # this will weight joint_loss same as each condition             
        overall_loss += total_loss.item()
        
        if model_flag == L21
        # L2-1 regularization encourage the weight vectors for different labels to be similar to each other. It provides a grouping effect by penalizing large differences in the weights across labels.  
            weight_norms = []
            # 1. Compute the L1 norm of each label's weight vector
            for name, param in model.named_parameters()
                if weight in name
                    weight_norms.append(torch.norm(param, p=1))
            # 2. Calculate the L2-1 regularization penalty term
            l2_1_reg = lam  torch.norm(torch.tensor(weight_norms))
            
            # 3. Add the L2-1 penalty term to the loss
            total_l1_loss = total_loss + l2_1_reg
                    
        else
        #L1 regularization
            regularization_loss = 0
            for param in model.parameters()
                regularization_loss += torch.sum(torch.abs(param))
                #regularization_loss += torch.norm(param, p=1)
            l1_reg = lam  regularization_loss
            total_l1_loss = total_loss + l1_reg
        
        # Backpropagation
        optimizer.zero_grad()
        total_l1_loss.backward()
        optimizer.step()
                    
    # Learning rate scheduler
    if scheduler
        scheduler.step()
        print('lr=',scheduler.get_lr())
    overall_loss = num_batches
    
    #print(ftrain loss {overall_loss7f} )


def valid_loop(dataloader, model, scheduler, loss_list, model_flag=0)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    nlabels = len(dataloader.dataset[0][1]) # i.e. y.size(dim=1) eg. 6 labels
    valid_loss, total_loss, correct = 0, 0, 0

    with torch.no_grad()
        for X, y in dataloader
            
            l = model(X) # new model output logits
            pred = torch.sigmoid(l)
            pred_any = 1-torch.prod(1-pred,dim=1)
            y_any = torch.max(y,dim=1).values
            loss = [nn.BCEWithLogitsLoss()(l[,ilabel],y[,ilabel]) for ilabel in range(nlabels)]
            
            if model_flag == '2in1weighted' # 2-step loss               
                pred_any = pred[,-1] # the last column is probability of any NDD
                sample_weight = pred_any.detach() # loss is not differentiable with respect to weight, need to detach
                loss = [nn.BCEWithLogitsLoss(weight=sample_weight)(l[,ilabel],y[,ilabel]) for ilabel in range(nlabels-1)]
                joint_loss = torch.tensor(0)
            else
                joint_loss = torch.tensor(0)             
            
            
            total_loss = sum(loss) + joint_loss # weight joint_loss same as each condition            
            valid_loss += total_loss.item()           
            
                
    valid_loss = num_batches  #compare the average loss
    loss_list.append(valid_loss)
    # Add learning rate schedular based on plateau of validation loss
    if scheduler # if schedule is not empty list, then use scheduler
        scheduler.step(valid_loss)        
    
    return valid_loss, loss_list
      

class TransData(Dataset)
    def __init__(self, xx, yy)
        self.X =xx
        self.y =yy

    def __len__(self)
        return self.X.shape[0]

    def __getitem__(self, idx)
        return torch.from_numpy(self.X[idx,]), torch.from_numpy(self.y[idx,])
        

def plot_roc_all_classes(y_target, y_pred, NDD_labels, model_name)
    y_target = np.array(y_target)
    y_pred = np.array(y_pred)
    
    fpr = dict()
    tpr = dict()
    ap  = dict()
    roc_auc = dict() # auc is the function name
    precision = dict()
    recall = dict()
    
    # only calculate if there are multiple labels
    #print(NDD_labels)
    if len(NDD_labels)  1 
        for j in range(len(NDD_labels))
            fpr[j], tpr[j], _ = roc_curve(y_target[,j], y_pred[,j])
            roc_auc[j] = roc_auc_score(y_target[,j], y_pred[,j])

            precision[j], recall[j], _ = precision_recall_curve(y_target[,j], y_pred[,j])
            ap[j] = average_precision_score(y_target[,j], y_pred[,j])
    else
        j=0
        fpr[j], tpr[j], _ = roc_curve(y_target[,j], y_pred[,j])
        roc_auc[j] = roc_auc_score(y_target[,j], y_pred[,j])

        precision[j], recall[j], _ = precision_recall_curve(y_target[,j], y_pred[,j])
        ap[j] = average_precision_score(y_target[,j], y_pred[,j])

    
    # Compute any ROC curve and ROC area
    y_target_any = y_target.max(axis=1)
    y_pred_any = 1-np.prod(1-y_pred,axis=1)
    
    fpr['any'], tpr['any'], _ = roc_curve(y_target_any, y_pred_any) # 
    roc_auc['any'] = roc_auc_score(y_target_any, y_pred_any)
    precision['any'], recall['any'], _ = precision_recall_curve(y_target_any, y_pred_any)   
    ap['any'] = average_precision_score(y_target_any, y_pred_any)
    
    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8)) #width and height (in inches)
    lw = 2
    # Plot for each label
    for j in range(len(NDD_labels))
        ax[0,0].plot(fpr[j], tpr[j],
                 lw=lw, label= '%s AUC = %.3f' % (NDD_labels[j], roc_auc[j]))    
    ax[0,0].plot([0, 1], [0, 1], 'k--', lw=lw, label='No information')
    ax[0,0].set_xlim([-.01, 1.01])
    ax[0,0].set_ylim([-.01, 1.01])
    ax[0,0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    ax[0,0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    ax[0,0].set_title(model_name + ' ROC for each NDD')
    ax[0,0].legend(loc=lower right, fontsize=9)

    
    # Plot AUC for any condition
    ax[0,1].plot(fpr['any'], tpr['any'], lw=lw, label= '%s AUC = %.3f' % ('Any condition', roc_auc['any']))   
    ax[0,1].plot([0, 1], [0, 1], 'k--', lw=lw, label='No information')
    ax[0,1].set_xlim([-.01, 1.01])
    ax[0,1].set_ylim([-.01, 1.01])
    ax[0,1].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    ax[0,1].set_ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    ax[0,1].set_title(model_name + ' ROC any condition')
    ax[0,1].legend(loc=lower right, fontsize=9)

    # Plot Precision-recall for each condition
    for j in range(len(NDD_labels))
        ax[1,0].plot(recall[j], precision[j],
                 lw=lw, label= '%s avg precision = %.3f' % (NDD_labels[j], ap[j])) 
    ax[1,0].set_xlim([-.01, 1.01])
    ax[1,0].set_ylim([-.01, 1.01])
    ax[1,0].set_xlabel('Recall (Sensitivity)', fontsize=14)
    ax[1,0].set_ylabel('Precision (Positive Predictive Value)', fontsize=14)
    ax[1,0].set_title(model_name + ' Precision-recall any condition')
    ax[1,0].plot([0, 1], [y_target_any.mean(), y_target_any.mean()], 'k--', label='No information')
    ax[1,0].legend(loc='upper right', fontsize=9)

    # Plot Precision-recall for any condition
    plt.subplot(224)
    plt.subplots_adjust(top=0.85)
    plt.plot(recall['any'], precision['any'], label='Avg precision for any condition = %.3f' % ap['any'])
    plt.xlim([-.01, 1.01])
    plt.ylim([-.01, 1.01])
    plt.xlabel('Recall (Sensitivity)', fontsize=14)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=14)
    plt.title(model_name + ' Precision-recall any condition')
    plt.plot([0, 1], [y_target_any.mean(), y_target_any.mean()], 'k--', label='No information')
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9]) # these 2 lines to avoid overlapping titles
    plt.subplots_adjust(top=0.85) # has to be after tight_layout
    return plt


# Calibration Plot
def calib_plot(y_target, y_pred, model_name, NDD_labels)
    from sklearn.calibration import calibration_curve
    import matplotlib.lines as mlines
    nbins = 10
    y_target_array = np.array(y_target)
    y_pred_array = np.array(y_pred)
    nlabels = len(NDD_labels)
    #cali_x = np.empty([nbins,nlabels])
    #cali_y = np.empty([nbins,nlabels])
    fig, ax = plt.subplots()
    for ilabel in range(nlabels)
        cali_y, cali_x = calibration_curve(y_target_array[,ilabel], y_pred_array[,ilabel], n_bins=nbins)
        plt.plot(cali_x, cali_y, marker = o, linewidth = 1, label = NDD_labels[ilabel])

    line = mlines.Line2D([0, 1], [0, 1], color=black)
    ax.add_line(line)
    fig.suptitle(model_name +  Calibration Plot, fontsize=15)
    ax.set_xlabel(Predicted Probability, fontsize=15)
    ax.set_ylabel(True Probability in Each Bin, fontsize=15)
    plt.xlim((-0.05,1.05))
    plt.ylim((-0.05,1.05))
    plt.legend()
    #plt.show() #if show, saved plot will be empty
    return plt
