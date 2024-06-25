# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:13:32 2024

@author: Louis Liu
"""

##### nn torch #######
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import calculateR2 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import torch
from torch import nn
import torchvision.transforms as transforms
torch.cuda.is_available()



########## cation data loading ##########  
import os
new_directory = "...your path..."
os.chdir(new_directory)
os.getcwd()

data = pd.read_excel("Dummy_data.xlsx",sheet_name=1)
#data.columns

x = data[['Mmine', 'SSA','Fe%', 'Fe/O','PZC'
          , 'Cella','Cellb', 'Cellc','Mmetal','Radius', 'Electronegativity'
          , 'pH', 'Cmine', 'COM','Cmetal','Time', 'Temperature', 'IonicStrength']]

y = data['η']


########## Anion data loading ##########  
import os
new_directory = "...your path..."
os.chdir(new_directory)
os.getcwd()

data = pd.read_excel("Dummy_data.xlsx",sheet_name=2)
#data.columns

x = data[['Mmine', 'SSA','Fe%', 'Fe/O','PZC'
          , 'Cella','Cellb', 'Cellc','Mmetal','ONum', 'Valence','Radius', 'Electronegativity'
          , 'pH', 'Cmine', 'COM','Cmetal','Time', 'Temperature', 'IonicStrength']]

y = data['η']

################ data splitting ################
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)   
y_test,y_train = np.array(y_test), np.array(y_train)


############# optimization #############
import optuna
from optuna.samplers import TPESampler
 
class Netopt(nn.Module):
    def __init__(self,  trial):
        super(Netopt, self).__init__()
        # Inputs to hidden layer linear transformation
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.3,step=0.1)
        n_hidden = trial.suggest_int("n_hidden", 18, 36, step=2) # !trial.suggest_int("n_hidden", 20, 40, step=2) for optimization of anion ANN model 
        self.layer = torch.nn.Sequential(
                     nn.Linear(18, n_hidden).cuda(), 
                     nn.ReLU(),
                     nn.Dropout(dropout_rate),
                     nn.Linear(n_hidden, n_hidden).cuda(),
                     nn.ReLU(),
                     nn.Dropout(dropout_rate),
                     nn.Linear(n_hidden, 1).cuda(),                            
                     )   
                        
    def forward(self, x):
        y_pred = self.layer(x)                  
        return y_pred   
       
# specify loss function (cross-entropy)
criterion = nn.MSELoss(reduction='mean') #nn.CrossEntropyLoss()
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kf = KFold(n_splits=5,shuffle=True)

def ANN_objective(trial):          
  global R2list_test, modelopt    
  lr = trial.suggest_float("lr", 1e-4, 1e-1,log=True)
  batch_size=trial.suggest_int("batch_size", 64, 256,step=64)
  R2list_test = [] 
  for pretrain_index, valid_index in kf.split(x_train,y_train):
      global x_pretrain,y_pretrain,x_valid,y_valid
      x_pretrain= x_train.iloc[pretrain_index,:] 
      y_pretrain= y_train[pretrain_index]  
      x_valid=x_train.iloc[valid_index,:] 
      y_valid=y_train[valid_index] 
      # build the training and test tensor
      sc = StandardScaler()
      sc.fit(x_pretrain)  
      x_pretrain = sc.transform(x_pretrain)
      x_valid = sc.transform(x_valid)
                              
      x_pretrain = x_pretrain.astype(np.float32)
      x_valid = x_valid.astype(np.float32)
      y_pretrain = np.array(y_pretrain).astype(np.float32)
      y_valid = np.array(y_valid).astype(np.float32)

      train_features = torch.from_numpy(x_pretrain).cuda()
      train_labels = torch.from_numpy(y_pretrain).unsqueeze(1).cuda()
      test_features = torch.from_numpy(x_valid).cuda()
      
      train_set = torch.utils.data.TensorDataset(train_features,train_labels)      
      train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size)                 
      
      modelopt = Netopt(trial).to(Device)
      optimizer = torch.optim.Adam(modelopt.parameters(), lr=lr)
      for i in range(2000):
          
          # train_acc = 0
          modelopt.train() 
          for tdata,tlabel in train_loader:
              
              y = modelopt(tdata)
              # loss recorded 
              loss = criterion(y, tlabel)
              # backpropagation
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
                                
      p_test = torch.Tensor.cpu(modelopt(test_features)).detach().numpy().flatten()
            
      r2_test = calculateR2(y_valid, p_test)                    
      R2list_test.append(r2_test)        
      
  R2_valid = np.array(R2list_test).mean()    
  return R2_valid
  
study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(ANN_objective, n_trials=20) 
study.best_params  
study.best_value  
 

############# optimized ANN structure setting #############
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Inputs to hidden layer linear transformation        
        self.layer = torch.nn.Sequential(
                     nn.Linear(len(x.iloc[1,:]), study.best_params['dropout_rate']).cuda(), 
                     nn.ReLU(),
                     nn.Dropout(study.best_params['dropout_rate']),
                     nn.Linear(study.best_params['dropout_rate'], study.best_params['dropout_rate']).cuda(),
                     nn.ReLU(),
                     nn.Dropout(study.best_params['dropout_rate']),                     
                     nn.Linear(study.best_params['dropout_rate'], 1).cuda(),                            
                     )   
                        
    def forward(self, x):
        y_pred = self.layer(x)                  
        return y_pred   
    
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


################ validation ################
# fivefold cross-validation
kf = KFold(n_splits=5,shuffle=True)
R2list_train = [] 
R2list_test = [] 
rmselist_train = [] 
rmselist_test = [] 
maelist_train = []
maelist_test = []

for pretrain_index, valid_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[pretrain_index,:] 
    y_pretrain= y_train[pretrain_index]  
    x_valid=x_train.iloc[valid_index,:] 
    y_valid=y_train[valid_index] 
    # build the training and test tensor
    sc = StandardScaler()
    sc.fit(x_pretrain)  
    x_pretrain = sc.transform(x_pretrain)
    x_valid = sc.transform(x_valid)
    
    x_pretrain = x_pretrain.astype(np.float32)
    x_valid = x_valid.astype(np.float32)
    y_pretrain = np.array(y_pretrain).astype(np.float32)
    y_valid = np.array(y_valid).astype(np.float32)


    train_features = torch.from_numpy(x_pretrain).cuda()
    train_labels = torch.from_numpy(y_pretrain).unsqueeze(1).cuda()
    test_features = torch.from_numpy(x_valid).cuda()
    test_labels = torch.from_numpy(y_valid).unsqueeze(1).cuda()
    train_set = torch.utils.data.TensorDataset(train_features,train_labels)
    test_set = torch.utils.data.TensorDataset(test_features,test_labels)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=study.best_params['batch_size'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=study.best_params['batch_size'])
    
    losses = []
    eval_losses = []
    model = Net()  
    optimizer = torch.optim.Adam(model.parameters(), lr=study.best_params['lr'])   
    criterion = nn.MSELoss(reduction='mean') #nn.CrossEntropyLoss()

    for i in range(2000):
        train_loss = 0
        # train_acc = 0
        model.train() #网络设置为训练模式 暂时可加可不加
        for tdata,tlabel in train_loader:
             
            y_ = model(tdata)
             
            loss = criterion(y_, tlabel)
             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
             
            train_loss = train_loss + loss.item()
        losses.append(train_loss / len(train_loader))   
            # validation
        eval_loss = 0
        model.eval()    
        for edata, elabel in test_loader:
             
            y_ = model(edata)
             
            loss = criterion(y_, elabel)
                 
            eval_loss = eval_loss + loss.item()
        eval_losses.append(eval_loss / len(test_loader))   
        if (i+1) % 100 == 0:
                    print('epoch: {}, trainloss: {}, evalloss: {}'.format(i+1, train_loss / len(train_loader), eval_loss / len(test_loader)))
 
    p_train = torch.Tensor.cpu(model(train_features)).detach().numpy().flatten() 
    p_test = torch.Tensor.cpu(model(test_features)).detach().numpy().flatten()
    
    r2_trian = calculateR2(y_pretrain, p_train)
    r2_test = calculateR2(y_valid, p_test) 
    rmse_train = np.sqrt(mean_squared_error(y_pretrain, p_train))
    rmse_test = np.sqrt(mean_squared_error(y_valid, p_test)) 
    mae_train = mean_absolute_error(y_pretrain, p_train)
    mae_test = mean_absolute_error(y_valid, p_test)
    
    R2list_train.append(r2_trian) 
    R2list_test.append(r2_test)  
    rmselist_train.append(rmse_train)
    rmselist_test.append(rmse_test) 
    maelist_train.append(mae_train)
    maelist_test.append(mae_test)
    
R2df=pd.DataFrame({'tarin':R2list_train,'valid':R2list_test})   
#np.mean(R2df.iloc[:,0]);np.mean(R2df.iloc[:,1])
rmsedf=pd.DataFrame({'tarin':rmselist_train,'valid':rmselist_test})    
#np.mean(rmsedf.iloc[:,0]);np.mean(rmsedf.iloc[:,1])         
maedf=pd.DataFrame({'tarin':maelist_train,'valid':maelist_test})    
#np.mean(maedf.iloc[:,0]);np.mean(maedf.iloc[:,1])   


################ test ################
# build the training and test tensor
sc = StandardScaler()
sc.fit(x_train)  
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)

train_features = torch.from_numpy(x_train).cuda()
train_labels = torch.from_numpy(y_train).unsqueeze(1).cuda()
test_features = torch.from_numpy(x_test).cuda()
test_labels = torch.from_numpy(y_test).unsqueeze(1).cuda()
train_set = torch.utils.data.TensorDataset(train_features,train_labels)
test_set = torch.utils.data.TensorDataset(test_features,test_labels)



model = Net()       
train_loader = torch.utils.data.DataLoader(train_set, batch_size=study.best_params['batch_size'])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=study.best_params['batch_size'])
criterion = nn.MSELoss(reduction='mean') #nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=study.best_params['lr'])   
 

losses = []
eval_losses = []
for i in range(2000):
    train_loss = 0
    # train_acc = 0
    model.train()  
    for tdata,tlabel in train_loader:
         
        y_ = model(tdata)
         
        loss = criterion(y_, tlabel)
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         
        train_loss = train_loss + loss.item()
    losses.append(train_loss / len(train_loader))   
     
    eval_loss = 0
    model.eval()   
    for edata, elabel in test_loader:
         
        y_ = model(edata)
         
        eloss = criterion(y_, elabel)
              
        eval_loss = eval_loss + eloss.item()
    eval_losses.append(eval_loss / len(test_loader))  
    if (i+1) % 100 == 0:
        print('epoch: {}, trainloss: {}, evalloss: {}'.format(i+1, train_loss / len(train_loader), eval_loss / len(test_loader)))

epoch = len(losses)

y_pred_train = torch.Tensor.cpu(model(train_features)).detach().numpy().flatten()
R2_train = calculateR2(y_train, y_pred_train);print(R2_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train));print(rmse_train)  
mae_train = mean_absolute_error(y_train, y_pred_train);print(mae_train)  

y_pred = torch.Tensor.cpu(model(test_features)).detach().numpy().flatten()
R2_test = calculateR2(y_test, y_pred);print(R2_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred));print(rmse)  
mae = mean_absolute_error(y_test, y_pred);print(mae)  

plt.clf();plt.close()
plt.scatter(y_test, y_pred)
plt.show()


