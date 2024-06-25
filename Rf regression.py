# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:29:07 2024

@author: Louis Liu
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import calculateR2 
import joblib
import matplotlib.pyplot as plt
import shap

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
new_directory = "D:/Study & work/Junqin Liu paper/paper four Fe-mine ads/paper data & code"
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


################ optimization ################
import optuna
from optuna.samplers import TPESampler
from sklearn.pipeline import make_pipeline


kf = KFold(n_splits=5,shuffle=True)
def rf_objective(trial):
  max_depth = trial.suggest_int('max_depth', 5, 30)
  n_estimators = trial.suggest_int("n_estimators", 50, 500, 50)
  rf = make_pipeline(StandardScaler(),RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth))

  kf = KFold(n_splits=5,shuffle=True)
  accuracy = cross_val_score(rf, x_train, y_train, scoring='r2', cv=kf)
  return accuracy.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(rf_objective, n_trials=50) 
study.best_params  
study.best_value    

rf = RandomForestRegressor(n_estimators=study.best_params['n_estimators'], max_depth=study.best_params['max_depth'])
r2 = cross_val_score(rf, x_train, y_train, scoring='r2', cv=kf)
r2.mean()

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
    sc = StandardScaler()
    sc.fit(x_pretrain)  
    x_pretrain = sc.transform(x_pretrain)
    x_valid = sc.transform(x_valid)
    rf = RandomForestRegressor(n_estimators=study.best_params['n_estimators'], max_depth=study.best_params['max_depth'])
    rf.fit(x_pretrain, y_pretrain) 

    p_train = rf.predict(x_pretrain)
    p_test = rf.predict(x_valid)  
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
sc = StandardScaler()
sc.fit(x_train)  
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#joblib.dump(sc,"Cation_sc")
#sc = joblib.load("Cation_sc")
 
rf = RandomForestRegressor(n_estimators=study.best_params['n_estimators'], max_depth=study.best_params['max_depth'])
rf.fit(x_train, y_train)

p_train = rf.predict(x_train)
R2_train = calculateR2(y_train, p_train);print(R2_train)
rmse_train = np.sqrt(mean_squared_error(y_train, p_train));print(rmse_train)  
mae_train = mean_absolute_error(y_train, p_train);print(mae_train)  

p_test = rf.predict(x_test)   
R2_test = calculateR2(y_test, p_test);print(R2_test)
rmse = np.sqrt(mean_squared_error(y_test, p_test));print(rmse)  
mae = mean_absolute_error(y_test, p_test);print(mae)  

plt.clf();plt.close()
plt.scatter(y_test, rf.predict(x_test))
plt.show()


########## imp ###########

impCation_rf = pd.DataFrame(rf.feature_importances_)
impCation_rf.index =  ['Mmine', 'SSA','Fe%', 'Fe/O','PZC'
          , 'Cella','Cellb', 'Cellc','Mmetal','Radius', 'Electronegativity'
          , 'pHe', 'Cmine', 'COM','Cmetal'
          ,'Time', 'Temperature', 'Ionic Strength']


explainer = shap.TreeExplainer(rf)
shap_value_rf = explainer.shap_values(x_train)
shap.summary_plot(shap_value_rf, x_train, feature_names= ['Mmine', 'SSA','Fe%', 'Fe/O','PZC'
          , 'Cella','Cellb', 'Cellc','Mmetal','Radius', 'Electronegativity'
          , 'pHe', 'Cmine', 'COM','Cmetal'
          ,'Time', 'Temperature', 'Ionic Strength'])


shap_df = pd.DataFrame(shap_value_rf)
shap_df.columns = ['Mmine', 'SSA','Fe%', 'Fe/O','PZC'
          , 'Cella','Cellb', 'Cellc','Mmetal','Radius', 'Electronegativity'
          , 'pHe', 'Cmine', 'COM','Cmetal'
          ,'Time', 'Temperature', 'Ionic Strength']
MeanABS_Shap_df =[]

# MAS calculation
for i in range(len(x.iloc[1,:])):
    MeanABS_Shap = sum(abs(shap_df.iloc[:,i]))/len(shap_df)
    MeanABS_Shap_df.append(MeanABS_Shap) 
    
MeanABS_Shap_df = pd.DataFrame(MeanABS_Shap_df)
MeanABS_Shap_df.index = ['Mmine', 'SSA','Fe%', 'Fe/O','PZC'
          , 'Cella','Cellb', 'Cellc','Mmetal','Radius', 'Electronegativity'
          , 'pHe', 'Cmine', 'COM','Cmetal'
          ,'Time', 'Temperature', 'Ionic Strength']

MeanABS_Shap_df = MeanABS_Shap_df.sort_values(by= 0, ascending=False)

with pd.ExcelWriter('MeanABS_shap_Cation.xlsx', engine='openpyxl') as writer:
    MeanABS_Shap_df.to_excel(writer, sheet_name='Sheet1')
