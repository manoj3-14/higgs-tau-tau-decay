import numpy as np
import pandas as pd
import xgboost as xgb

from helpers import get_data

train_file, test_file = "./data/train_data.csv", "./data/test_data.csv"

x_train,y_train,w_train,x_test,y_test,w_test,columns = get_data(train_file,test_file)

test_size = 50000

#resacaling training weights
w_train = w_train * float(test_size) / len(y_train)
#sum of weigths of label = 1 (signal)
sum_wpos = sum(w_train[i] for i in range(len(y_train)) if y_train[i] == 1.0)
#sum of weigths of label = 0 (background)
sum_wneg = sum(w_train[i] for i in range(len(y_train)) if y_train[i] == 0.0)

# construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
xgmat = xgb.DMatrix(x_train, label=y_train, missing = -999.0, weight=w_train)

param = {}
# use logistic regression loss, use raw prediction before logistic transformation
param['objective'] = 'binary:logitraw'
param['scale_pos_weight'] = sum_wneg/sum_wpos
param['bst:eta'] = 0.01
param['bst:max_depth'] = 9
param['bst:subsample'] = 0.9
param['eval_metric'] = 'ams@0.14'
#param['eval_metric'] = 'error'
param['silent'] = 1
param['nthread'] = 16

# we can directly throw param in, though we want to watch multiple metrics here 
plst = list(param.items())#+[('eval_metric', 'ams@0.15')]

watchlist = [ (xgmat,'train') ]
# boost 120 tres
num_round = 3000
#num_round = 200
print ('loading data end, start to boost trees')
bst = xgb.train( plst, xgmat, num_round, watchlist);
# save out model
bst.save_model('./models/model-xgboost.model')
print ('finish training')
