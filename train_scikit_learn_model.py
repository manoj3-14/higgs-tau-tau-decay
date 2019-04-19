
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.externals import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_file, test_file = "./data/train_data.csv", "./data/test_data.csv"

x_train,y_train,w_train,x_test,y_test,w_test,columns = get_data(train_file,test_file)

test_size = 50000
# rescale weights
w_train = w_train * float(test_size) / len(y_train)

# bdt = GradientBoostingClassifier()

dt = DecisionTreeClassifier(max_depth=9,min_samples_leaf=0.5)
bdt = AdaBoostClassifier(dt,algorithm='SAMME',n_estimators=800,learning_rate=0.5)

model = bdt.fit(x_train, y_train,sample_weight=w_train)
#storing the model in local disk
joblib.dump(model,"./models/sklearn-model.joblib.pkl",compress=9)
