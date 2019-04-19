
from ROOT import TMVA as tmva
from sklearn.externals import joblib
from sklearn.metrics import (classification_report,
								roc_auc_score,
								roc_curve)
from matplotlib.patches import Rectangle
from helpers import *

import numpy as np
import pandas as pd
import xgboost as xgb
import ROOT as root
import matplotlib.pyplot as plt

test_file = "./data/test_data.csv"
x_train,y_train,w_train,x_test,y_test,w_test,columns = get_data(None,test_file)


def scikit_model():
	
	model_path = "./models/sklearn-model.joblib.pkl"

	print "loading scikit-learn....."
	print "....\n...\n..\n."
	sk_model = joblib.load(model_path)

	#decision_function
	# scores corresponding to background
	bg_scores = sk_model.decision_function(x_test[y_test==0])
	# score corresponding to signal
	sg_scores = sk_model.decision_function(x_test[y_test==1])

	plt.figure("fig-1")
	plt.subplot(1, 2, 1)
	#background histogram
	plt.hist(bg_scores.ravel(),
					color='r',
					alpha=0.5,
					range=(-0.7,0.4),
					bins=30)

	#SIGNAL(tau-tau decay) histogram
	plt.hist(sg_scores.ravel(),
					color='b',
					ec='b',
					alpha=0.5,
					range=(-.7,0.4),
					bins=30)
	#show histgrams

	plt.xlabel("scikit-learn BDT output")
	plt.ylabel("Events")

	handles = [Rectangle((0,1),1,1,color='r',ec="k"),
				 Rectangle((0,1),1,1,color='b',ec="k")]
	plt.legend(handles,["background","signal"])
	# plt.show()

	#calculate probabilities
	probs = sk_model.predict_proba(x_test)
	probs = probs[:,1] #take only signal probs

	# print classification_report(y_test, probs, target_names=["background", "signal"])
	AUC = roc_auc_score(y_test,probs)
	print "Area under ROC curve: %.4f"%(AUC)

	plt.subplot(1, 2, 2)
	#calculating false and true postive rate
	fpr, tpr, thresholds = roc_curve(y_test, probs)

	#draw ROC curve
	draw_roc_curve(plt,fpr,tpr,AUC)
	plt.show()

def tmva_model():

	model_path = "dataset_dnn/weights/TMVAClassification_BDT.weights.xml"
	print "loading TMVA........."
	print "....\n...\n..\n."
	reader = tmva.Reader()

	#reloading all the features into the loader
	#tmva requires that
	le = len(y_test)
	for col in columns:
	    reader.AddVariable(col,np.zeros(le,dtype='float32')) # accepts 32 bit only.
	    
	reader.BookMVA("BDT",model_path)

	decision_value = []
	prediction = []
	for row in x_test:
	    a = root.std.vector(root.Double)()
	    for r in row:
	        a.push_back(r)
	        
	    value = reader.EvaluateMVA(a, "BDT")
	    decision_value.append(value)
	    if value > 0:
	        prediction.append(1)
	    else:
	        prediction.append(0)

	# print classification_report(y_test, prediction, target_names=["background", "signal"])
	# print "Area under ROC curve: %.4f"%(roc_auc_score(y_test, prediction))

	dval_arr = np.array(decision_value)
	
	bg_scores = dval_arr[y_test==0] # scores corresponding to background
	sg_scores = dval_arr[y_test==1] # scores corresponding to signal

	plt.figure("fig-1")
	plt.subplot(1, 2, 1)
	#background histogram
	h2 = plt.hist(bg_scores.ravel(),
					color='r',
					alpha=0.5,
					range=(-0.7,0.4),
					bins=30)

	#SIGNAL(tau-tau decay) histogram
	h1 = plt.hist(sg_scores.ravel(),
					color='b',
					ec='b',
					alpha=0.5,
					range=(-.7,0.4),
					bins=30)

	plt.xlabel("TMVA BDT output")
	plt.ylabel("Events")

	handles = [Rectangle((0,1),1,1,color='r',ec="k"),
			   Rectangle((0,1),1,1,color='b',ec="k")]

	plt.legend(handles,["background","signal"])

	AUC = roc_auc_score(y_test,prediction)
	print "Area under ROC curve: %.4f"%(AUC)

	plt.subplot(1, 2, 2)
	#calculating false and true postive rate
	fpr, tpr, thresholds = roc_curve(y_test, dval_arr)
	draw_roc_curve(plt,fpr,tpr,AUC)
	plt.show()

def xgboost_model():

	model_path = './models/higgs.model.%dstep.depth%s'%(3000,9)

	print "loading xgBoost........"
	print "....\n...\n..\n."

	xgmat = xgb.DMatrix(x_test, missing=-999.0, weight=w_test)
	bst = xgb.Booster({'nthread':16})
	bst.load_model(model_path)
	probs = bst.predict(xgmat,ntree_limit=2)

	AUC = roc_auc_score(y_test,probs)
	print "Area under ROC curve: %.4f"%(AUC)

	plt.figure("fig-1")
	#calculating false and true postive rate
	fpr, tpr, thresholds = roc_curve(y_test, probs)
	draw_roc_curve(plt,fpr,tpr,AUC)
	plt.show()

scikit_model()
tmva_model()
xgboost_model()
