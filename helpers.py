
import pandas as pd

def get_data(tr_file_name,ts_file_name):
	
	x_train,y_train,w_train = None,None,None
	x_test,y_test,w_test = None,None,None
	cols = None

	if tr_file_name:
		print "Loading training dataset..."
		print "..\n."
		tr_data = pd.read_csv(tr_file_name,index_col=0)
		cols = tr_data.columns

		x_train = tr_data[cols[2:-1]].values # features
		y_train = tr_data[cols[0]].values # label 0:background, 1:signal
		w_train = tr_data[cols[-1]].values #weights
		# rescale weight to make it same as test set
		# w_train = w_train * float(test_size) / len(y_train)

	if ts_file_name:
		print "test dataset loading..."
		print "..\n."
		ts_data = pd.read_csv(ts_file_name,index_col=0)
		cols = ts_data.columns

		x_test = ts_data[cols[2:-1]].values # features
		y_test = ts_data[cols[0]].values # label 0:background, 1:signal
		w_test = ts_data[cols[-1]].values #weights

	return x_train,y_train,w_train,x_test,y_test,w_test,cols[2:-1]

def draw_roc_curve(plt,fpr,tpr,auc):
	
	plt.plot([0, 1], [0, 1], linestyle='--')
	plt.plot(fpr, tpr, marker='.')
	plt.text(0.60, 0.05, "AUC = %.4f"%(auc))
	plt.xlabel("true-postive rate")
	plt.ylabel("false-postive rate")
