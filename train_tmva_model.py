
from ROOT import TMVA as tmva
import ROOT as root

import matplotlib.pyplot as plt
from helpers import get_data

train_file, test_file = "./data/train_data.csv", "./data/test_data.csv"

x_train,y_train,w_train,x_test,y_test,w_test,columns = get_data(train_file,test_file)
test_size = 50000
# rescale weights
w_train = w_train * float(test_size) / len(y_train)
# ROOT and TMVA require an open file to store things
# while the excute
# recreate : if exist overwrite
output_file = root.TFile('./models/tmva_output.root', 'recreate')

factory = tmva.Factory("TMVAClassification", output_file, "AnalysisType=Classification")
loader = tmva.DataLoader("dataset_dnn")

#loading features into the loader
for col in columns:
    loader.AddVariable(col, "F")

#loading training data_set to loader
count = 0;
for y,row in zip(y_train, x_train):
    a = root.std.vector(root.Double)()
    for r in row:
        a.push_back(r)

    w = w_train[count]
    if y == 1:
        loader.AddSignalTrainingEvent(a,w)
    else:        
        loader.AddBackgroundTrainingEvent(a,w)
    count += 1 

count = 0
for y,row in zip(y_test, x_test):
    a = root.std.vector(root.Double)() # instantiate a std::vector<double>
    for r in row:
        a.push_back(r)
        
    w = w_test[count]
    if y == 1:
        loader.AddSignalTestEvent(a,w)
    else:        
        loader.AddBackgroundTestEvent(a,w)

#preparing traing trees and test trees
loader.PrepareTrainingAndTestTree(root.TCut("1"),"SplitMode=Random:NormMode=NumEvents");

#select traing method and book them for training
factory.BookMethod(loader,tmva.Types.kBDT, "BDT","nCuts=-1")

#train with all the booked methods
factory.TrainAllMethods()

print "The model is trained."