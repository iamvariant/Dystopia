import pickle
import requests
from numpy import load
import numpy as np
# faces={"keane":1,"tham":2,"shuan":3}
# pickle.dump(faces,open("faces.p","wb"))

# svm=pickle.load(open("svm.pkl","rb"))

# print(svm)
data = load("library.npz", allow_pickle=True)
print(data['arr_1'])
