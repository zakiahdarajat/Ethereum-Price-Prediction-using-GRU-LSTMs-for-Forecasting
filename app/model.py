import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import keras
from datetime import datetime


base = '/Users/adityavs14/Documents/Internship/Pianalytix/Ethereum/app'

model = keras.models.load_model('/Users/adityavs14/Documents/Internship/Pianalytix/Ethereum/app/model.h5')
length = 240

with open(f'{base}/scaler.pkl', 'rb') as f:
    sc = pickle.load(f)
    f.close()

with open(f'{base}/window.npy', 'rb') as f:
    windows_sc = np.load(f)
    f.close()
with open(f'{base}/target.npy', 'rb') as f:
    target_sc = np.load(f)
    f.close()


def create_steps(to):
    str_d1 = '2020/04/16'
    str_d2 = to
    d1 = datetime.strptime(str_d1, "%Y/%m/%d")
    d2 = datetime.strptime(str_d2, "%Y/%m/%d")

    
   
    delta = d2-d1
    steps_in_future = delta.days * 24


    return steps_in_future

def future(steps_in_future):
    f_wind=windows_sc[-1]
    f_tar=target_sc[-1]
    new=[]

    for i in range(steps_in_future):
        curr = np.append(f_wind[1:],[f_tar]).reshape(-1,1)
        #print(curr,end="\n\n")
        next_pred = model.predict(curr.reshape(1,length,1))
        #pred_ic = sc.inverse_transform(next_pred)
        new.append(next_pred[0][0])
        f_wind = curr
        f_tar=next_pred
    new = sc.inverse_transform(np.array(new).reshape(-1,1))
    return new
