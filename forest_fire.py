#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")


data=pd.read_csv("D:\\2011CS010144\\finalised_dataset.csv")
data
data=data.fillna(data.mean())
dummies1=pd.get_dummies(data["state_names"])
dummies1
dummies2=pd.get_dummies(data["crop_names"])
dummies2
dummies3=pd.get_dummies(data["district_names"])
dummies3
dummies4=pd.get_dummies(data["season_names"])
dummies4
dummies5=pd.get_dummies(data["soil_type"])
dummies5
data1=pd.concat([data,dummies1,dummies2,dummies3,dummies4,dummies5],axis=1)
data1=data1.drop(["state_names","district_names","crop_names","season_names","soil_type"],axis=1)
data1["temperature"].astype("int64",errors='ignore')
data1["humidity"].astype("int64",errors='ignore')
x=data1(["area","temperature","humidity"])
y=data1(["Yield"])
log_reg = LinearRegression()


log_reg.fit(x,y)

inputt=[int(x) for x in x.split(' ')]
final=[np.array(inputt)]

b = log_reg.predict(final)


pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


