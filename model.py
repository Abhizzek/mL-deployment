import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=pd.read_csv(r'C:\Users\91913\OneDrive\Desktop\untitled\hepatitis.csv')

X=df.loc[:,'age':'histology']
y=df.loc[:,['class']]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()

def mymodel(model):
    model.fit(X_train,y_train)
    return model
def makepredict():
    rf = rf(kernel= 'rbf')
    model=mymodel(rf)
    return model

