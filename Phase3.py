# Group 23

import __future__

import numpy as np
import pandas as pd
import math #for nan check
from datetime import *


# Classifiers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier



#Cross-validation and tuning
from sklearn.model_selection import cross_val_score


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def preprocess(df):
    df = df.drop(['COLLISION_ID','LOCATION','X','Y'], 1)


    for index, row in df.iterrows():
        minute = int( row['TIME'][ (row['TIME'].index(":")+1): -6 ] )
        hour = int(row['TIME'][:row['TIME'].index(":")])
        apm = (row['TIME'])[-3:]

        # DATE
        df.at[index, 'DATE'] = row['DATE'][5:7] + row['DATE'][8:]

        # TIME
        now = datetime.strptime( row['DATE'] + ' ' + (row['TIME'])[:-6] + apm, '%Y-%m-%d %I:%M %p' )
        df.at[index, 'TIME'] = now.strftime('%H%M')
    
        # ENVIRONMENT
        df.at[index, 'ENVIRONMENT'] = row['ENVIRONMENT'][:2]

        # LIGHT
        df.at[index, 'LIGHT'] = row['LIGHT'][:2]

        # SURFACE CONDITION
        if (row['SURFACE_CONDITION'][:2] == '01'):
            #df.at[index, 'SURFACE_CONDITION'] = row['SURFACE_CONDITION'][:2]
            df.at[index, 'SURFACE_CONDITION'] = 0
        else:
            df.at[index, 'SURFACE_CONDITION'] = 1

        # TRAFFIC CONTROL
        if ( not isinstance(row['TRAFFIC_CONTROL'], float) ):
            #print(type((row['TRAFFIC_CONTROL_CONDITION']))) # float if empty, string otherwise
            df.at[index, 'TRAFFIC_CONTROL'] = row['TRAFFIC_CONTROL'][:2]

        # TRAFFIC CONTROL CONDITION
        if ( not isinstance(row['TRAFFIC_CONTROL_CONDITION'], float) ):
            df.at[index, 'TRAFFIC_CONTROL_CONDITION'] = row['TRAFFIC_CONTROL_CONDITION'][:2]
    
        # COLLISION CLASSIFICATION
        df.at[index, 'COLLISION_CLASSIFICATION'] = row['COLLISION_CLASSIFICATION'][:2]

        # IMPACT TYPE
        df.at[index, 'IMPACT_TYPE'] = row['IMPACT_TYPE'][:2]


    df['TARGET'] = df.iloc[:,6] # TARGET = SURFACE_CONDITION
    df = df[['DATE','TIME','ENVIRONMENT','LIGHT','TRAFFIC_CONTROL','TRAFFIC_CONTROL_CONDITION', 'COLLISION_CLASSIFICATION', 'IMPACT_TYPE', 'TARGET']]

    # remove rows with nans - results in removal of 7390 rows (from 14843 to 7453)
    df = df.loc[df.TRAFFIC_CONTROL_CONDITION.apply(type) != float]
    df = df.loc[df.TRAFFIC_CONTROL.apply(type) != float]

    #df = df.drop(df[df.TARGET == '01'].index) # DROP DRy


    print(len(df[ (df['TARGET']==0) ])) # 4974
    print(len(df[ (df['TARGET']==1) ])) # 2479
    return df


def train(df):

    print(df.head(3))
    

    features = ['DATE','TIME','ENVIRONMENT','LIGHT','TRAFFIC_CONTROL','TRAFFIC_CONTROL_CONDITION', 'COLLISION_CLASSIFICATION', 'IMPACT_TYPE']
    target = 'TARGET'
    df[target] = df[target].astype('int')

    # ---------------------------------------- train model ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33)  # for personal testing (will not be used in final program)

    scaler = StandardScaler()
    X_train = scaler.fit(X_train).transform(X_train)

    #X_test = X_test.fillna(X_train.mean())

    #classifier = OneVsRestClassifier(LinearSVC()) # performance: ~67-74%
    classifier = LogisticRegression(C=70, class_weight='balanced')
    #classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=2)
    # classifier = DummyClassifier(strategy='stratified') # performance: ~45%
    classifier.fit(X_train, y_train)



    # ---------------------------------------- results ------------------------------------
    X_test = scaler.fit(X_test).transform(X_test)
    predictions = classifier.predict(X_test)
    #print(predictions)

    #print(classifier.score(X_test, y_test))
    # try print recall
    print('recall_score: ' + str(recall_score(y_test, predictions)))
    print('accuracy_score: ' + str(accuracy_score(y_test, predictions)))
    print('precision_score: ' + str(precision_score(y_test, predictions)))
    print('f1_score: ' + str(f1_score(y_test, predictions)))
    print('roc_auc: ' + str(roc_auc_score(y_test, predictions)))


    print("======== Cross Validated Scores: ========")

    scores = cross_val_score(classifier, X_test, y_test, cv=5, scoring='recall')
    print('recall: ' + str(scores.mean()))

    scores = cross_val_score(classifier, X_test, y_test, cv=5, scoring='accuracy')
    print('accuracy: ' + str(scores.mean()))

    scores = cross_val_score(classifier, X_test, y_test, cv=5, scoring='precision')
    print('precision: ' + str(scores.mean()))

    scores = cross_val_score(classifier, X_test, y_test, cv=5, scoring='f1')
    print('f1: ' + str(scores.mean()))

    # ---------------------------------------- Graph of prediction ------------------------------------
    #plt.scatter(y_test, predictions)
    #plt.xlabel('True Values')
    #plt.ylabel('Predictions')
    #plt.show()
    



''' ----------------------------------------------------------------------------------------
    ------------------------------------- MAIN PROGRAM ------------------------------------- 
    ---------------------------------------------------------------------------------------- '''

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False) # --- display whole dataframe

np.set_printoptions(threshold=np.inf)


df = pd.read_csv("2014collisionsfinal.csv")
df = preprocess(df)

# drop rows with missing values
#df = df.dropna(inplace=True)

train(df)