# Group 23

import __future__

import numpy as np
import pandas as pd
import math #for nan check
from datetime import *

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
        df.at[index, 'SURFACE_CONDITION'] = row['SURFACE_CONDITION'][:2]

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


    print(df.head(5))

''' ----------------------------------------------------------------------------------------
    ------------------------------------- MAIN PROGRAM ------------------------------------- 
    ---------------------------------------------------------------------------------------- '''

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False) # --- display whole dataframe


df = pd.read_csv("2014collisionsfinal.csv")
preprocess(df)

