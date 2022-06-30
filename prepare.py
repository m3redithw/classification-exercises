import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import acquire

def prep_iris(df):
    
   
    cols_to_drop = ['species_id']
    df = df.drop(columns=cols_to_drop)
    df.rename(columns = {'species_name':'species'}, inplace = True)
    dummy_df = pd.get_dummies(df[['species']], dummy_na= False)
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_titanic(df):

    cols_to_drop = ['deck', 'embarked', 'class', 'age']
    df = df.drop(columns=cols_to_drop)
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False)
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_telco(df):
    cols_to_drop = ['contract_type_id', 'customer_id', 'internet_service_type_id', 'payment_type_id', 'payment_type_id.1', 'internet_service_type_id', 'internet_service_type_id.1']
    df = df.drop(columns=cols_to_drop)
    df.total_charges = df.total_charges.str.strip().replace('',0).astype(float)
    dummy_df = pd.get_dummies(df[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn', 'contract_type', 'payment_type', 'internet_service_type']], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df