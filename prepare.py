import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import acquire

def handle_missing_values(df):
    return df.assign(
        embark_town=df.embark_town.fillna('Other'),
        embarked=df.embarked.fillna('O'),
    )

def remove_columns(df):
    return df.drop(columns=['deck'])

def encode_embarked(df):
    encoder = LabelEncoder()
    encoder.fit(df.embarked)
    return df.assign(embarked_encode = encoder.transform(df.embarked))

def prep_titanic(df):

    cols_to_drop = ['embarked', 'pclass']
    df = df.drop(columns=cols_to_drop)
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    dummy_df = pd.get_dummies(df[['sex', 'class', 'embark_town']], dummy_na=False, drop_first = True)
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_iris(df):
    cols_to_drop = ['species_id']
    df = df.drop(columns=cols_to_drop)
    df.rename(columns = {'species_name':'species'}, inplace = True)
    dummy_df = pd.get_dummies(df[['species']], dummy_na= False)
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_telco(df):
    cols_to_drop = ['contract_type_id', 'customer_id', 'internet_service_type_id', 'payment_type_id', 'payment_type_id.1', 'internet_service_type_id', 'internet_service_type_id.1']
    df = df.drop(columns=cols_to_drop)
    df.total_charges = df.total_charges.str.strip().replace('',0).astype(float)
    dummy_df = pd.get_dummies(df[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn', 'contract_type', 'payment_type', 'internet_service_type']], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df

# spliting titanic dataset
def train_validate_test_split(df, seed=123):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df.survived
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify=train_and_validate.survived,
    )
    return train, validate, test



def split(df, stratify_by=None):
    """
    Crude train, validate, test split
    To stratify, send in a column name
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.2, random_state=123)
        train, validate = train_test_split(train, test_size=.3, random_state=123)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train[stratify_by])
    
    return train, validate, test