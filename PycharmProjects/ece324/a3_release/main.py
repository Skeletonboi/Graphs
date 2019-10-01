import argparse
from time import time
import sklearn

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *

""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 0

# =================================== LOAD DATASET =========================================== #

raw = pd.read_csv('data/adult.csv', low_memory=False)

# =================================== DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

# print('shape:',raw.shape)
# print('cols',raw.columns)
# print('head',raw.head)
# verbose_print(raw)
# print('inc',raw["income"].value_counts())


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = raw.columns
num_rows = raw.shape[0]
for feature in col_names:
    print(feature, raw[feature].isin(["?"]).sum())




# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

for feature in col_names:
    raw = raw[raw[feature] != "?"]

print(raw, raw.shape)

# =================================== BALANCE DATASET =========================================== #
great = raw[raw["income"] == ">50K"]
less = raw[raw["income"] == "<=50K"]
less2 = less.sample(great.shape[0],random_state=1)
data = pd.concat([great,less2])
print(data,data.shape)
print(data["income"].value_counts())
# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method
#
# print(data.describe())
# verbose_print(data)

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                     'relationship', 'gender', 'native-country', 'income']

# for feature in categorical_feats:
#     print(data[feature].value_counts())
#
# # visualize the first 3 features using pie and bar graphs
# for feature in categorical_feats:
#     pie_chart(data,feature)
#     binary_bar_chart(data,feature)

# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# NORMALIZE CONTINUOUS FEATURES
cont_features = ["age","fnlwgt","educational-num","capital-gain","capital-loss","hours-per-week"]

cont_data = data[cont_features]
for feature in cont_features:
    mean = cont_data[feature].mean()
    std = cont_data[feature].std()
    cont_data[feature] = (cont_data[feature] - mean)/std
    print(cont_data[feature].values)



# ENCODE CATEGORICAL FEATURES

# transformed_cat = []
label_encoder = LabelEncoder()
for feature in categorical_feats:
    data[feature] = label_encoder.fit_transform(data[feature])
    # transformed_cat.append(trans)


cat_data = data[categorical_feats]
oneh_encoder = OneHotEncoder(categories="auto")
temp = oneh_encoder.fit_transform(cat_data)
cat_oneh = temp.toarray()
print(temp)


income = data["income"].to_numpy()
data.drop(columns=['income'])

# Convert
# cat_data_new = pd.DataFrame(cat_oneh)
# print(cat_data_new)

# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

######

# 3.7 YOUR CODE HERE

######

# =================================== LOAD DATA AND MODEL =========================================== #

def load_data(batch_size):
    ######

    # 4.1 YOUR CODE HERE

    ######

    return train_loader, val_loader


def load_model(lr):
    ######

    # 4.4 YOUR CODE HERE

    ######

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    ######

    # 4.6 YOUR CODE HERE

    ######

    return float(total_corr) / len(val_loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()

    ######

    # 4.5 YOUR CODE HERE

    ######


if __name__ == "__main__":
    main()
