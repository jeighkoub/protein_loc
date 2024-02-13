import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import *
from models.ffn import *
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


def process_df(data):
    # cols: Entry,Embedding,Mean,Location
    data = data.dropna()
    data = data.reset_index(drop=True)
    data = data.drop(columns=['Entry','Mean'])
    data = data.rename(columns={'Location':'Label'})

    #to lowercase all labels
    data['Label'] = data['Label'].apply(lambda x: x.lower())

    #mapping of location/label to new label
    label_map = {
        'extracellular': 0,
        'cytoplasm': 1,
        'outer Membrane': 2,
        'periplasm': 3,
        'inner Membrane': 4,
        'cell inner membrane': 4,
        'cell membrane': 2,
        'secreted': 0,
        'cellular thylakoid membrane': 4,
        'cell outer membrane': 2,
        'membrane': 2,
        'fimbrium': 2,
        'bacterial flagellum basal body': 2,
        'bacterial microcompartment': 4
    }

    print("unique  labels in data!!!!!!!",data['Label'].nunique())
    print("label: ",data.iloc[0].loc['Label'])



    #if label not in label_map set to 5, else set to value in label_map
    data['Label'] = data['Label'].apply(lambda x: label_map[x] if x in label_map.keys() else 5)

    #num unique labels
    print("unique  labels in data!!!!!!!",data['Label'].nunique())
    print("label: ",data.iloc[0].loc['Label'])

    data['Embedding'] = data['Embedding'].apply(lambda x: np.array([float(val) for val in x.split(' ')]).astype('float'))

    return data


def csv2df(csv_path):
    df = pd.read_csv(csv_path)
    return process_df(df)

