# Helper file for common training functions.

import numpy as np
import h5py
import pandas as pd
import re
import ast
import itertools
import os
import torch
from torch.nn.functional import softmax
from sklearn import metrics
import utils

def cleanDF(df):
    print("received df of size:", df.shape, "with columns:", df.columns, "for cleaning")

    df = df.dropna()

    #clean up location column
    df['Location'] = df['Location'].str.replace('SUBCELLULAR LOCATION: ', '')
    df['Location'] = df['Location'].str.replace(r'^.*?\]', '', regex=True)
    df['Location'] = df['Location'].str.replace(r' \{.*', '', regex=True) #delete everything after " {"
    df['Location'] = df['Location'].str.replace(r'\..*', '', regex=True) #delete everything after first period
    df['Location'] = df['Location'].str.replace(r'\;.*', '', regex=True) #delete everything after semicolon
    df['Location'] = df['Location'].str.replace(r'\:.*', '', regex=True) #delete everything after colon
    df['Location'] = df['Location'].str.replace(r'\,.*', '', regex=True) #delete everything after comma

    #print size of dataframe
    print("size after cleaning:", df.shape)



def count_parameters(model):
    """Count number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, epoch, checkpoint_dir):
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict()
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)

