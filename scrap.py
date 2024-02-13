#scrap work, ignore

import numpy as np
import h5py
import pandas as pd
import re
import ast


def main():


    path = 'data/please.csv'

    #read csv
    df = pd.read_csv(path)
    print(df.head())
    print(df.columns)
    print(df.shape)

    #print number unique locations
    print("nunique please",df['Location'].nunique())

    df2 = pd.read_csv('data/bacteria_reviewed_embed.csv')
    print("nunique bacteria_reviewed_embed",df2['Location'].nunique())


    '''
    path = 'protloc/data/bacteria_reviewed.h5'

    # h5 -> csv
    with h5py.File(path, "r") as file:
        # find all columns h5
        #entries = list(file.keys())

        for sequence_id, embedding in file.items():
            embedding_data = np.array(embedding)
            embedding_str = ' '.join(map(str, embedding_data))

        # create dataframe
        df = pd.DataFrame()


        # convert to csv
        #df.to_csv('protloc/data/bacteria_reviewed.csv', index=False)
    
    
    '''



if __name__ == "__main__":
    main()
