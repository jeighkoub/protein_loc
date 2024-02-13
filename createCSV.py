import numpy as np
import h5py
import pandas as pd
import string
from utils import *

def createCSV(h5_path, tsv_path, csv_path):

    print("starting createCSV")

    df_ref = pd.read_csv(tsv_path, sep='\t')
    #df = pd.read_csv(csv_path)



    with h5py.File(h5_path, "r") as file:
        data = []

        print("with h5py.File")

        for sequence_id, embedding in file.items():

            print("sequence_id:", sequence_id)

            embedding_data = np.array(embedding)
            embedding_mean = embedding_data.mean()

            # df_ref has Entry, Sequence, Location
            # match sequence_id to Entry to find location
            location = df_ref.loc[df_ref['Entry'] == sequence_id, 'Location'].iloc[0]

            # Convert embedding array to a string representation within square brackets
            embedding_str = ' '.join(map(str, embedding_data))

            data.append({'Entry': sequence_id, 'Embedding': embedding_str, 'Mean': embedding_mean, 'Location': location})

        df_final = pd.DataFrame(data)

        #remove rows with no location
        df_final = df_final[df_final['Location'].notna()]

        #simplify each location, remove unnecessary information
        df_final['Location'] = df_final['Location'].apply(lambda x: x.split('{')[0])
        df_final['Location'] = df_final['Location'].apply(lambda x: x.split(':')[1])

        #clean up location names
        df_final['Location'] = df_final['Location'].apply(lambda x: x.lower())
        df_final['Location'] = df_final['Location'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        df_final['Location'] = df_final['Location'].apply(lambda x: x.strip())

        #count number of unique locations
        print(df_final['Location'].nunique())

        #for each unique location, how many datapoints are there with that location
        print(df_final['Location'].value_counts())

        df_final.to_csv(csv_path, index=False)


def main():
    import sys
    h5_path = sys.argv[1]
    tsv_path = sys.argv[2]
    csv_path = sys.argv[3]
    createCSV(h5_path, tsv_path, csv_path)

if __name__ == "__main__":
    main()
