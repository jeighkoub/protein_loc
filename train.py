# train protein localization model
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
from csv2df import *
#from skimage import io, transform


#todo: clean this up, train_epoch(), 


#Dataset is a pandas df
class ProteinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform

    def __len__(self):
        # find number of rows in csv file
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get protein embedding
        embedding = self.data.iloc[idx, 0]
        embedding = np.array([embedding]).astype('float')
        #embedding = embedding.astype('float')
        
        # get protein label, an int
        label = self.data.iloc[idx, 1]

        return embedding, label


def main(file_path):
    seed = 442

    # load data
    #path = 'data/bacteria_reviewed_embed.csv' # note change to user input
    path = file_path
    data = csv2df(path)

    #make sure embeddings are of len 1024
    print("checking lengths of embeddings")
    print(len(data['Embedding'][0]))
    

    # split data into train, val, test
    # train (80%) and temp (20%)
    train_data, temp_data = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=seed)

    # 20% temp -> val (10%) and test (10%)
    val_data, test_data = sklearn.model_selection.train_test_split(temp_data, test_size=0.5, random_state=seed)

    print("making datasets")
    train_dataset = ProteinDataset(train_data)
    val_dataset = ProteinDataset(val_data)
    test_dataset = ProteinDataset(test_data)

    print("num unique labels in train:", train_data['Label'].nunique())

    #get item: return embedding, label

    ### print stats
    print('Train dataset size: ', len(train_dataset))
    print('Val dataset size: ', len(val_dataset))
    print('Test dataset size: ', len(test_dataset))

    print("length of col0:",len(train_dataset[0][0]))
    print("[0][0] dtype: ",train_dataset[0][0].dtype)
    print("[0][1] dtype: ",train_dataset[0][1].dtype)

    print("checking lengths of embeddings")
    print(len(train_dataset[0][0]))

    # Create data loaders
    batch_size = 2000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    ####

    model = FFN(batch_size=batch_size)

   
    

    patience = 20
    curr_count_to_patience = 0

    global_min_loss = float('inf')

    max_epochs = 250
    epoch = 0
    learning_rate = 0.008 * (0.9 ** (epoch // 10))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.05)

    best_epoch = 0


    #initial performance before training
    with torch.no_grad():
        test_loss = 0
        correct_predictions = 0
        total_samples = 0

        for x_batch, y_batch in test_loader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item()

            _, predicted_labels = torch.max(y_pred, 1)
            correct_predictions += (predicted_labels == y_batch).sum().item()
            total_samples += y_batch.size(0)

        test_loss /= len(test_loader)
        accuracy = correct_predictions / total_samples

        print('\nTest Loss: {:.4f}'.format(test_loss))
        print('Test Accuracy: {:.2%}'.format(accuracy))

  
    model.train()


    # Training loop
    while curr_count_to_patience < patience and epoch < max_epochs:
        print('Epoch: ', epoch)
        train_loss = 0





        # train epoch
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.float() 
            optimizer.zero_grad()
            y_pred = model(x_batch)
            y_pred = torch.softmax(y_pred, dim=1)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        val_loss = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < global_min_loss:
            global_min_loss = val_loss
            curr_count_to_patience = 0
            best_epoch = epoch
        else:
            curr_count_to_patience += 1
        with torch.no_grad():
            train_acc = 0
            for i in range(len(y_batch)):
                if torch.argmax(y_pred,dim=1)[i] == y_batch[i]:
                    train_acc += 1
            train_acc /= len(y_batch)

        #val acc
        #"batch pred shapes",y_batch,'\n', torch.argmax(y_pred,dim=1))
        #sklearn doesnt work
        accuracy = 0
        for i in range(len(y_batch)):
            if torch.argmax(y_pred,dim=1)[i] == y_batch[i]:
                accuracy += 1
        accuracy /= len(y_batch)

        print('\nTrain Loss: ', train_loss, 'train acc',train_acc,'Val Loss: ', val_loss, 'Val Accuracy: ', accuracy)
        save_checkpoint(model, epoch, 'checkpoints')

        epoch += 1

    # Load best_epoch model for testing
    model = FFN()

    print('Loading model from epoch: ', best_epoch)
    checkpoint = torch.load('checkpoints/epoch={}.checkpoint.pth.tar'.format(best_epoch))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Testing
    test_loss = 0
    #####messing with it now
    with torch.no_grad():
        test_loss = 0
        correct_predictions = 0
        total_samples = 0

        for x_batch, y_batch in test_loader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item()

            _, predicted_labels = torch.max(y_pred, 1)
            correct_predictions += (predicted_labels == y_batch).sum().item()
            total_samples += y_batch.size(0)

        test_loss /= len(test_loader)
        accuracy = correct_predictions / total_samples

        print('\nTest Loss: {:.4f}'.format(test_loss))
        print('Test Accuracy: {:.2%}'.format(accuracy))
    print('Best epoch: ', best_epoch)


if __name__ == '__main__':
    import sys
    file_path = sys.argv[1]
    main(file_path)