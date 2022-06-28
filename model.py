import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import random_split,DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import gc
import torchmetrics
import argparse


parser = argparse.ArgumentParser()


class Metrics:
    def score(yhat,y):
        """Accuracy Score

        Args:
            yhat (float): probabilities
            y (integer(0/1)): class label 

        Returns:
            float: range 0-1
        """
        accuracy = torchmetrics.Accuracy()
        return accuracy(yhat,y)
    def f1(yhat,y):
        """F1-Score

        Args:
            yhat (float): probabilities
            y (integer(1/0)): class label

        Returns:
            float: range 0-1
        """
        f1 = torchmetrics.F1Score()
        return f1(yhat,y)
    

class Config:
    IMAGE_SIZE   = (224,224)
    TRAIN_FOLDER = './cat-dog/training_set/'
    TEST_FOLDER  = './cat-dog/test_set/'
    DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"
    SPLIT        = (0.7,0.3)
    TRANSFORMS   =  transforms.Compose([transforms.Resize(size=IMAGE_SIZE),
                                       transforms.ToTensor()])
    BATCH_SIZE   = 5
    MODEL_DIR    = 'Pytorch/predefined/'
    EPOCH        =  10
    torch.hub.set_dir(MODEL_DIR)    

    
class TrainTestLoader:
    def __init__(self):
        """ DataLoader Class """
        pass
    def get_data(self, data_folder, split_ratio=Config.SPLIT,
                 split=True, batch_size=Config.BATCH_SIZE,
                 transforms=Config.TRANSFORMS):
        if split:
            """ Call method for quick data loading """
            self.train = ImageFolder(data_folder,transform=transforms)
            self.batch_size = batch_size
            self.split_ratio = split_ratio
            tr,ts = int(len(self.train)*split_ratio[0]),len(self.train)-int(len(self.train)*split_ratio[0])
            train_ds,val_ds = random_split(self.train, [tr,ts])
            self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            self.val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
            return self.train_loader,self.val_loader
        else:
            self.test = ImageFolder(data_folder,transform=transforms)
            self.test_loader = DataLoader(self.test)
            return self.test_loader

class ModelFinal(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def make(self, resnet):
        self.resnet = resnet
    def train(self, X_cuda, y_cuda):
        out = self(X_cuda)
        return torch.nn.functional.binary_cross_entropy(out, y_cuda)
    def fit(self, train_loader, val_loader=None, epochs=1, lr=0.001):
        opt = torch.optim.Adam(self.parameters(),lr=lr)
        for epoch in range(epochs):
            total_avg_loss = 0
            total_loss = 0
            iterator_loader = tqdm.tqdm(train_loader, desc='Train Batch', total=len(train_loader))
            for iteration,batch in enumerate(iterator_loader):
                X_cuda = batch[0].to(dtype=torch.float32, device=Config.DEVICE)
                y_cuda = batch[1].to(dtype=torch.float32, device=Config.DEVICE).unsqueeze(1)
                loss = self.train(X_cuda, y_cuda)
                total_loss += loss.item()
                total_avg_loss = total_loss/(iteration+1)
                loss.backward()
                opt.step()
                opt.zero_grad()
                torch.cuda.empty_cache()
                iterator_loader.set_postfix({'total batch loss':loss.item(), 'total avg loss': total_avg_loss})
            print()
            self.validate(val_loader)
            gc.collect()

    def validate(self,val_loader):
        average_val_score = 0
        average_f1_score = 0
        total_f1 = 0
        total_score = 0
        iterator_loader = tqdm.tqdm(val_loader, desc='Val Batch', total=len(val_loader))
        for iteration,batch in enumerate(iterator_loader):
            X_val_cuda = batch[0].to(dtype=torch.float32, device=Config.DEVICE)
            y_val = torch.unsqueeze(batch[1].to(dtype=torch.int32), dim=1)
            predictions  = self.predict(X_val_cuda).to('cpu')
            
            accuracy = Metrics.score(predictions, y_val)
            total_score += accuracy.item()
            average_val_score  = total_score/(iteration+1)
            
            f1 = Metrics.f1(predictions,y_val)
            total_f1 += f1.item()
            average_f1_score = total_f1/(iteration+1)
            
            iterator_loader.set_postfix({'average f1 score':average_f1_score, 'average accuracy score': average_val_score})
            
        return average_val_score, average_f1_score
            
    def forward(self, X):
        model_output = self.resnet(X)
        softmax_output = torch.nn.Sigmoid()(model_output)
        return softmax_output
    def predict(self,X):
        with torch.no_grad():
            return self(X)


if __name__ == '__main__':
    print('loading data---')
    dl = TrainTestLoader()
    train_loader,val_loader = dl.get_data(Config.TRAIN_FOLDER)
    test_loader = dl.get_data(Config.TEST_FOLDER,split=False)
    print('loaded data---')


    print('loading model---')
    gc.collect()
    resnet    = torch.hub.load('pytorch/vision:v0.10.0','resnet18', pretrained=False)
    num_ftrs  = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs,1)
    resnet    = resnet.to('cuda')
    model_attached = ModelFinal()
    model_attached.make(resnet=resnet)
    print('loaded model---')

    print('starting training---')
    model_attached.fit(train_loader=train_loader)
    print('training finished---')