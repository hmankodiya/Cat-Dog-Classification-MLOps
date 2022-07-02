import torch
import numpy as np
import tqdm
from torch.utils.data import random_split,DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import gc
import argparse
from src.utils.all_utils import Metrics, create_directory, read_yaml
import os

class Config:
    MODEL_DIR    = None
    TRAIN_FOLDER = None
    TEST_FOLDER  = None
    IMAGE_SIZE   = None
    TRANSFORMS   = None 
    SPLIT        = None
    BATCH_SIZE   = None
    EPOCH        = None
    DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--config',
                      '-c',
                      default='config/config.yaml')
    arguments.add_argument('--params',
                      '-p',
                      default='params.yaml')
    
    args = arguments.parse_args()
    # print(args.config)
    
    config = read_yaml(args.config)
    params = read_yaml(args.params)
    
    print(params)
    
    artifacts_dir = config['artifacts']['artifacts_dir']
    model_dir     = config['artifacts']['model_dir']
    model_dir     = os.path.join(artifacts_dir, model_dir)
    create_directory([model_dir])
    torch.hub.set_dir(model_dir)
    Config.MODEL_DIR = model_dir
        
    data_dir = config['data_source']['data_dir']

    train_folder = config['data_source']['train_folder']
    train_folder = os.path.join(data_dir, train_folder)

    test_folder = config['data_source']['test_folder']
    test_folder = os.path.join(data_dir, test_folder)
    
    Config.TRAIN_FOLDER = train_folder
    Config.TEST_FOLDER  = test_folder
    
    image_size = params['IMAGE_SIZE']
    Config.IMAGE_SIZE = image_size
    aug = transforms.Compose([transforms.Resize(size=image_size),
                                transforms.ToTensor()])
    config.TRANSFORMS = aug
    
    split = params['SPLIT']
    Config.split = split
    
    batch_size = params['BATCH_SIZE']
    Config.BATCH_SIZE = batch_size
    
    epoch = params['epoch']
    Config.EPOCH = epoch
    
    
    # torch.hub.set_dir(config['model_dir'])   
    
    # print('loading data---')
    # dl = TrainTestLoader()
    # train_loader,val_loader = dl.get_data(Config.TRAIN_FOLDER)
    # test_loader = dl.get_data(Config.TEST_FOLDER,split=False)
    # print('loaded data---')


    # print('loading model---')
    # gc.collect()
    # resnet    = torch.hub.load('pytorch/vision:v0.10.0','resnet18', pretrained=False)
    # num_ftrs  = resnet.fc.in_features
    # resnet.fc = torch.nn.Linear(num_ftrs,1)
    # resnet    = resnet.to('cuda')
    # model_attached = ModelFinal()
    # model_attached.make(resnet=resnet)
    # print('loaded model---')

    # print('starting training---')
    # model_attached.fit(train_loader=train_loader)
    # print('training finished---')