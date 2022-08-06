import argparse
import os
import logging
import torch
from torch.utils.data import random_split, DataLoader, tr
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from src.utils.common import read_yaml
from src.stage_02_featurization import TrainTestLoader
STAGE = "03-training" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


class Config:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    def __init__(self, model_dir, train_folder, test_folder, image_size, 
                 batch_size, split, epochs):
        torch.hub.set_dir(model_dir)    
        self.image_size   = image_size
        self.train_folder = train_folder
        self.test_folder  = test_folder
        self.split        = split
        self.transforms   = transforms.Compose([transforms.Resize(size=self.image_size), 
                                                transforms.ToTensor()])
        self.batch_size   = batch_size
        self.model_dir    = model_dir
        self.epoch        =  epochs

def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    
    artifacts_dir = config['artifacts']['artifacts_dir']
    
    data_dir     = config['artifacts']['data_source']['data_dir']
    train_data   = config['artifacts']['data_source']['train_folder']
    test_data    = config['artifacts']['data_source']['test_folder']
    
    train_dir = os.path.join(data_dir, train_data)
    test_dir  = os.path.join(data_dir, test_data)
    
    logging.info(f'Train Dir {train_dir}\nTest Dir {test_dir}')
    
    return train_dir, test_dir






if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e