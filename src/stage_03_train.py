import argparse
import os
import logging
import torch
from torchvision.transforms import transforms
from src.utils.common import read_yaml
from src.stage_02_featurization import TrainTestLoader, get_data_paths
from src.utils.metrics import Metrics
import tqdm
import gc

STAGE = "03-training" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


class Config:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    image_size   = None
    transforms   = None
    # transforms.Compose([transforms.Resize(size=image_size), 
    #                                     transforms.ToTensor()])
    train_folder  = None
    test_folder   = None
    split         = None
    batch_size    = None
    model_dir     = None
    epochs        = None
    learning_rate = None
    random_state  = None
    model_name    = None
    repository    = None
        

class ModelFinal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def make(self):
        gc.collect()
        logging.info(f'loading model {Config.model_name} from {os.path.join(Config.model_dir, Config.repository)}')
        self.resnet    = torch.hub.load(os.path.join(Config.model_dir, Config.repository), 
                                        Config.model_name, source='local', pretrained=False)
        num_ftrs       = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, 1)
        self.resnet    = self.resnet.to(Config.DEVICE)
        logging.info(f'model {Config.model_name} loaded successfully')

    def train(self, X_cuda, y_cuda):
        out = self(X_cuda)
        return torch.nn.functional.binary_cross_entropy(out, y_cuda)
    
    def fit(self, train_loader, val_loader=None, epochs=Config.epochs,
            lr=Config.learning_rate):
        logging.info('============== Training Initiated ==============')
        logging.info(f'Total Epochs: {epochs}, Train Size: {len(train_loader)}, Validataion Size: {len(val_loader)}, Batch Size: {Config.batch_size}')
        opt = torch.optim.Adam(self.parameters(), lr=Config.learning_rate)
        for epoch in range(epochs):
            total_avg_loss = 0
            total_loss = 0
            iterator_loader = tqdm.tqdm(train_loader, desc='Train Batch',
                                        total=len(train_loader))
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
                iterator_loader.set_postfix({'total batch loss':loss.item(), 
                                             'total avg loss': total_avg_loss})
            print()
            self.validate(val_loader)
            gc.collect()
        logging.info('============== Training Ended ==============')
    def validate(self,val_loader):
        logging.info('--------- Validation Initiated ---------')
        average_val_score = 0
        average_f1_score = 0
        total_f1 = 0
        total_score = 0
        iterator_loader = tqdm.tqdm(val_loader, desc='Val Batch', 
                                    total=len(val_loader))
        
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
            
            iterator_loader.set_postfix({'average f1 score': average_f1_score,
                                         'average accuracy score': average_val_score})
        logging.info(f'Average Val Score: {average_val_score}, Average F1 Score: {average_f1_score}')
        logging.info('--------- Validation Completed ---------')
        return average_val_score, average_f1_score
            
    def forward(self, X):
        model_output = self.resnet(X)
        softmax_output = torch.nn.Sigmoid()(model_output)
        return softmax_output
    
    def predict(self,X):
        with torch.no_grad():
            return self(X)


def main(config_path, params_path):

    train_dir, test_dir, model_dir, repository, model_name = get_data_paths(config_path=config_path)    
    logging.info(f'Train Dir {train_dir}\nTest Dir {test_dir}')
    params = read_yaml(params_path)
    
    Config.image_size    = (params['IMAGE_SIZE']['HEIGHT'], params['IMAGE_SIZE']['WIDTH'])
    Config.split         = (params['TRAIN_SPLIT'], params['TEST_SPLIT'])
    Config.epochs        = params['EPOCH']
    Config.batch_size    = params['BATCH_SIZE']
    Config.learning_rate = params['LEARNING_RATE']
    Config.random_state  = params['RANDOM_STATE']
    Config.train_folder  = train_dir
    Config.test_folder   = test_dir
    Config.model_dir     = model_dir
    Config.repository    = repository
    Config.model_name    = model_name
    Config.transforms    = transforms.Compose([transforms.Resize(size=Config.image_size),
                                               transforms.ToTensor()])
    torch.hub.set_dir(model_dir)
    
    data_loader = TrainTestLoader()
    train_loader, val_loader = data_loader.get_data(Config.train_folder, 
                                                    split_ratio=Config.split, 
                                                    split=True, batch_size=Config.batch_size, 
                                                    transforms=Config.transforms)
    logging.info(f'Training Size {len(train_loader)}\nValidataion Size {len(val_loader)}')
    
    model_attached = ModelFinal()
    model_attached.make()
    model_attached.fit(train_loader=train_loader, val_loader=val_loader, 
                       epochs=Config.epochs, lr=Config.learning_rate)        
    return

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