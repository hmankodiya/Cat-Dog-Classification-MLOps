import argparse
import os
import logging
import torch
from torchvision.transforms import transforms
from src.utils.common import create_directory, read_yaml, save_json
from src.stage_02_featurization import TrainTestLoader, get_data_paths
from src.utils.metrics import Metrics
import tqdm
import gc

STAGE = "03-model-training" ## <<< change stage name 

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
    model_path    = None
        

class ModelFinal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def make(self):
        gc.collect()
        logging.info(f'loading model {Config.model_name} from {os.path.join(Config.model_dir, Config.repository)}')
        self.resnet    = torch.hub.load(os.path.join(Config.model_dir, Config.repository), 
                                        Config.model_name, source='local')
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
        train_itr_loss = []
        train_batch_loss = []
        validation_metrics = []
        for epoch in range(epochs):
            print(f'+++++++ EPOCH {epoch+1} +++++++')
            avg_batch_loss = 0
            iterator_loader = tqdm.tqdm(train_loader, desc='Train Batch',
                                        total=len(train_loader))
            for iteration, batch in enumerate(iterator_loader):
                X_cuda = batch[0].to(dtype=torch.float32, device=Config.DEVICE)
                y_cuda = batch[1].to(dtype=torch.float32, device=Config.DEVICE).unsqueeze(1)
                loss = self.train(X_cuda, y_cuda)
                avg_batch_loss += loss.item()
                train_itr_loss.append({'Iteration':epoch*len(train_loader)+iteration+1,
                                       'Iteration Loss': loss.item()})
                loss.backward()
                opt.step()
                opt.zero_grad()
                torch.cuda.empty_cache()
                iterator_loader.set_postfix({'iteration loss':loss.item()})
                
            avg_batch_loss = avg_batch_loss/len(train_loader)
            logging.info(f'avg batch loss: {avg_batch_loss}')
            average_val_score, average_f1_score = self.validate(val_loader)
            train_batch_loss.append({'Epoch': epoch+1,
                                     'Batch Loss': avg_batch_loss})
            validation_metrics.append({'Epoch': epoch+1, 'Accuracy': average_val_score, 
                                       'F1-Score':average_f1_score})
            gc.collect()
        
        logging.info('============== Training Ended ==============')
        return train_itr_loss, train_batch_loss, validation_metrics 
    
    def validate(self, loader):    
        logging.info('--------- Validation Initiated ---------')
        total_score = 0
        total_f1 = 0
        iterator_loader = tqdm.tqdm(loader, desc='Val Batch', 
                                    total=len(loader))
        
        for iteration,batch in enumerate(iterator_loader):
            
            X_val_cuda = batch[0].to(dtype=torch.float32, device=Config.DEVICE)
            y_val = torch.unsqueeze(batch[1].to(dtype=torch.int32), dim=1)
            predictions  = self.predict(X_val_cuda).to('cpu')
            
            accuracy = Metrics.score(predictions, y_val)
            total_score += accuracy.item()
            
            f1 = Metrics.f1(predictions,y_val)
            total_f1 += f1.item()
            
        
        average_val_score = total_score/len(loader)
        average_f1_score = total_f1/len(loader)
        
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

    train_dir, test_dir, model_dir, repository, model_name, weights_dir = get_data_paths(config_path=config_path)    
    logging.info(f'Train Dir {train_dir}\nTest Dir {test_dir}')
    
    config = read_yaml(config_path)
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
    Config.model_saved   = weights_dir
    
    
    
    torch.hub.set_dir(model_dir)
    
    data_loader = TrainTestLoader()
    train_loader, val_loader = data_loader.get_data(Config.train_folder, 
                                                    split_ratio=Config.split, 
                                                    split=True, batch_size=Config.batch_size, 
                                                    transforms=Config.transforms)
    logging.info(f'Training Size {len(train_loader)}\nValidataion Size {len(val_loader)}')
    
    model_attached = ModelFinal()
    model_attached.make()
    train_itr_loss, train_batch_loss, validation_metrics = model_attached.fit(train_loader=train_loader, val_loader=val_loader,
                       epochs=Config.epochs, lr=Config.learning_rate)
    
    create_directory([config['plots']['plots_dir']])
    
    itr_loss_name = config['plots']['itr_loss']
    avg_batch_loss_name = config['plots']['avg_batch_loss']
    validation_metrics_name = config['plots']['validation_metrics']
    
    itr_loss_path = os.path.join(config['plots']['plots_dir'], itr_loss_name)
    avg_batch_loss_path = os.path.join(config['plots']['plots_dir'], avg_batch_loss_name)
    validation_metrics_path = os.path.join(config['plots']['plots_dir'], validation_metrics_name)
    
    save_json(itr_loss_path, {
        itr_loss_name[:-5]: train_itr_loss
    })
    save_json(avg_batch_loss_path, {
        avg_batch_loss_name[:-5]: train_batch_loss
    })
    save_json(validation_metrics_path, {
        validation_metrics_name[:-5]: validation_metrics
    })
    
    create_directory([weights_dir])
    saved_model_name = f'{Config.model_name}_batch-size-{Config.batch_size}_epochs-{Config.epochs}_learning_rate-{Config.learning_rate}.pth'
    logging.info(f'############## model saving ##############')
    torch.save(model_attached, os.path.join(weights_dir, saved_model_name))
    logging.info(f'model {saved_model_name} save successfully at {weights_dir}')
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
