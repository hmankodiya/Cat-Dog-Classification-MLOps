import argparse
import os
import logging
import torch
from torchvision.transforms import transforms
from src.utils.common import create_directory, read_yaml, save_json
from src.stage_03_train import ModelFinal
from src.stage_02_featurization import TrainTestLoader

STAGE = "Evaluate" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    checkpoint = config['checkpoint']['checkpoint_name']

    checkpoint_path = os.path.join(config['artifacts']['artifacts_dir'], 
                 config['artifacts']['model']['model_weights'],
                 config['artifacts']['model']['model_name'],
                 checkpoint)
    
    logging.info('============== Testing Initiated ==============')
    
    test_folder = os.path.join(config['artifacts']['data_source']['data_dir'],
                               config['artifacts']['data_source']['test_folder'])
    
    logging.info(f'Test set: {test_folder}')
    image_size = (params['IMAGE_SIZE']['WIDTH'], params['IMAGE_SIZE']['HEIGHT'])    
    
    data_loader = TrainTestLoader()
    test_loader = data_loader.get_data(test_folder, split=False, 
                                        transforms=transforms.Compose(
                                            [transforms.Resize(size=image_size),
                                            transforms.ToTensor()])
                                        )
    logging.info(f'Test set loaded successfully')
    
    logging.info(f'Loading Checkpoint: {checkpoint} of model: {config["artifacts"]["model"]["model_name"]} from {checkpoint_path}')
    model = ModelFinal()
    model = torch.load(checkpoint_path)
    # model.load_state_dict(torch.load(checkpoint_path))
    logging.info('Checkpoint {checkpoint} loaded successfully')

    test_average_score, test_average_f1_score = model.validate(test_loader)

    save_json(os.path.join(config['metrics']['metrics_dir'], config['metrics']['test_f1']), {
                               config['metrics']['test_f1'][:-5]: test_average_f1_score
                                   })
    
    save_json(os.path.join(config['metrics']['metrics_dir'], config['metrics']['test_score']), {
                               config['metrics']['test_score'][:-5]: test_average_score
                                   })
    
    logging.info('============== Testing Ended ==============')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config,
             params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e