import argparse
import os
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directory
import torch

STAGE = "01-prepare-model" ## <<< change stage name 

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
    
    artifacts_dir = config['artifacts']['artifacts_dir']
    
    model_dir     = config['artifacts']['model']['model_dir']
    model_name    = config['artifacts']['model']['model_name']
    model_dir = os.path.join(artifacts_dir, model_dir, model_name)
    create_directory([model_dir])
    torch.hub.set_dir(model_dir) 
    logging.info("model loading initializing")
    
    torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=False)
    logging.info(f"model {model_name} loaded successfully at {model_dir}")


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