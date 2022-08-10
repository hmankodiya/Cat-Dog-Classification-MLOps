import argparse
import os
import logging
from src.utils.common import read_yaml
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder


STAGE = "02-featurization"  # <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)


class TrainTestLoader:
    def __init__(self):
        """ DataLoader Class """
        pass
    def get_data(self, data_folder, split_ratio=None, split=True, batch_size=None,
                 transforms=None):
        if split:
            """ Call method for quick data loading """
            self.train = ImageFolder(data_folder, transform=transforms)

            self.batch_size = batch_size
            self.split_ratio = split_ratio

            tr, ts = int(len(self.train)*split_ratio[0]), len(self.train)-int(len(self.train)*split_ratio[0])
            train_ds, val_ds = random_split(self.train, [tr, ts])

            self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

            return self.train_loader, self.val_loader
        else:

            self.test = ImageFolder(data_folder, transform=transforms)
            self.test_loader = DataLoader(self.test)

            return self.test_loader


def get_data_paths(config_path):
    # read config files
    config = read_yaml(config_path)

    artifacts_dir = config['artifacts']['artifacts_dir']

    data_dir = config['artifacts']['data_source']['data_dir']
    train_data = config['artifacts']['data_source']['train_folder']
    test_data = config['artifacts']['data_source']['test_folder']
    model_dir = config['artifacts']['data_source']['model_dir']
    model_name = config['artifacts']['model']['model_name']
    repository = config['artifacts']['model']['repo']        
    
    train_dir = os.path.join(data_dir, train_data)
    test_dir = os.path.join(data_dir, test_data)
    model_dir = os.path.join(artifacts_dir, model_dir, model_name)

    logging.info(f'Train Dir {train_dir}\nTest Dir {test_dir}\n Model Dir {model_dir}')

    return train_dir, test_dir, model_dir, repository, model_name


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        # get_data_paths(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
