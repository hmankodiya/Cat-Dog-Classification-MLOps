import  os
import torchmetrics
import yaml
import json

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
    

def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path,exist_ok=True)
        print(f'directory created at {dir_path}')



def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content

def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path,exist_ok=True)
        print(f'directory created at {dir_path}')
        
        
def save_local_df(data, data_path, index_status=False):
    data.to_csv(data_path, index=index_status)
    print(f'data is saved at {data_path}')

def save_reports(report: dict, report_path: str):
    with open(report_path,'w') as f:
        json.dump(report, f, indent=4)
    print(f'reports are save at {report_path}')