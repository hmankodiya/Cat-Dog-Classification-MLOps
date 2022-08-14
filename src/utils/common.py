import  os
import yaml
import json
import logging

def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path,exist_ok=True)
        print(f'directory created at {dir_path}')


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content
       
        
def save_local_df(data, data_path, index_status=False):
    data.to_csv(data_path, index=index_status)
    print(f'data is saved at {data_path}')

def save_reports(report: dict, report_path: str):
    with open(report_path,'w') as f:
        json.dump(report, f, indent=4)
    print(f'reports are save at {report_path}')
    

def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")