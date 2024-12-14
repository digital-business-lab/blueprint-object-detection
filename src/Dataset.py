from src.Config import ConfigYAML, load_env, ConfigPaths

import os

from roboflow import Roboflow


class Dataset(Roboflow, ConfigYAML, ConfigPaths):
    def __init__(self):
        Roboflow.__init__(self)
        ConfigYAML.__init__(self)
        ConfigPaths.__init__(self)

        self.workspace_name = self.config_data["dataset"]["workspace_name"]
        self.project_name = self.config_data["dataset"]["project_name"]
        self.project_version = self.config_data["dataset"]["project_version"]
        self.model = self.config_data["dataset"]["model"]


    def load_data(self, download_location: str):
        api_key = load_env()
        project = Roboflow(api_key=api_key).workspace(self.workspace_name).project(self.project_name)
        project.version(self.project_version).download(self.model, location=download_location, overwrite=False)
        self.__redefine_paths(dataset_yaml_path=f"{download_location}/data.yaml") 
    
 
    #-----------Private Methods-----------#
    def __redefine_paths(self, dataset_yaml_path: str):
        dir_path = os.getcwd()
        data = self.read_yaml(dataset_yaml_path)
        data["train"] = f"{dir_path}/data/{self.project_name}/train/images"
        data["val"] = f"{dir_path}/data/{self.project_name}/valid/images"
        data["test"] = f"{dir_path}/data/{self.project_name}/test/images"
        
        ConfigYAML.write_yaml(data=data, file_path=dataset_yaml_path)
    

if __name__ == "__main__":
    data = Dataset()
    data.load_data()
