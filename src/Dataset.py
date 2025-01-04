""" 
Holds all the classes and functions for loading and processing datasets

File is written in pylint standard
"""

import os

from roboflow import Roboflow

from src.Config import ConfigYAML, load_env, ConfigPaths


class Dataset(Roboflow, ConfigYAML, ConfigPaths):
    """
    Class for loading a dataset from Roboflow
    ---------------------
    Methods:
        load_data
            Loads dataset from Roboflow

    Private Methods:
        __redefine_paths
            Redefines relative paths from data.yaml into absolute
            -> Dataset can not be loaded from YOLO otherwise
    """
    def __init__(self):
        Roboflow.__init__(self)
        ConfigYAML.__init__(self)
        ConfigPaths.__init__(self)

        self.roboflow = self.config_data["Roboflow"]
        self.workspace_name = self.config_data["dataset"]["workspace_name"]
        self.project_name = self.config_data["dataset"]["project_name"]
        self.project_version = self.config_data["dataset"]["project_version"]
        self.model = self.config_data["dataset"]["model"]


    def load_data(self, download_location: str) -> None:
        """
        Loads dataset from Roboflow

        Parameters:
        -----------
            download_location: str

        Returns:
        -------
            None
        """
        if self.roboflow == 0:
            api_key = load_env()
            project = Roboflow(api_key=api_key).workspace(
                self.workspace_name).project(self.project_name)
            project.version(self.project_version).download(
                self.model, location=download_location, overwrite=False
                )
            self.__redefine_paths(dataset_yaml_path=f"{download_location}/data.yaml")


    #-----------Private Methods-----------#
    def __redefine_paths(self, dataset_yaml_path: str) -> None:
        """
        Redefines relative paths from data.yaml into absolute
        -> Dataset can not be loaded from YOLO otherwise

        Parameters:
        -----------
            dataset_yaml_path: str

        Returns:
        -------
            None
        """
        dir_path = os.getcwd()
        data: dict = self.read_yaml(dataset_yaml_path)
        data["train"] = f"{dir_path}/data/{self.project_name}/train/images"
        data["val"] = f"{dir_path}/data/{self.project_name}/valid/images"
        data["test"] = f"{dir_path}/data/{self.project_name}/test/images"

        ConfigYAML.write_yaml(data=data, file_path=dataset_yaml_path)


if __name__ == "__main__":
    data_yaml = Dataset()
