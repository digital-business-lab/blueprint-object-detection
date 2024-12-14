import os
from pathlib import Path
from dotenv import load_dotenv

import yaml


def load_env(file_path: str =".env"):
    load_dotenv(Path(file_path))
    roboflow_api = os.getenv("ROBOFLOW_API_KEY")
    return roboflow_api


class ConfigPaths:
    def __init__(self):
        self.defaultpaths = os.listdir()
    
    def folder_config(self) -> str:
        """
        Path to the folder 'pages'

        return: str
        """
        if "config" in self.defaultpaths:
            config_folder = os.path.join(".", "config")
        else:
            config_folder = os.path.join("..", "config")
        return self.__path_checker(config_folder)
    
    def folder_data(self) -> str:
        """
        Path to the folder 'pages'

        return: str
        """
        if "data" in self.defaultpaths:
            folder_data = os.path.join(".", "data")
        else:
            folder_data = os.path.join("..", "data")
        return self.__path_checker(folder_data)
    
    def folder_model(self) -> str:
        """
        Path to the folder 'pages'

        return: str
        """
        if "model" in self.defaultpaths:
            folder_model = os.path.join(".", "model")
        else:
            folder_model = os.path.join("..", "model")
        return self.__path_checker(folder_model)
    
    def folder_src(self) -> str:
        """
        Path to the folder 'pages'

        return: str
        """
        if "src" in self.defaultpaths:
            folder_src = os.path.join(".", "src")
        else:
            folder_src = os.path.join("..", "src")
        return self.__path_checker(folder_src)
    
    #----------Private Methods----------#
    def __path_checker(self, path: str) -> str:
        """
        Checks if the path exists
        Parameters:
          path: str

        return: str
        """
        return path if os.path.exists(path) else self.__logger.error("Path does not exist.")


class ConfigYAML:
    def __init__(self):
        self.config_data = self.read_yaml()

    def read_yaml(self, file_path: str ="./config/config.yaml"):
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return data
    
    @staticmethod
    def write_yaml(data: dict, file_path: str ="./config/config.yaml"):
        yaml_str = yaml.safe_dump(data, sort_keys=False)
        with open(file_path, "w") as file:
            file.write(yaml_str)





if __name__ == "__main__":
    print(ConfigPaths().folder_config())
