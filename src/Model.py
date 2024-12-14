from src.Config import ConfigYAML, ConfigPaths

from ultralytics import YOLO


class Model(ConfigYAML, ConfigPaths):
    def __init__(self):
        ConfigYAML.__init__(self)
        ConfigPaths.__init__(self)
        self.model = YOLO(f"{self.folder_model()}/{self.config_data['modelName']}")

    def train(self):
        epochs = self.config_data["modelSpecific"]["epochs"]
        imgsz = self.config_data["modelSpecific"]["imgsz"]
        data = f"{self.folder_data()}/{self.config_data['dataset']['project_name']}/data.yaml"
        patience = self.config_data["modelSpecific"]["patience"]
        batch = self.config_data["modelSpecific"]["batch"]
        optimizer = self.config_data["modelSpecific"]["optimizer"]
        lr0 = self.config_data["modelSpecific"]["lr0"]
        lrf = self.config_data["modelSpecific"]["lrf"]
        momentum = self.config_data["modelSpecific"]["momentum"]
        weight_decay = self.config_data["modelSpecific"]["weight_decay"]
        dropout = self.config_data["modelSpecific"]["dropout"]
        project = self.config_data["modelSpecific"]["project"]

        results = self.model.train(data=data, epochs=epochs, imgsz=imgsz, patience=patience,
                                   batch=batch, optimizer=optimizer, lr0=lr0, lrf=lrf, 
                                   momentum=momentum, weight_decay=weight_decay, dropout=dropout,
                                   project=project)
        return results
    
    def predict(self, file_paths: list, confidence: float):
        if not isinstance(file_paths, list):
            raise TypeError(
                f"Variable 'file_paths' has to be a list of strings! You set an variable of type: '{type(file_paths)}'"
                )
        
        results = []
        for file_path in file_paths:
            result: list = self.model.predict(file_path, conf=confidence)
            results.append(result)
        return results



if __name__ == "__main__":
    yolo_model = Model()
    # yolo_model.train()
    # results = yolo_model.predict(["test_123.jpg"])
    # for result in results:
    #     result.show()