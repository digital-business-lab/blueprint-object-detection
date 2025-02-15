""" 
Holds class for model training and prediction
This file is all about the YOLO-Model and logging with mlflow

File is written in pylint standard
"""

import glob
import logging

import mlflow
from ultralytics import YOLO

from src.Config import ConfigYAML, ConfigPaths


class Model(ConfigYAML, ConfigPaths):
    """
    Class for model training and prediction
    ---------------------
    Methods:
        train
            Trains the model
        prediction
            Makes predictions with model

    """
    def __init__(self):
        ConfigYAML.__init__(self)
        ConfigPaths.__init__(self)
        self.model = YOLO(f"{self.folder_model()}/{self.config_data['model']['modelName']}")

        mlflow.set_tracking_uri(f"file:{self.folder_mlruns()}")
        mlflow.set_experiment(self.config_data["dataset"]["project_name"])

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized class 'Model'.")

    def train(self) -> None:
        """
        Trains the model
            - Start mlflow run
            - Load hyperparams
            - Log hyperparams
            - Save model

        Returns:
        -------
            None
        """
        self.logger.info("Started training.")
        with mlflow.start_run():
            epochs = self.config_data["modelParams"]["epochs"]
            imgsz = self.config_data["modelParams"]["imgsz"]
            data = f"{self.folder_data()}/{self.config_data['dataset']['project_name']}/data.yaml"
            patience = self.config_data["modelParams"]["patience"]
            batch = self.config_data["modelParams"]["batch"]
            optimizer = self.config_data["modelParams"]["optimizer"]
            lr0 = self.config_data["modelParams"]["lr0"]
            lrf = self.config_data["modelParams"]["lrf"]
            momentum = self.config_data["modelParams"]["momentum"]
            weight_decay = self.config_data["modelParams"]["weight_decay"]
            dropout = self.config_data["modelParams"]["dropout"]
            plots = self.config_data["modelParams"]["plots"]
            project = self.config_data["modelParams"]["project"]

            # Log hyperparams
            mlflow.log_params({
                "epochs": epochs,
                "imgsz": imgsz,
                "patience": patience,
                "batch": batch,
                "optimizer": optimizer,
                "lr0": lr0,
                "lrf": lrf,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "dropout": dropout
            })

            results = self.model.train(data=data, epochs=epochs, imgsz=imgsz, patience=patience,
                                    batch=batch, optimizer=optimizer, lr0=lr0, lrf=lrf,
                                    momentum=momentum, weight_decay=weight_decay, dropout=dropout,
                                    plots=plots, project=project)

            # Log training metrics if available
            metrics = {
                "best_mAP": getattr(results, "best_fitness", None),
                "loss": getattr(results, "loss", None),
                "precision": getattr(results, "precision", None),
                "recall": getattr(results, "recall", None)
            }
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})

            # Save the model
            model_path = f"{self.folder_model()}/{self.config_data['dataset']['model']}_trained.pt"
            self.model.save(model_path)
            mlflow.log_artifact(model_path)

            mlflow.end_run()

        self.logger.info("Finished training.")
        return results

    def prediction(self, confidence: float) -> list:
        """
        Makes predictions with model

        Parameters:
        -----------
            confidence: float
                -> With which confidence the model should make a prediction

        Returns:
        -------
            list
        """
        self.logger.info("Starting prediction with confidence: %d", confidence)
        save_path: str = self.config_data["datasetCustom"]["PredictionSavePath"]
        file_paths: str = self.config_data["datasetCustom"]["PredictionPath"]
        files = glob.glob(f"{file_paths}/*.{self.config_data['datasetCustom']['ImageType']}")

        results = []
        for idx, file in enumerate(files):
            result: list = self.model.predict(file, conf=confidence)
            result[0].save(f"{save_path}/result_{idx}.jpg")
            results.append(result)

            # Log artifact
            mlflow.log_artifact(f"{save_path}/result_{idx}.jpg")

        return results


if __name__ == "__main__":
    yolo_model = Model()
    yolo_model.prediction(confidence=0.75)
    # yolo_model.train()
    # results = yolo_model.predict(["test_123.jpg"])
    # for result in results:
    #     result.show()
    