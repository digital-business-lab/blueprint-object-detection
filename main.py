""" 
This script is the main module and starts the whole pipeline
    - Loading dataset
    - Loading configs
    - Starting training / prediction
    - Logging data

File is written in pylint standard
"""

import logging

from src.Dataset import Dataset
from src.Model import Model
from src.Config import ConfigPaths, ConfigYAML

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logs.log",
    format='%(levelname)s : %(name)s : %(asctime)s -> %(message)s <- %(filename)s : %(funcName)s',
    level=logging.INFO
)


def load_dataset() -> None:
    """
    Loads dataset

    Returns:
    -------
        None
    """
    Dataset().load_data(f"{ConfigPaths().folder_data()}/{Dataset().project_name}")

def model_mode_output(model: Model, confidence: float, mode: str):
    """
    Loads the model mode "train", "predict" or "train_predict" and executes
    accordingly

    Parameters:
    -----------
        model: Model
            -> Model which should be used
        confidence: float
            -> Confidence with which the model should predict
        mode: str
            -> Specific mode

    Returns:
    -------
        None
    """
    logger.info("Selected mode for model :'%s'", mode)
    results = None

    if mode == "train":
        results = model.train()

    elif mode == "predict":
        results = model.prediction(confidence=confidence)

    elif mode == "train_predict":
        model.train()
        results = model.prediction(confidence=confidence)

    else:
        results = ValueError(
            f"Wrong input! Select 'train', 'predict' or 'train_predict'! Your input was: '{mode}'"
            )
        logger.critical(results)

    return results

def get_model_mode() -> str:
    """
    Loads mode from config

    Returns:
    -------
        str
    """
    data: dict = ConfigYAML().config_data
    return data["model"]["modelMode"], data["model"]["modelConfidence"]


if __name__ == "__main__":
    load_dataset()

    MODEL = Model()
    MODE, CONFIDENCE = get_model_mode()

    results_model = model_mode_output(model=MODEL,
                        confidence=CONFIDENCE,
                        mode=MODE)
