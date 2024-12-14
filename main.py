from src.Dataset import Dataset
from src.Model import Model
from src.Config import *


def load_dataset():
    Dataset().load_data(f"{ConfigPaths().folder_data()}/{Dataset().project_name}")

def model_mode_output(model: Model, file_paths: list, confidence: float =0.1, mode: str ="predict"):
    """Trains, Predicts or Trains + Predicts given on the input"""

    if mode == "train":
        model.train()

    elif mode == "predict":
        return model.predict(file_paths=file_paths, confidence=confidence)
    
    elif mode == "train_predict":
        model.train()
        return model.predict(file_paths=file_paths, confidence=confidence)
    
    else:
        raise ValueError(f"Input for mode has to be 'train', 'predict' or 'train_predict'! Your input was: '{mode}'")
    
def get_model_mode():
    data = ConfigYAML().config_data
    return data["modelMode"], data["modelConfidence"]
    


if __name__ == "__main__":
    load_dataset()

    MODEL = Model()
    MODE, CONFIDENCE = get_model_mode()

    model_mode_output(model=MODEL,
                      file_paths=["test123.jpg", "345test.jpg"],
                      confidence=CONFIDENCE,
                      mode=MODE)
