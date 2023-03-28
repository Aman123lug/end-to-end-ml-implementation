import sys
import pandas as pd
from src.exception import CustomErrorHandler
from src.utils import load_object
from src.logger import logging
from src.components.data_transformation import DataTranformForTest


class CustomData:
    def __init__(self) -> None:
        pass
    
    def predict_pipeline(self, features):
        model_path = "saved_ml_model/model.pkl"
        
        model = load_object(model_path)
        logging.info("model loaded!")
        process_data = DataTranformForTest(features)
        logging.info("preprocssing step is done for testing data")
        
        prediction = model.predict(process_data)
        logging.info("prediction done!")
        
        
        return prediction
    
    
if __name__ == "__main__":
    obj = CustomData()
    yes = obj.predict_pipeline("you won a watch click here for deal !")
    print(yes)