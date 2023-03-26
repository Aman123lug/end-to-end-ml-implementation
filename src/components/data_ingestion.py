import os
import sys
from src.exception import CustomErrorHandler
from src.logger import logging
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataInjestionzConfig():
    train_data_path: str= os.path.join("artifacts", "train.csv")
    test_data_path: str= os.path.join("artifacts", "test.csv")
    raw_data_path: str= os.path.join("artifacts", "raw.csv")
    
    
class DataInjestion():
    def __init__(self):
        self.all_path = DataInjestionzConfig()
        
    
    def data_injestion(self):
        logging.info("entered the data injestion method")
        try:
            data = pd.read_csv('notebook/data/spamsms.csv')
            logging.info("folder created")
            os.makedirs(os.path.dirname(self.all_path.train_data_path), exist_ok=True)
            data.to_csv(self.all_path.raw_data_path, index=False, header=True)
            
            logging.info("train test split part")
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            train_data.to_csv(self.all_path.train_data_path,index=False, header=True)
            test_data.to_csv(self.all_path.train_data_path,index=False, header=True)
            
            return(
                self.all_path.train_data_path,
                self.all_path.test_data_path
            )
            
        except Exception as e:
            raise CustomErrorHandler(e, sys)
                    
        
            
        
        
if __name__ == "__main__":
    obj=DataInjestion()
    obj.data_injestion()
    print(obj.all_path)
    