import pandas as pd
import sys
from src.exception import CustomErrorHandler
from src.logger import logging
from sklearn.metrics import accuracy_score
import dill
def evaluate_models(X_train, X_test, y_train, y_test, models:dict):
    report = {}
    
    for i in range(len(models)):
        model = list(models.values())[i]
        
        model.fit(X_train, y_train)
        
        ypred = model.predict(X_test)
        
        score = accuracy_score(ypred, y_test)
        report[model] = score
   
    
    return report
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomErrorHandler(e, sys)
    
    
