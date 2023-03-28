import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import sys
from src.components.data_transformation import getDoc
from src.exception import CustomErrorHandler
from src.logger import logging
from src.utils import evaluate_models
from dataclasses import dataclass
import os
from sklearn.metrics import accuracy_score
import pickle
import warnings


@dataclass
class Model_path:
    model_path = os.path.join("saved_ml_model","model.pkl")
    train_data_path = "notebooks/data/train.csv"
    test_data_path = "notebooks/data/test.csv"
    
    
class Model_trainings:
    def __init__(self):
        self.model_path = Model_path()
        
    def re_training(self, train_path, test_path):
        data = pd.read_csv(train_path)
        logging.info("data read successfully!")
        cv = CountVectorizer()

        data.rename(columns={"0":"text","v1":"label"}, inplace=True)
        data = data[["text", "label"]]
        logging.info("data column rename successfully!")

        X = getDoc(data["text"])
        new_x = cv.fit_transform(X)
        new_X_train = new_x.toarray()
        logging.info("data preprocessing step completed successfully!")
       
        
        y = data[["label"]]
        lb = LabelEncoder()
        y_train = lb.fit_transform(y)
       
        X_train, X_test, y_train, y_test = train_test_split(new_X_train, y_train, test_size=0.33, random_state=42)
        logging.info("data train test split successfully!")
        
        models = {
                "LogisticRegression": LogisticRegression(),
                "MultinomialNB": MultinomialNB(),
                "Guassian": GaussianNB(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "XGBclassifier": XGBClassifier(),
                "Kneighbors": KNeighborsClassifier()
                
            }
        
       
            
        all_score= evaluate_models(X_train, X_test, y_train, y_test, models)
        logging.info("model training successfully!")
        print(all_score)
        
         
if __name__ == "__main__":
    obj = Model_trainings()
    # train = obj.model_path.train_data_path
    # test = obj.model_path.test_data_path
    obj.re_training("notebooks\\data\\train.csv", "notebooks\data\train.csv")
  