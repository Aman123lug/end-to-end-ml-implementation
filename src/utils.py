import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import sys
from src.exception import CustomErrorHandler
from src.logger import logging
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
import os


def evaluate_models(X_train, X_test, y_train, y_test, models:dict):
    report = {}
    
    for i in range(len(models)):
        model = list(models.values())[i]
        
        model.fit(X_train, y_train)
        
        ypred = model.predict(X_test)
        
        score = accuracy_score(ypred, y_test)
        report[model] = score
   
    
    return report
    
