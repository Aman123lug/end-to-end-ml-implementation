from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')

def index():
    return render_template("html_templates/index.html")


@app.route('/prediction', methods=["POST", "GET"]) 
def predict():
    if request.method == "GET":
        render_template("html_templates/index.html")
        
    else:
        pass
    
    

