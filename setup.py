from setuptools import find_packages, setup
from typing import List

HYPNE_E_DOT = '-e .'

def get_requires(file_path:str)-> List[str]:
    req = []
    with open(file_path) as f:
        req = f.readlines()
    
        req=[i.replace('\n','') for i in req]
            
        if HYPNE_E_DOT in req:
            req.remove(HYPNE_E_DOT)
            
        return req
    
    
    

    

setup(
    name="mlproject",
    version="0.1",
    author="amanprajapati",
    author_email="ak06465676@gmail.com",
    packages=find_packages(),
    install_requires=get_requires('requirements.txt'),
    
)