# FakeNewsDetection

## Overview
In this project, we have leverage natural language processing techniques and machine learning algorithms to classify fake news articles using sci-kit libraries from python.

# Setting up
These instructions will help you set the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites

Requirements for installation:

1. Python 3.7

This setup requires that your machine has python 3.7 installed on it. you can refer to this url https://www.python.org/downloads/ to download python. Once you have python downloaded and installed, you will need to setup PATH variables (if you want to run python program directly, detail instructions are below in how to run software section). To do that check this: https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/. Setting up PATH variable is optional as you can also run program without it and more instruction are given below on this topic.
    
2. Python libraries 

You will also need to download and install the following:
  - pandas
  - sklearn
  - textblob
  - imblearn
  - dash
  - plotly
  - numpy

# File Structure

The file structure is the following

src .
    |
    +-- main.py
    +-- modelling
    |   +-- optimization.py
    |   +-- predict.py
    |   +-- train_model.py
    +-- preparation
    |   +-- impoty_data.py
    +-- processing
    |   +-- data_cleaning.py
    |   +-- data_preparation.py
    |   +-- feature_extraction.py
    +-- ui
    |   +-- app.py
    |   +-- index.py
    |   +-- layout  .
                    |   +-- home.py 
                    |   +-- predict.py
                    |   +-- results.py

# File descriptions

## main.py
- Inputs: Taining Data File path, Test Data File Path
- Output: Trained models, Train Results

### impoty_data.py
- Imports data and cleans data before feature extraction

### feature_extraction.py
- Extracts text, stylistic and grammatic features

### data_preparation.py
- Perform feature selection and prepares data for modelling

### train_model.py
- Trains models and saves models into model flder

### predict.py
- Makes predictions on unseen data 

## index.py
- Creats a dash User interface that leverages saved models to give predictin
