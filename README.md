# Burkina Faso Rainfall Prediction
Overview

This repository contains code for predicting rainfall in Burkina Faso using geospatial data collected from four satellites. The project aims to provide accurate rainfall predictions for 143 locations in Burkina Faso and classify the intensity of rainfall into categories.
## Data Collection

The geospatial rainfall data is collected from four satellites covering Burkina Faso. The dataset includes information about rainfall patterns, dates, and geographical coordinates of 143 locations across the country.
### Model Architecture

The rainfall prediction model is based on a Spatial Convolutional Neural Network (ConvNet). The model takes as input various features such as geographical coordinates, date, and satellite data to predict rainfall levels. The model architecture consists of multiple convolutional layers followed by max-pooling layers to capture spatial patterns in the data.
### Training

The model is trained using historical rainfall data from the satellites. The training process involves optimizing the model parameters to minimize the mean squared error between predicted and actual rainfall values. Training is performed using the Adam optimizer and Mean Squared Error (MSE) loss function.
#### Prediction

Once trained, the model can be used to make predictions for future rainfall. Given input features such as date and geographical coordinates, the model predicts the rainfall levels for the specified locations. Additionally, the predicted rainfall values are classified into categories representing different intensity levels, such as no rain, light rain, storm, and flood.
Requirements



Install these dependencies using pip install -r requirements.txt.
Usage

To use the prediction app, run the app.py script using Streamlit. Enter the latitude, longitude, and name of the station for which you want to predict rainfall. The app will display the predicted rainfall level and intensity category for the specified location.
Contributors

    <I>Maximilien Kpizingui<br>
   <I> Arnaud Kima
