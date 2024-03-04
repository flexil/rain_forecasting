import streamlit as st
import torch
import pandas as pd
import torch.nn as nn

# Define your model class
class SpatialConvNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SpatialConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        return x

# Load the model state dictionary
model_state_dict = torch.load('trained_model.pth')

# Instantiate your model class
input_channels = 75  # Update with the appropriate number of input channels
output_channels = 1   # Update with the appropriate number of output channels
model = SpatialConvNet(input_channels=input_channels, output_channels=output_channels)

# Load the model's state dictionary
model.load_state_dict(model_state_dict)
model.eval()  # Set the model to evaluation mode

station_columns = ['station_baguera', 'station_banfora_agri', 'station_baraboule', 'station_batie', 'station_bereba', 'station_beregadougou', 'station_betare', 'station_bogande', 'station_bousse', 'station_dedougou', 'station_diebougou', 'station_dionkele', 'station_fada_ngourma', 'station_gayeri', 'station_hounde', 'station_kamboince', 'station_koupela', 'station_leo', 'station_leri', 'station_mane', 'station_mogtedo', 'station_nobere', 'station_pabre', 'station_piela', 'station_po', 'station_pobe_mengao', 'station_reo_agri', 'station_safane', 'station_sapone', 'station_seguenega', 'station_sideradougou', 'station_soubakaniedougou', 'station_tanghin_dassouri', 'station_tikare', 'station_toeni', 'station_vallee_du_kou', 'station_zabre']

def preprocess_input(latitude, longitude, station_name):
    # Check if the station name exists in the list of station names
    if 'station_' + station_name in station_columns:
        station_data = [0] * len(station_columns)
        station_data[station_columns.index('station_' + station_name)] = 1
        latlong_data = [latitude, longitude]
        return station_data, latlong_data
    else:
        # Return None for station_data and latlong_data if station name is not found
        return None, None


def predict_rainfall(year, month, day, station_data, latlong_data):
    features = [year, month, day] + station_data + latlong_data
    features_tensor = torch.tensor([features], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(features_tensor).item()
    return prediction

def classify_rainfall_intensity(predicted_rainfall):
    if predicted_rainfall == 0:
        return 'No Rain'
    elif predicted_rainfall < 10:
        return 'Light Rain'
    elif predicted_rainfall < 50:
        return 'Storm'
    else:
        return 'Flood'

st.title('Rainfall Prediction in Burkina Faso')

image = 'location-burkina.png'  
st.image(image, caption='Rainfall historical data collected from 143 locations in Burkina collected between 2010 t0 2020  using  tamsat_rain,arc2_rain, chirps_rain and rfe_rain satellites', use_column_width=True)


year = st.number_input('Enter year', value=2021)
month = st.number_input('Enter month', min_value=1, max_value=12, value=1)
day = st.number_input('Enter day', min_value=1, max_value=31, value=1)

latitude = st.number_input('Enter latitude')
longitude = st.number_input('Enter longitude')
station_name = st.text_input('Enter station name')

station_data, latlong_data = preprocess_input(latitude, longitude, station_name)


if st.button('Predict Rainfall'):
    prediction = predict_rainfall(year, month, day, station_data, latlong_data)
    st.success(f'Predicted Rainfall: {prediction} mm')
    category = classify_rainfall_intensity(prediction)
    st.success(f'Rainfall Category: {category}')
