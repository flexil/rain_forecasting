import streamlit as st
import torch
import pandas as pd


model = torch.load('trained_model.pth')


def preprocess_input(latitude, longitude, station_name):

    station_data = [0] * len(station_columns)
    station_data[station_columns.index('station_' + station_name)] = 1
    latlong_data = [latitude, longitude]
    return station_data, latlong_data

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

st.title('Rainfall Prediction App in 143 region in Burkina Faso')

image = 'location-burkina.png'  
st.image(image, caption='143 location in Burkina collected from 4 satellites', use_column_width=True)

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
