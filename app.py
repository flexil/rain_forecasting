import streamlit as st
import torch
import pandas as pd


model = torch.load('trained_model.pth')

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
