import numpy as np
import pandas as pd
import math
from dash import Dash
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date
import random
import tensorflow as tf
import pytz
from dash import Output

app = Dash(__name__)

# Diese Funktion fügt zyklische Merkmale zu einem DataFrame hinzu.
    # Sie transformiert den Monat in zwei neue Features: sin und cos,
    # um die zyklische Natur der Monate zu repräsentieren.
    # Diese Transformationen erlauben es, den periodischen Charakter der Daten beizubehalten.
    # Berechnet den Sinus des Monats, um die zyklische Natur der Daten zu erfassen.
    # Die Formel (2 * π * Monat / 12) normalisiert den Monatswert auf den Bereich [0, 2π],
    # da ein kompletter Zyklus (ein Jahr) 12 Monate hat.
def add_cyclic_features(df):
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    # Berechnet den Kosinus des Monats, um die zyklische Natur der Daten zu erfassen.
    # Dies ergänzt das Sinus-Feature und hilft, die Periodizität der Monate zu modellieren.
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    return df


def get_lstm_predictions(flight_Abflug, flight_Ankunft):
    # Setzen vom Seed
    seed = 45
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Laden und Filtern der Daten
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df.reset_index(drop=True)

    # Konvertiere YearMonth in Datetime und sortiere die Daten nach Datum
    df['Date'] = pd.to_datetime(df['YearMonth'], format='%Y%m')
    df = df.sort_values(by='Date')

    # Zyklische Features hinzufügen (Funktion muss definiert sein)
    df = add_cyclic_features(df)

    # Initialisierung des MinMaxScalers
    scaler = MinMaxScaler(feature_range=(0, 1))
    Real_Value = df["$Real"].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(Real_Value)
    training_data_length = math.ceil(len(Real_Value) * 0.8)
    lookback_window = 12

    # Erstellung der Trainingsdaten
    Xtrain, Ytrain = [], []
    for i in range(lookback_window, training_data_length):
        Xtrain.append(scaled_data[i - lookback_window:i, 0])
        Ytrain.append(scaled_data[i, 0])
    Xtrain, Ytrain = np.array(Xtrain), np.array(Ytrain)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))

    # Modell erstellen und trainieren
    model = Sequential()
    neurons = 128
    model.add(LSTM(neurons, return_sequences=True, input_shape=(Xtrain.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(neurons, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(neurons, return_sequences=False))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtrain, Ytrain, epochs=200, batch_size=35)

    # Vorhersagen erstellen
    last_sequence = scaled_data[-lookback_window:]
    predictions = []
    data_points = 12

    for _ in range(data_points):
        input_data = np.reshape(last_sequence, (1, lookback_window, 1))
        prediction = model.predict(input_data)
        predictions.append(prediction[0])
        last_sequence = np.append(last_sequence[1:], prediction[0])
    predictions = scaler.inverse_transform(predictions)

    # Berechnung der tatsächlichen Testwerte
    Xtest, Ytest = [], []
    for i in range(training_data_length, len(scaled_data)):
        Xtest.append(scaled_data[i - lookback_window:i, 0])
        Ytest.append(scaled_data[i, 0])
    Xtest, Ytest = np.array(Xtest), np.array(Ytest)
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))
    test_predictions = model.predict(Xtest)
    test_predictions = scaler.inverse_transform(test_predictions)

    # Berechnung der Fehlermaße
    Ytest = scaler.inverse_transform([Ytest])
    Ytest = np.reshape(Ytest, (-1,))
    mae = round(mean_absolute_error(Ytest, test_predictions), 2)
    mse = round(mean_squared_error(Ytest, test_predictions), 2)
    rmse = round(np.sqrt(mse), 2)

    # Erstellen der Grafik
    fig = go.Figure()
    x_values = df['Date']
    y_values = df["$Real"]
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", name="Historische Daten"))
    fig.add_trace(go.Scatter(x=pd.date_range(start=x_values.iloc[-1], periods=data_points + 1, freq="M")[1:], 
                             y=predictions.flatten(), mode='lines+markers', name="Vorhersagen", line=dict(color='yellow')))
    fig.add_trace(go.Scatter(x=[x_values.iloc[-1], pd.date_range(start=x_values.iloc[-1], periods=data_points + 1, freq="M")[1]],
                             y=[y_values.iloc[-1], predictions.flatten()[0]], mode="lines", showlegend=False, line=dict(color='yellow', dash='dash')))
    yearly_avg = df.groupby(df['Date'].dt.year)['$Real'].mean().reset_index()
    fig.add_trace(go.Scatter(x=yearly_avg['Date'], y=yearly_avg['$Real'], mode="lines+markers", name="Jährl. Durchschnittspreis", line=dict(color='orange')))
    fig.update_layout(template="plotly_dark", height=600)
    fig.update_xaxes(title="Jahr")
    fig.update_yaxes(title="Preis ($)")
    fig.update_layout(title="LSTM-Prognose für die Strecke: {} & {}".format(flight_Abflug, flight_Ankunft))

    # Manuelles Hinzufügen der Metriken in die Legende
    metrics_table = f"LSTM<b>Metriken</b><br>MSE: {mse:.2f}<br>MAE: {mae:.2f}<br>RMSE: {rmse:.2f}"
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='rgba(0,0,0,0)'),
        showlegend=True,
        name=metrics_table,
        hoverinfo='none'
    ))

    fig.update_layout(
        legend=dict(
            x=0.99, y=0.99,
            traceorder='normal',
            font=dict(size=12, color="white"),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
            
        )
    )

    return fig

# Beispielaufruf der Funktion
#flight_Abflug = "Cairns"
#flight_Ankunft = "Melbourne"
#predictions, rmse, mae = get_lstm_predictions(flight_Abflug, flight_Ankunft)
#print(f"Vorhersagen: {predictions}")





if __name__ == "__main__":
    app.run(debug = True)




  