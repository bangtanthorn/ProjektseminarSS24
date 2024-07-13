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



# Initialisiere die Dash-Anwendung mit dem aktuellen Modulnamen
app = Dash(__name__)  

# Diese Funktion fügt zyklische Merkmale zu einem DataFrame hinzu.
# Sie transformiert den Monat in zwei neue Features: sin und cos,
# um die zyklische Natur der Monate zu repräsentieren.
def add_cyclic_features(df):
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)  # Sinus des Monats
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)  # Kosinus des Monats
    return df


def get_lstm_predictions(flight_Abflug, flight_Ankunft, seed=45):
    # Setze den Seed für die Reproduzierbarkeit der Ergebnisse
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Lade und filtere die Daten aus der CSV-Datei
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df.reset_index(drop=True)

    # Konvertiere YearMonth in DateTime und sortiere nach Datum
    df['Date'] = pd.to_datetime(df['YearMonth'], format='%Y%m')
    df = df.sort_values(by='Date')

    # Füge zyklische Features hinzu
    df = add_cyclic_features(df)

    # Initialisiere den MinMaxScaler für die Datennormalisierung
    scaler = MinMaxScaler(feature_range=(0, 1))
    Real_Value = df["$Real"].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(Real_Value)
    training_data_length = math.ceil(len(Real_Value) * 0.8)
    lookback_window = 12

    # Erstelle die Trainingsdaten für das LSTM-Modell
    Xtrain, Ytrain = [], []
    for i in range(lookback_window, training_data_length):
        Xtrain.append(scaled_data[i - lookback_window:i, 0])
        Ytrain.append(scaled_data[i, 0])
    Xtrain, Ytrain = np.array(Xtrain), np.array(Ytrain)
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))

    # Erstelle das LSTM-Modell und trainiere es
    model = Sequential()  # Initialisiere ein sequentielles Modell
    neurons = 128  # Anzahl der Neuronen in den LSTM-Schichten
    model.add(LSTM(neurons, return_sequences=True, input_shape=(Xtrain.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(neurons, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(neurons, return_sequences=False))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtrain, Ytrain, epochs=200, batch_size=35)

    # Generiere Vorhersagen
    last_sequence = scaled_data[-lookback_window:]
    predictions = []
    data_points = 12

    for _ in range(data_points):
        input_data = np.reshape(last_sequence, (1, lookback_window, 1))
        prediction = model.predict(input_data)
        predictions.append(prediction[0])
        last_sequence = np.append(last_sequence[1:], prediction[0])
    predictions = scaler.inverse_transform(predictions)
    prediction = predictions.flatten()

    # Berechne die tatsächlichen Testwerte
    Xtest, Ytest = [], []
    for i in range(training_data_length, len(scaled_data)):
        Xtest.append(scaled_data[i - lookback_window:i, 0])
        Ytest.append(scaled_data[i, 0])
    Xtest, Ytest = np.array(Xtest), np.array(Ytest)
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))
    test_predictions = model.predict(Xtest)

    Ytest_scaled = Ytest
    test_predictions_scaled = model.predict(Xtest)

    # Berechne die Fehlermaße mit skalierten Werten
    mae = round(mean_absolute_error(Ytest_scaled, test_predictions_scaled), 2)
    mse = round(mean_squared_error(Ytest_scaled, test_predictions_scaled), 2)
    rmse = round(np.sqrt(mse), 2)

    # Normalisiere die Metriken
    max_mae = np.max(Ytest_scaled) - np.min(Ytest_scaled)
    max_mse = (np.max(Ytest_scaled) - np.min(Ytest_scaled)) ** 2
    max_rmse = np.sqrt(max_mse)

    normalized_mae_lstm = mae / max_mae if max_mae != 0 else 0
    normalized_mse_lstm = mse / max_mse if max_mse != 0 else 0
    normalized_rmse_lstm = rmse / max_rmse if max_rmse != 0 else 0

    # Erstelle das Diagramm für die Vorhersagen
    fig = go.Figure()
    x_values = df['Date']
    y_values = df["$Real"]
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", name="Historische Daten"))
    fig.add_trace(go.Scatter(x=pd.date_range(start=x_values.iloc[-1], periods=data_points + 1, freq="MS")[1:],
                             y=predictions.flatten(), mode='lines+markers', name="Prognose", line=dict(color='yellow')))
    fig.add_trace(go.Scatter(x=[x_values.iloc[-1], pd.date_range(start=x_values.iloc[-1], periods=data_points + 1, freq="MS")[1]],
                             y=[y_values.iloc[-1], predictions.flatten()[0]], mode="lines", showlegend=False,
                             line=dict(color='yellow', dash='dash')))
    yearly_avg = df.groupby(df['Date'].dt.year)['$Real'].mean().reset_index()
    fig.add_trace(go.Scatter(x=yearly_avg['Date'], y=yearly_avg['$Real'], mode="lines+markers",
                             name="Jährl. Durchschnittspreis", line=dict(color='orange')))
    fig.update_layout(template="plotly_dark", height=600)
    fig.update_xaxes(title="Jahr")
    fig.update_yaxes(title="Preis ($)")
    fig.update_layout(title=f"LSTM für die Strecke: {flight_Abflug} & {flight_Ankunft}")

    rounded_mae = round(normalized_mae_lstm, 2)
    rounded_mse = round(normalized_mse_lstm, 2)
    rounded_rmse = round(normalized_rmse_lstm, 2)

    # Füge die Metriken als Annotation zum Diagramm hinzu
    metrics_table = f"<b>Metriken</b><br>MSE: {rounded_mse}<br>MAE: {rounded_mae}<br>RMSE: {rounded_rmse}"
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
    print(predictions)

    return fig, rounded_mae, rounded_mse, rounded_rmse, predictions

# Hauptprogrammablauf
if __name__ == "__main__":
    app.run(debug=True)  # Starte die Dash-Anwendung im Debug-Modus


  