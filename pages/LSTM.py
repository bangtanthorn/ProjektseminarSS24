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
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route","$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df.reset_index(drop=True)

    # Konvertiere YearMonth in Datetime und sortiere die Daten nach Datum
    df['Date'] = pd.to_datetime(df['YearMonth'], format='%Y%m')
    df = df.sort_values(by='Date')

    # Zyklische Features hinzufügen
    df = add_cyclic_features(df)

    # Überprüfen, ob genügend Daten vorhanden sind
    #if len(df) < 50:
        #print("Zu wenige Datenpunkte für das Training.")
        #return [], 0, 0

    # Initialisierung des MinMaxScalers
    # Der MinMaxScaler skaliert die Daten in den Bereich [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Extrahiere die Spalte "$Real" aus dem DataFrame und forme sie zu einem 2D-Array um
    Real_Value = df["$Real"].values.reshape(-1, 1)
    # Wende den MinMaxScaler auf die Daten an, um sie zu skalieren
    scaled_data = scaler.fit_transform(Real_Value)
    # Bestimme die Länge der Trainingsdaten, indem 80% der gesamten Daten verwendet werden
    training_data_length = math.ceil(len(Real_Value) * 0.8)
    # Definiere das Fenster, das verwendet wird, um in die Vergangenheit zu schauen (Lookback-Fenster)
    # Hier wird ein Fenster von 12 Monaten verwendet
    lookback_window = 12


    # Erstellung der Trainingsdaten
    Xtrain, Ytrain = [], []
    # Schleife über die Daten, um Trainingsbeispiele zu erstellen
    for i in range(lookback_window, training_data_length):
        # Füge eine Sequenz von "lookback_window" Datenpunkten zu Xtrain hinzu
        Xtrain.append(scaled_data[i - lookback_window:i, 0])
        # Füge den nächsten Datenpunkt zu Ytrain hinzu
        Ytrain.append(scaled_data[i, 0])

    # Konvertiere Listen in numpy-Arrays
    Xtrain, Ytrain = np.array(Xtrain), np.array(Ytrain)

    # Reshape Xtrain, um das Format [Anzahl der Beispiele, Sequenzlänge, 1 Feature] zu haben
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))

    # Modell erstellen und trainieren
    model = Sequential()
    neurons = 128
    model.add(LSTM(neurons, return_sequences=True, input_shape=(Xtrain.shape[1], 1)))
    model.add(Dropout(0.2))  # Dropout zur Regularisierung
    model.add(LSTM(neurons, return_sequences=True))
    model.add(Dropout(0.2))  # Dropout zur Regularisierung
    model.add(LSTM(neurons, return_sequences=False))
    model.add(Dense(25, activation='relu'))  # ReLU in der versteckten Schicht
    model.add(Dense(1, activation='linear'))  # Linear in der Ausgabeschicht
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xtrain, Ytrain, epochs=200, batch_size=35)

    # Vorhersagen erstellen
    last_sequence = scaled_data[-lookback_window:]
    predictions = []
    data_points = 5  # Anzahl der Datenpunkte, die vorhergesagt werden sollen



    # Schleife, die data_points-mal ausgeführt wird, um Vorhersagen zu erstellen
    for _ in range(data_points):
        # Formatiere die last_sequence in das Format, das das LSTM-Modell erwartet:
        # (Batch-Größe, Sequenzlänge, Anzahl der Features)
        input_data = np.reshape(last_sequence, (1, lookback_window, 1))
        # Verwende das trainierte LSTM-Modell, um eine Vorhersage basierend auf input_data zu erstellen
        prediction = model.predict(input_data)
        # Fügt die Vorhersage für den nächsten Datenpunkt der Liste predictions hinzu
        # prediction[0] ist erforderlich, da model.predict ein Array zurückgibt
        predictions.append(prediction[0])
        # Aktualisiere last_sequence, um die neueste Vorhersage zu enthalten
        # last_sequence[1:] entfernt den ersten Wert der aktuellen Sequenz,
        # wodurch Platz für die neue Vorhersage geschaffen wird
        # np.append(..., prediction[0]) fügt die neue Vorhersage an das Ende der Sequenz an
        last_sequence = np.append(last_sequence[1:], prediction[0])
        # Skaliere die Vorhersagen zurück auf den ursprünglichen Wertebereich
        # scaler.inverse_transform nimmt die Liste predictions, die im skalierten Wertebereich liegt,
        # und transformiert sie zurück in den ursprünglichen Bereich
    predictions = scaler.inverse_transform(predictions)


    # Initialisierung von Xtest und Ytest
    Xtest, Ytest = [], []
    # Schleife, die von training_data_length bis zum Ende der skalierten Daten läuft
    # Dies stellt sicher, dass die Testdaten aus den letzten 20% der Daten bestehen
    for i in range(training_data_length, len(scaled_data)):
        # Füge die Sequenz der Länge lookback_window zur Liste der Eingabedaten (Xtest) hinzu
        Xtest.append(scaled_data[i - lookback_window:i, 0])
        # Füge den tatsächlichen Wert, der vorhergesagt werden soll, zur Liste der Zielwerte (Ytest) hinzu
        Ytest.append(scaled_data[i, 0])


    # Konvertiere die Listen Xtest und Ytest in numpy Arrays
    Xtest, Ytest = np.array(Xtest), np.array(Ytest)
    # Reshape Xtest in das Format, das das LSTM-Modell erwartet:
    # (Anzahl der Testbeispiele, Sequenzlänge, Anzahl der Features)
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))
    # Verwende das trainierte LSTM-Modell, um Vorhersagen für die Testdaten zu erstellen
    test_predictions = model.predict(Xtest)
    # Skaliere die Vorhersagen zurück auf den ursprünglichen Wertebereich
    # scaler.inverse_transform nimmt die Liste test_predictions, die im skalierten Wertebereich liegt,
    # und transformiert sie zurück in den ursprünglichen Bereich
    test_predictions = scaler.inverse_transform(test_predictions)



    # Berechnung der tatsächlichen Testwerte
    Ytest = scaler.inverse_transform([Ytest])
    Ytest = np.reshape(Ytest, (-1,))

    # Berechnung der Fehlermaße
    mae = round(mean_absolute_error(Ytest, test_predictions), 2)
    mse = round(mean_squared_error(Ytest, test_predictions), 2)
    rmse = round(np.sqrt(mse), 2)
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")


    # Grafik erstellen
    fig = go.Figure()
    # x- und y-Werte für die historischen Daten
    x_values = df['Date']
    y_values = df["$Real"]
    # Linienplot für die historischen Daten hinzufügen
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", name="Historical Data"))
    # Markierungsplot für die Vorhersagen hinzufügen
    fig.add_trace(go.Scatter(x=pd.date_range(start=x_values.iloc[-1], periods=data_points + 1, freq="M")[1:], 
                             y=predictions.flatten(), mode="markers", name="Predictions"))
    # Linienplot hinzufügen, um die Lücken zwischen den historischen Daten und den Vorhersagen zu schließen
    fig.add_trace(go.Scatter(x=[x_values.iloc[-1], pd.date_range(start=x_values.iloc[-1], periods=data_points + 1, freq="M")[1]],
                             y=[y_values.iloc[-1], predictions.flatten()[0]], mode="lines", name="Combined Line", showlegend=False, line=dict(color='#4169E1')))
    # Durchschnittspreis pro Jahr berechnen
    yearly_avg = df.groupby(df['Date'].dt.year)['$Real'].mean().reset_index()
    # Linienplot für den jährlichen Durchschnittspreis hinzufügen
    fig.add_trace(go.Scatter(x=yearly_avg['Date'], y=yearly_avg['$Real'], mode="lines+markers", name="Yearly Avg Price", line=dict(color='orange')))
    # Das Layout der Grafik anpassen
    fig.update_layout(template="plotly_dark", height=600)
    # Die Achsenbeschriftungen für das Datum und den Kurs festlegen
    fig.update_xaxes(title="Year")  
    fig.update_yaxes(title="$Real")  

    fig.update_layout(title="LSTM-Prognose für die Strecke: {} & {}".format(flight_Abflug, flight_Ankunft))


    return fig


# Beispielaufruf der Funktion
#flight_Abflug = "Cairns"
#flight_Ankunft = "Melbourne"
#predictions, rmse, mae = get_lstm_predictions(flight_Abflug, flight_Ankunft)
#print(f"Vorhersagen: {predictions}")





if __name__ == "__main__":
    app.run(debug = True)




  