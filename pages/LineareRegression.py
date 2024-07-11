#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime as dt
from datetime import date, datetime
import plotly.graph_objects as go
import plotly.io as pio
#from dash import Dash
#import tensorflow as tf
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


def LineareRegression(flight_Abflug, flight_Ankunft):
    # Setzen des Seeds für die Reproduzierbarkeit der Ergebnisse
    seed = 45
    np.random.seed(seed)

    # Laden und Filtern der Daten aus der CSV-Datei
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df.reset_index(drop=True)

    # Überprüfen, ob genügend Daten vorhanden sind; andernfalls abbrechen
    if df.empty:
        print("Keine Daten für die angegebene Route vorhanden.")
        return None

    # Datum als fortlaufende Zahl und Erstellung von Monats-Dummies für die saisonale Analyse
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    df['Date_ordinal'] = df['Date'].map(datetime.toordinal)
    month_dummies = pd.get_dummies(df['Month'], prefix='Month')
    df = pd.concat([df, month_dummies], axis=1)

    # Definieren der Features (Eingangsvariablen) und der Zielvariable
    feature_columns = ['Date_ordinal'] + [col for col in month_dummies.columns]
    X = df[feature_columns]
    y = df['$Real'].values.reshape((-1, 1))

    # Normalisieren der Zielvariablen y für die Skalierung
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y)

    # Anpassen des linearen Regressionsmodells
    model = LinearRegression()
    model.fit(X, y_scaled)

    # Erstellen einer Prognose für das nächste Jahr basierend auf dem letzten bekannten Datum
    last_date = df['Date'].max()
    next_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 13)]
    future_data = {
        'Date': next_dates,
        'Date_ordinal': [datetime.toordinal(date) for date in next_dates]
    }
    future_df = pd.DataFrame(future_data)
    future_month_dummies = pd.get_dummies(future_df['Date'].dt.month, prefix='Month')
    future_df = pd.concat([future_df, future_month_dummies], axis=1).reindex(columns=feature_columns, fill_value=0)
    predictions_scaled = model.predict(future_df[feature_columns])

    # Rücktransformation der skalierten Vorhersagen für die Interpretation
    predictions = scaler.inverse_transform(predictions_scaled)

    # Berechnung der Vorhersagen für die historischen Daten für das Training
    y_pred_train_scaled = model.predict(X)
    y_pred_train = scaler.inverse_transform(y_pred_train_scaled)

    # Berechnung der Metriken (MSE, MAE, RMSE) für die Bewertung des Modells
    mse = mean_squared_error(y, y_pred_train)
    mae = mean_absolute_error(y, y_pred_train)
    rmse = np.sqrt(mse)

    # Normalisierung der Metriken, um Vergleiche über verschiedene Datensätze zu ermöglichen
    max_mae = np.max(y) - np.min(y)
    max_mse = (np.max(y) - np.min(y)) ** 2
    max_rmse = np.sqrt(max_mse)

    normalized_mae = mae / max_mae if max_mae != 0 else 0
    normalized_mse = mse / max_mse if max_mse != 0 else 0
    normalized_rmse = rmse / max_rmse if max_rmse != 0 else 0

    # Ausgabe der normalisierten Metriken zur Überwachung der Modellgenauigkeit
    print(normalized_mae)
    print(normalized_mse)

    # Erstellen eines interaktiven Plotly-Diagramms für die Visualisierung der Vorhersagen und historischen Daten
    dates = [datetime.fromordinal(int(date_val)).date() for date_val in df['Date_ordinal'].values]
    dates_pred = [date.date() for date in next_dates]
    y_pred = predictions.flatten()

    fig = go.Figure()
    x_values = df['Date']
    y_values = df["$Real"]
    fig.add_trace(go.Scatter(x=dates, y=y.flatten(), mode="lines", name="Historische Daten"))
    yearly_avg = df.groupby(df['Date'].dt.year)['$Real'].mean().reset_index()
    fig.add_trace(go.Scatter(x=yearly_avg['Date'], y=yearly_avg['$Real'], mode="lines+markers", name="Jährl. Durchschnittspreis", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=dates_pred, y=y_pred, mode='lines+markers', name="Prognose", marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=[dates[-1], dates_pred[0]], y=[y.flatten()[-1], y_pred[0]], mode="lines", showlegend=False, line=dict(color='red', dash='dash')))

    # Hinzufügen der Metriken als Annotation zum Diagramm
    metrics_table = f"<b>Metriken</b><br>MSE: {normalized_mse:.2f}<br>MAE: {normalized_mae:.2f}<br>RMSE: {normalized_rmse:.2f}"
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='rgba(0,0,0,0)'),
        showlegend=True,
        name=metrics_table,
        hoverinfo='none'
    ))

    # Aktualisieren des Layouts des Plotly-Diagramms für die Darstellung und Formatierung
    fig.update_layout(template="plotly_dark", height=600, title=f"Saisonale Lineare Regression-Prognose für die Strecke: {flight_Abflug} & {flight_Ankunft}")
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
    # Beschriftung der Achsen
    fig.update_xaxes(title="Jahr")
    fig.update_yaxes(title="Preis ($)")

    # Festlegen der Plotvorlage für das Plotly-Diagramm
    pio.templates.default = "plotly_dark"
    print("SLR prediction")
    print(y_pred)
    
    return fig, normalized_mae, normalized_mse, normalized_rmse, y_pred