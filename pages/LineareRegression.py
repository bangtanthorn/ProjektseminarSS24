import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime as dt
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def LineareRegression(flight_Abflug, flight_Ankunft):
    # Setzen vom Seed
    seed = 45
    np.random.seed(seed)

    # Laden und Filtern der Daten
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df.reset_index(drop=True)

    # Überprüfen, ob es genügend Daten gibt
    if df.empty:
        print("Keine Daten für die angegebene Route vorhanden.")
        return None

    # Jahr und Monat in eine fortlaufende Zahl umwandeln
    df['Date'] = df.apply(lambda row: datetime(row['Year'], row['Month'], 1), axis=1)
    df['Date_ordinal'] = pd.to_datetime(df['Date']).map(datetime.toordinal)

    # Merkmale und Zielvariable definieren
    x = np.array(df['Date_ordinal']).reshape((-1, 1))
    y = df['$Real'].values.reshape((-1, 1))

    # Lineare Regression anpassen
    model = LinearRegression()
    model.fit(x, y)

    # Prognose für die nächsten 5 Monate erstellen
    last_date = df['Date'].max()
    next_months = [last_date + timedelta(days=30 * i) for i in range(1, 6)]  # Für die nächsten 5 Monate
    next_months_ordinal = np.array([datetime.toordinal(date) for date in next_months]).reshape(-1, 1)

    predictions = model.predict(next_months_ordinal)
   

    # Plotly-Grafik erstellen
    dates = [datetime.fromordinal(int(date_val)).date() for date_val in df['Date_ordinal'].values]
    dates_pred = [datetime.fromordinal(int(date_val)).date() for date_val in next_months_ordinal.flatten()]
    y_pred = predictions.flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y.flatten(), mode="lines", name="Historische Daten"))
    fig.add_trace(go.Scatter(x=dates_pred, y=y_pred, mode="markers", name="Vorhersagen", marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=[dates[-1], dates_pred[0]], y=[y.flatten()[-1], y_pred[0]], mode="lines", showlegend=False, line=dict(color='red', dash='dash')))

    yearly_avg = df.groupby(df['Date'].dt.year)['$Real'].mean().reset_index()
    yearly_avg['Date'] = yearly_avg['Date'].apply(lambda x: datetime(x, 1, 1))
    fig.add_trace(go.Scatter(x=yearly_avg['Date'], y=yearly_avg['$Real'], mode="lines+markers", name="Jährl. Durchschnittspreis", line=dict(color='orange')))

    fig.update_layout(template="plotly_dark", height=600, title=f"Lineare Regressions-Prognose für die Strecke: {flight_Abflug} & {flight_Ankunft}")
    fig.update_xaxes(title="Jahr")
    fig.update_yaxes(title="Preis ($)")
    pio.templates.default = "plotly_dark"

    return fig

# Beispielaufruf
#fig, y_pred, r_sq, mse = LineareRegression("Adelaide", "Gold Coast")
#fig.show()

#print(f"R^2: {r_sq}")
#print(f"Mean Squared Error (MSE): {mse}")
#print(f"Vorhersagewerte: {y_pred}")





# Beispielaufruf
fig = LineareRegression("Hobart", "Melbourne")
fig.show()

#print(f"R^2: {r_sq}")
#print(f"Mean Squared Error (MSE): {mse}")
#print(f"Vorhersagewerte: {y_pred}")
