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



def LineareRegression(flight_Abflug, flight_Ankunft):
    # Setzen vom Seed
    seed = 45
    np.random.seed(seed)

    # Laden und Filtern der Daten
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df.reset_index(drop=True)

    # Überprüfen, ob genügend Daten vorhanden sind
    if df.empty:
        raise ValueError("Keine Daten für die angegebene Route gefunden.")

    # Vorbereitung der Daten für die Regression
    df['Date'] = pd.to_datetime(df['YearMonth'], format='%Y%m')
    df['Date_ordinal'] = df['Date'].map(dt.datetime.toordinal)
    x = df['Date_ordinal'].values.reshape(-1, 1)
    y = df['$Real'].values.reshape(-1, 1)

    # Daten in Trainings- und Testdaten aufteilen
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

    # Lineares Regressionsmodell erstellen und trainieren
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Vorhersage auf den Testdaten
    y_pred_testing = model.predict(x_test)

    # Vergleich der tatsächlichen und prognostizierten Werte
    pred_y_df = pd.DataFrame({"Actual Value": y_test.flatten(), "Predicted Value": y_pred_testing.flatten(), "Difference": y_test.flatten() - y_pred_testing.flatten()})
    print(round(pred_y_df.head(5), 2))
    print("")

    # Berechnung des R^2-Werts und des Mean Squared Error (MSE)
    r_sq = model.score(x_test, y_test)
    mse = np.mean((y_test.flatten() - y_pred_testing.flatten()) ** 2)

    # Vorhersage für zukünftige Datenpunkte
    future_periods = 5
    x_pred = np.array([df['Date_ordinal'].max() + i for i in range(1, future_periods + 1)]).reshape(-1, 1)
    y_pred = model.predict(x_pred)
    dates_pred = [dt.date.fromordinal(int(date_val)) for date_val in x_pred.flatten()]

    # Erstellung der Diagramme
    dates = [dt.date.fromordinal(int(date_val)) for date_val in x.flatten()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y.flatten(), mode="lines", name="Historische Daten"))
    fig.add_trace(go.Scatter(x=dates_pred, y=y_pred.flatten(), mode="markers", name="Vorhersagen", marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=[dates[-1], dates_pred[0]], y=[y.flatten()[-1], y_pred.flatten()[0]], mode="lines", showlegend=False, line=dict(color='red', dash='dash')))
    yearly_avg = df.groupby(df['Date'].dt.year)['$Real'].mean().reset_index()
    fig.add_trace(go.Scatter(x=yearly_avg['Date'], y=yearly_avg['$Real'], mode="lines+markers", name="Jährl. Durchschnittspreis", line=dict(color='orange')))
    fig.update_layout(template="plotly_dark", height=600, title=f"Lineare Regressions-Prognose für die Strecke {flight_Abflug} to {flight_Ankunft}")
    fig.update_xaxes(title="Datum")
    fig.update_yaxes(title="Preis ($)")
    pio.templates.default = "plotly_dark"

    return fig

# Beispielaufruf
#fig, y_pred, r_sq, mse = LineareRegression("Adelaide", "Gold Coast")
#fig.show()

#print(f"R^2: {r_sq}")
#print(f"Mean Squared Error (MSE): {mse}")
#print(f"Vorhersagewerte: {y_pred}")
