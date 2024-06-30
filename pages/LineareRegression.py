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


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.io as pio

def LineareRegression(flight_Abflug, flight_Ankunft):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio
    from sklearn.linear_model import LinearRegression
    from datetime import datetime, timedelta

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

    # Datum als fortlaufende Zahl und Erstellung von Monats-Dummies
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    df['Date_ordinal'] = df['Date'].map(datetime.toordinal)
    month_dummies = pd.get_dummies(df['Month'], prefix='Month')
    df = pd.concat([df, month_dummies], axis=1)

    # Merkmale und Zielvariable definieren
    feature_columns = ['Date_ordinal'] + [col for col in month_dummies.columns]
    X = df[feature_columns]
    y = df['$Real'].values.reshape((-1, 1))

    # Lineare Regression anpassen
    model = LinearRegression()
    model.fit(X, y)

    # Prognose für das nächste Jahr erstellen
    last_date = df['Date'].max()
    next_dates = [last_date + timedelta(days=30 * i) for i in range(1, 13)]
    future_data = {
        'Date': next_dates,
        'Date_ordinal': [datetime.toordinal(date) for date in next_dates]
    }
    future_df = pd.DataFrame(future_data)
    future_month_dummies = pd.get_dummies(future_df['Date'].dt.month, prefix='Month')
    future_df = pd.concat([future_df, future_month_dummies], axis=1).reindex(columns=feature_columns, fill_value=0)
    predictions = model.predict(future_df[feature_columns])

    # Vorhersagen für die historischen Daten berechnen
    y_pred_train = model.predict(X)

    # Berechnung der Metriken
    mse = mean_squared_error(y, y_pred_train)
    mae = mean_absolute_error(y, y_pred_train)
    rmse = np.sqrt(mse)

    # Plotly-Grafik erstellen
    dates = [datetime.fromordinal(int(date_val)).date() for date_val in df['Date_ordinal'].values]
    dates_pred = [date.date() for date in next_dates]
    y_pred = predictions.flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y.flatten(), mode="lines", name="Historische Daten"))
    fig.add_trace(go.Scatter(x=dates_pred, y=y_pred, mode='lines+markers', name="Vorhersagen", marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=[dates[-1], dates_pred[0]], y=[y.flatten()[-1], y_pred[0]], mode="lines", showlegend=False, line=dict(color='red', dash='dash')))

    # # Metriken als Annotation hinzufügen
    # metrics_text = f"MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}"
    # fig.add_annotation(
    #     xref="paper", yref="paper",
    #     x=0.5, y=-0.15,
    #     showarrow=False,
    #     text=metrics_text,
    #     font=dict(size=12, color="white"),
    #     align="center",
    #     bgcolor="rgba(0,0,0,0.5)",
    #     bordercolor="rgba(0,0,0,0.5)"
    # )

    fig.update_layout(template="plotly_dark", height=600, title=f"Saisonale Lineare Regression-Prognose für die Strecke: {flight_Abflug} & {flight_Ankunft}")

    metrics_table = f"<b>Metriken</b><br>MSE: {mse:.2f}<br>MAE: {mae:.2f}<br>RMSE: {rmse:.2f}"
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
    fig.update_xaxes(title="Jahr")
    fig.update_yaxes(title="Preis ($)")
    pio.templates.default = "plotly_dark"

    return fig
