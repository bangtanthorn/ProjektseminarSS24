import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import dash
from dash import dcc, html, callback, Output, Input
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import itertools
from pages.LSTM import get_lstm_predictions  # Import der LSTM-Funktion

dash.register_page(__name__, path='/prognosen', name="Prognosen")

# Daten laden
csv_file_path_fares = 'AUS_Fares_March2024.csv'
df = pd.read_csv(csv_file_path_fares)

# Datenvorbereitung
df['YearMonth'] = pd.to_datetime(df['YearMonth'], format='%Y%m')
df = df.rename(columns={'$Value': 'Value', '$Real': 'Real'})
df = df.drop_duplicates(subset=['YearMonth', 'Route'])

# Liste der verfügbaren Routen
routes = df['Route'].unique()

# SARIMA-Modellparameter-Optimierung
def optimize_sarima(endog, seasonal_periods):
    p = d = q = range(0, 2)
    seasonal_pdq = [(x[0], x[1], x[2], seasonal_periods) for x in list(itertools.product(p, d, q))]
    
    best_aic = float("inf")
    best_params = None
    
    for param in list(itertools.product(p, d, q)):
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(endog, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (param, param_seasonal)
            except:
                continue
    return best_params

layout = html.Div([
    dcc.Dropdown(
        id='route-dropdown',
        options=[{'label': route, 'value': route} for route in routes],
        value=routes[0],
        style={'width': '70%', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block', 'color': 'black', 'font-family': 'Constantia', 'font-size': '20px'}
    ),
    dcc.Graph(id='price-forecast-graph', style={'width': '70%', 'height': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block'}),
    html.Div(id='error-message', style={'color': 'red'}),
    html.Div(id='model-metrics', style={'color': 'white', 'fontSize': 16}),
    dcc.Graph(id="Method-Graph", style={'width': '70%', 'height': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block', 'margin-top': '100'})
], style={'background-color': "#121212", 'width': '100%', 'height': '95%', 'font-family': 'Constantia', "margin-top": "200px"})

# Callback für SARIMA-Prognosen
@callback(
    [Output('price-forecast-graph', 'figure'),
     Output('error-message', 'children'),
     Output('model-metrics', 'children')],
    [Input('route-dropdown', 'value')]
)
def update_graph(selected_route):
    try:
        # Daten für die ausgewählte Route filtern
        route_df = df[df['Route'] == selected_route].copy()
        route_df.set_index('YearMonth', inplace=True)
        route_df = route_df[~route_df.index.duplicated(keep='first')]

        # Sicherstellen, dass die Frequenz monatlich ist und fehlende Werte interpolieren
        route_df = route_df.asfreq('MS').interpolate()

        # Optimierte SARIMA-Parameter ermitteln
        best_params = optimize_sarima(route_df['Real'], 12)
        
        # SARIMA-Modell anpassen
        model = SARIMAX(route_df['Real'], order=best_params[0], seasonal_order=best_params[1])
        results = model.fit()

        # Vorhersagen für die nächsten 12 Monate
        forecast = results.get_forecast(steps=12)
        forecast_df = pd.DataFrame({
            'Forecast': forecast.predicted_mean,
            'Lower CI': forecast.conf_int().iloc[:, 0],
            'Upper CI': forecast.conf_int().iloc[:, 1]
        }, index=pd.date_range(start=route_df.index[-1], periods=13, freq='MS')[1:])

        # Daten für das Diagramm vorbereiten
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=route_df.index, y=route_df['Real'], mode='lines', name='Tatsächliche Preise', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines+markers', name='Prognose', line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower CI'], mode='lines', line=dict(color='grey'), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper CI'], mode='lines', line=dict(color='grey'), fill='tonexty', showlegend=False))

        # Fehlermetriken berechnen
        mse = mean_squared_error(route_df['Real'], results.fittedvalues)
        mae = mean_absolute_error(route_df['Real'], results.fittedvalues)
        rmse = np.sqrt(mse)
        metrics_text = f'MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}'

        # Layout anpassen
        fig.update_layout(
            title=f'SARIMA Prognose für Route: {selected_route}',
            xaxis_title='Datum',
            yaxis_title='Preis ($)',
            template='plotly_dark',
            annotations=[{
                'text': metrics_text,
                'x': 0.5,
                'y': -0.2,
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'color': 'white'}
            }]
        )
        return fig, "", metrics_text
    except Exception as e:
        error_message = f"Fehler bei der Prognose für die Route {selected_route}: {str(e)}"
        fig = go.Figure()
        fig.update_layout(
            title='Fehler bei der Prognose',
            xaxis_title='Datum',
            yaxis_title='Preis ($)',
            template='plotly_dark'
        )
        return fig, error_message, ""

# Callback für LSTM-Prognosen
@callback(
    Output(component_id="Method-Graph", component_property="figure"),
    Input(component_id="flight_Abflug", component_property="data"),
    Input(component_id="flight_Ankunft", component_property="data"),
)
def LSTM(flight_Abflug, flight_Ankunft):
    lstm_result = get_lstm_predictions(flight_Abflug, flight_Ankunft)
    fig = lstm_result  # Entpacke die zwei erwarteten Werte
        
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',', dtype={"Year": int})
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Value", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]

    return fig
