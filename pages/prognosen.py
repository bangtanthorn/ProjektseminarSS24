import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from pages.LSTM import get_lstm_predictions
from dash import html, callback
import plotly.graph_objs as go


dash.register_page(__name__, path='/prognosen', name="Prognosen")

# Daten laden
csv_file_path_fares = 'AUS_Fares_March2024.csv'
df = pd.read_csv(csv_file_path_fares)

# Datenvorbereitung
df['YearMonth'] = pd.to_datetime(df['YearMonth'], format='%Y%m')
df = df.rename(columns={'$Value': 'Value', '$Real': 'Real'})

# Duplikate entfernen
df = df.drop_duplicates(subset=['YearMonth', 'Route'])

# Liste der verfügbaren Routen
routes = df['Route'].unique()

layout = html.Div([
    #html.H1("Flugpreisprognosen"),
    dcc.Dropdown(
        id='route-dropdown',
        options=[{'label': route, 'value': route} for route in routes],
        value=routes[0],
        style={'width': '70%',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'display': 'block',
            'color': 'black',  # Schriftfarbe der Auswahl
            'font-family': 'Constantia',
            'font-size': '20px'}, # Standardmäßig erste Route auswählen
    ),
    dcc.Graph(id='price-forecast-graph', style={'width': '70%', "height": '60%', "margin-left": "auto", "margin-right": "auto", "display": "block"}),
    html.Div(id='error-message', style={'color': 'red'}),
    dcc.Graph(id="Method-Graph", style={'width': '70%', "height": '60%', "margin-left": "auto", "margin-right": "auto", "display": "block", "margin-top": "100"})
]),
style={'background-color': "#121212",
      'background-size': '100%',
      'width': '100%',
      'height':'95%',
      'font-family': 'Constantia',
      "margin-top":"200px",

      }

# Callback-Funktion zur Aktualisierung des Diagramms
@callback(
    Output('price-forecast-graph', 'figure'),
    Output('error-message', 'children'),
    Input('route-dropdown', 'value')
)
def update_graph(selected_route):
    try:
        # Daten für die ausgewählte Route filtern
        route_df = df[df['Route'] == selected_route].copy()
        route_df.set_index('YearMonth', inplace=True)
        route_df = route_df[~route_df.index.duplicated(keep='first')]

        # Sicherstellen, dass die Frequenz monatlich ist
        route_df = route_df.asfreq('MS')

        # SARIMA-Modell anpassen
        model = SARIMAX(route_df['Real'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        # Vorhersagen für die nächsten 6 Monate
        forecast = results.get_forecast(steps=6)
        forecast_index = pd.date_range(start=route_df.index[-1], periods=7, freq='MS')[1:]
        forecast_df = pd.DataFrame({'Forecast': forecast.predicted_mean}, index=forecast_index)

        # Daten für das Diagramm vorbereiten
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines+markers', name='Prognose', line=dict(dash='dash')))
        fig.update_layout(
            title=f'SARIMA Prognose für die Route: {selected_route}',
            xaxis_title='Datum',
            yaxis_title='Preis ($)',
            template='plotly_dark'
        )
        return fig, ""
    except Exception as e:
        error_message = f"Fehler bei der Prognose für die Route {selected_route}: {str(e)}"
        fig = go.Figure()
        fig.update_layout(
            title='Fehler bei der Prognose',
            xaxis_title='Datum',
            yaxis_title='Preis ($)',
            template='plotly_dark'
        )
        return fig, error_message



@callback(
    Output(component_id= "Method-Graph", component_property="figure"),
    Input(component_id= "flight_Abflug", component_property="data"),
    Input(component_id= "flight_Ankunft", component_property="data"),
   
)

    
def LSTM(flight_Abflug, flight_Ankunft):
    lstm_result = get_lstm_predictions(flight_Abflug, flight_Ankunft)
    fig = lstm_result  # Entpacke die zwei erwarteten Werte
        
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',', dtype={"Year": int})
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Value", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]

    #prediction_value = round(predictions[4], 2)

    return fig

