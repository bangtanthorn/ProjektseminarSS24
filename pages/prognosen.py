import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

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
    html.H1("Flugpreisprognosen"),
    dcc.Dropdown(
        id='route-dropdown',
        options=[{'label': route, 'value': route} for route in routes],
        value=routes[0]  # Standardmäßig erste Route auswählen
    ),
    dcc.Graph(id='price-forecast-graph'),
    html.Div(id='error-message', style={'color': 'red'})
])

@dash.callback(
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
        fig = {
            'data': [
                {'x': forecast_df.index, 'y': forecast_df['Forecast'], 'type': 'line', 'name': 'Prognose', 'line': {'dash': 'dash'}}
            ],
            'layout': {
                'title': f'SARIMA Prognose für die Route: {selected_route}',
                'xaxis': {'title': 'Datum'},
                'yaxis': {'title': 'Preis ($)'}
            }
        }
        return fig, ""
    except Exception as e:
        error_message = f"Fehler bei der Prognose für die Route {selected_route}: {str(e)}"
        fig = {
            'data': [],
            'layout': {
                'title': 'Fehler bei der Prognose',
                'xaxis': {'title': 'Datum'},
                'yaxis': {'title': 'Preis ($)'}
            }
        }
        return fig, error_message
