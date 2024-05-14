import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import os
import numpy as np
import random

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

csv_file_path_airports = 'koordinaten.csv'
csv_file_path_fares = 'AUS_Fares_March2024.csv'

if not os.path.exists(csv_file_path_airports) or not os.path.exists(csv_file_path_fares):
    print("Eine der Dateien wurde nicht gefunden. Bitte überprüfen Sie den Dateipfad.")
    exit()

df_fares = pd.read_csv(csv_file_path_fares)
df_fares['Origin'] = df_fares['Port1']
df_fares['Destination'] = df_fares['Port2']
df_cleaned = df_fares[['Year', 'Month', 'Origin', 'Destination', '$Value']].copy()

df_airports = pd.read_csv(csv_file_path_airports)

common_airports = set(df_cleaned['Origin']).intersection(df_airports['Airport'])
if not common_airports:
    print("Keine gemeinsamen Flughäfen gefunden. Bitte überprüfen Sie die Daten.")
    exit()

# Vordefinierte Farbpalette
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']

# Zufällige Auswahl von Farben für jeden Flughafen
airport_colors = [random.choice(colors) for _ in range(len(df_airports))]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Fluganalyse-Dashboard", className="text-center mb-4"),
    html.Div([
        html.H2("Flugpreisanalyse und Flugrouten-Dashboard", className="text-center mb-3"),
        html.Div([
            html.Div([
                html.Label('Wähle einen Abflugort:'),
                dcc.Dropdown(
                    id='origin-dropdown',
                    options=[{'label': airport, 'value': airport} for airport in sorted(common_airports)],
                    value=sorted(common_airports)[0]
                ),
            ], className='mb-3'),
            html.Div([
                html.Label('Wähle ein Ziel:'),
                dcc.Dropdown(id='destination-dropdown'),
            ], className='mb-3'),
            html.Div([
                html.Label('Wähle ein Jahr:'),
                dcc.Slider(
                    id='year-slider',
                    min=df_cleaned['Year'].min(),
                    max=df_cleaned['Year'].max(),
                    value=df_cleaned['Year'].min(),
                    marks={str(year): str(year) for year in df_cleaned['Year'].unique()},
                    step=None,
                    included=False,
                )
            ], className='mb-3')
        ], className='row'),
        dcc.Graph(id='price-time-series', style={'height': '60vh', 'width': '100%'}),
        dcc.Graph(id='average-price-bar', style={'height': '60vh', 'width': '100%'}),
        dcc.Graph(id='price-heatmap', style={'height': '60vh', 'width': '100%'}),
        dcc.Graph(id='flight-route-map', style={'height': '100vh', 'width': '100%'}),
        html.Div(id='distance-text', style={'textAlign': 'center', 'fontSize': 20, 'margin': '10px'}),
        html.Div(id='stats-summary', className='mb-3'),
    ], className='container-fluid')
], className='container-fluid', style={'padding': '0px', 'margin': '0px'})

@app.callback(
    Output('destination-dropdown', 'options'),
    Input('origin-dropdown', 'value')
)
def set_destination_options(selected_origin):
    destinations = df_cleaned[df_cleaned['Origin'] == selected_origin]['Destination'].unique()
    return [{'label': dest, 'value': dest} for dest in destinations if dest in common_airports]

@app.callback(
    Output('destination-dropdown', 'value'),
    Input('destination-dropdown', 'options')
)
def set_default_destination(options):
    if options:
        return options[0]['value']
    return None

@app.callback(
    [Output('price-time-series', 'figure'),
     Output('average-price-bar', 'figure'),
     Output('price-heatmap', 'figure'),
     Output('flight-route-map', 'figure'),
     Output('distance-text', 'children'),
     Output('stats-summary', 'children')],
    [Input('origin-dropdown', 'value'),
     Input('destination-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_graph_and_map(selected_origin, selected_destination, selected_year):
    try:
        if not selected_origin or not selected_destination or not selected_year:
            raise ValueError("Einer der Auswahlwerte ist leer. Bitte überprüfen Sie die Auswahl.")

        filtered_data = df_cleaned[(df_cleaned['Origin'] == selected_origin) & 
                                   (df_cleaned['Destination'] == selected_destination) &
                                   (df_cleaned['Year'] == selected_year)]
        
        if filtered_data.empty:
            raise ValueError("Die gefilterten Daten sind leer. Überprüfen Sie die Datenquelle oder die Filterkriterien.")
        
        line_fig = px.line(filtered_data, x='Month', y='$Value', title=f'Monatliche Preise von {selected_origin} nach {selected_destination} in {selected_year}')
        avg_prices = filtered_data.groupby('Month')['$Value'].mean().reset_index()
        bar_fig = px.bar(avg_prices, x='Month', y='$Value', title=f'Durchschnittspreise pro Monat von {selected_origin} nach {selected_destination} in {selected_year}')
        
        heatmap_data = df_cleaned[(df_cleaned['Origin'] == selected_origin) & (df_cleaned['Destination'] == selected_destination)]
        heatmap_fig = px.density_heatmap(heatmap_data, x='Month', y='Year', z='$Value', marginal_x="histogram", marginal_y="histogram", title='Preisfluktuationen über Zeit')
        
        origin_coords = df_airports[df_airports['Airport'] == selected_origin].iloc[0]
        destination_coords = df_airports[df_airports['Airport'] == selected_destination].iloc[0]
        distance_km = haversine(origin_coords['Longitude'], origin_coords['Latitude'], destination_coords['Longitude'], destination_coords['Latitude'])
        
        flight_path = go.Scattergeo(
            lon=[origin_coords['Longitude'], destination_coords['Longitude']],
            lat=[origin_coords['Latitude'], destination_coords['Latitude']],
            mode='lines',
            line=dict(width=2, color='blue'),
            text=f"{selected_origin} to {selected_destination}",
            hoverinfo='text',
            name="Ausgewählte Route"
        )

        airport_symbols = go.Scattergeo(
            lon=df_airports['Longitude'],
            lat=df_airports['Latitude'],
            mode='markers',
            marker=dict(size=10, color='rgba(0, 0, 0, 0.7)', symbol='circle'),
            hoverinfo='text',
            name="Nicht ausgewählte Flughäfen",
            text=df_airports['Airport']
        )

        selected_airports = go.Scattergeo(
            lon=[origin_coords['Longitude'], destination_coords['Longitude']],
            lat=[origin_coords['Latitude'], destination_coords['Latitude']],
            mode='markers',
            marker=dict(size=12, color='red', symbol='circle'),
            hoverinfo='text',
            text=[selected_origin, selected_destination],
            name="Abflug und Ziel Flughafen"
        )

        flight_route_map = go.Figure(data=[flight_path, airport_symbols, selected_airports])
        flight_route_map.update_layout(
            title=f"Flugroute von {selected_origin} nach {selected_destination}",
            geo=dict(
                showland=True,
                landcolor="lightgray",
                showcountries=True,
                countrycolor="white",
                showcoastlines=True,
                projection_type='orthographic'
            )
        )

        distance_text = f"Distanz von {selected_origin} nach {selected_destination}: {distance_km:.2f} km"
        
        stats_data = df_cleaned[(df_cleaned['Origin'] == selected_origin) & 
                                (df_cleaned['Destination'] == selected_destination) &
                                (df_cleaned['Year'] == selected_year)]
        summary = stats_data['$Value'].describe()
        summary_table = html.Div([
            dash_table.DataTable(
                data=summary.reset_index().to_dict('records'),
                columns=[{"name": i, "id": i} for i in summary.reset_index().columns],
                style_cell={'textAlign': 'left'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            )
        ])

        return line_fig, bar_fig, heatmap_fig, flight_route_map, distance_text, summary_table
    except Exception as e:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), f"Fehler: {str(e)}", "Statistische Daten nicht verfügbar"

if __name__ == '__main__':
    app.run_server(debug=True)