import dash
from dash import dash_table
from dash import html, dcc, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import random
import os
from dash.dash_table.Format import Group
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import os
import random
import plotly.express as px
import plotly.graph_objects as go



# Registrieren der Seite für Dash mit Pfad 
dash.register_page(__name__, path='/zweite-seite', name="Fluganalyse2")


# Funktion zur Berechnung der Haversine-Distanz zwischen zwei Punkten für die Weltkugel
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

# Pfade zu den CSV-Dateien
csv_file_path_airports = 'koordinaten.csv'
csv_file_path_fares = 'AUS_Fares_March2024.csv'
# Überprüfen, ob die Dateien existieren
if not os.path.exists(csv_file_path_airports) or not os.path.exists(csv_file_path_fares):
    print("Eine der Dateien wurde nicht gefunden. Bitte überprüfen Sie den Dateipfad.")
    exit()

# Lesen der Daten aus den CSV-Dateien
df_fares = pd.read_csv(csv_file_path_fares)
df_fares['Origin'] = df_fares['Port1']
df_fares['Destination'] = df_fares['Port2']
df_cleaned = df_fares[['Year', 'Month', 'Origin', 'Destination', '$Real']].copy()

df_airports = pd.read_csv(csv_file_path_airports)

# Überprüfen, ob gemeinsame Flughäfen in beiden Datensätzen vorhanden sind

common_airports = set(df_cleaned['Origin']).intersection(df_airports['Airport'])
if not common_airports:
    print("Keine gemeinsamen Flughäfen gefunden. Bitte überprüfen Sie die Daten.")
    exit()

# Vordefinierte Farbpalette
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']

# Zufällige Auswahl von Farben für jeden Flughafen
airport_colors = [random.choice(colors) for _ in range(len(df_airports))]

# Layout der Seite definieren
layout = html.Div([
    html.Div([
        html.P(""),
        html.P(""),
        html.Div([
            html.Div([
                html.P(""),     
                html.P(""), 
                html.Label('Wählen Sie ein Jahr:', style={'font-family': 'Constantia'}),
                dcc.Slider(
                    id='year-slider',
                    min=df_cleaned['Year'].min(),
                    max=df_cleaned['Year'].max(),
                    value=df_cleaned['Year'].min(),
                    marks={str(year): str(year) for year in df_cleaned['Year'].unique()},
                    step=None,
                    included=False,
                    className='slider-styles'
                ),
            ], className='mb-3', style={'width': '65%', 'margin': '0 auto'})  # Sliderelementbreite auf 65% gesetzt
        ], className='row'),
        dcc.Graph(id='price-time-series', style={'width': '65%', "height": '90%', "margin": "0 auto", "color": "#696969", "margin-top": "40px"}),
        dcc.Graph(id='average-price-bar', style={'width': '65%', "height": '%', "margin": "0 auto", "color": "#696969", "margin-top": "40px"}),
        dcc.Graph(id='price-heatmap', style={'width': '65%', "height": '90%', "margin": "0 auto", "color": "#696969", "margin-top": "40px"}),
        dcc.Graph(id='flight-route-map', style={'width': '65%', "height": '120%', "margin": "0 auto", "color": "#696969", "margin-top": "40px"}),
        html.Div(id='distance-text', style={'textAlign': 'center', 'fontSize': 20, 'margin': '10px'}),
        html.Div(id='stats-summary', className='mb-3', style={'width': '65%', "height": '100%', "margin": "0 auto", "margin-top": "40px"}),
    ], className='container-fluid', style={'padding': '0px', 'margin': '0px', 'text-align': 'center', 'width': '100%'})
], className='container-fluid', style={'padding': '0px', 'margin': '0px'})


@callback(
    Output('destination-dropdown', 'options'),
    Input('Port3', 'value')
)
def set_destination_options(selected_origin):
    # Filtere die Daten, um nur die Zeilen zu erhalten, deren Abflugort dem ausgewählten Abflugort entspricht.
    destinations = df_cleaned[df_cleaned['Origin'] == selected_origin]['Destination'].unique()
    # Erstelle eine Liste von Dictionaries, die als Optionen für ein Dropdown-Menü verwendet werden können.
    # Jedes Dictionary enthält ein Ziel als 'label' und 'value', aber nur, wenn das Ziel in den common_airports enthalten ist.
    return [{'label': dest, 'value': dest} for dest in destinations if dest in common_airports]



# Callback-Funktion  für das Zielflughafen-Dropdown
@callback(
    Output('destination-dropdown', 'value'),
    Input('destination-dropdown', 'options')
)
# Callback-Funktion zum Setzen vom Standardwerten für das Zielflughafen-Dropdown
def set_default_destination(options):
    if options:
        return options[0]['value']
    return None

# Callback-Funktion zum Aktualisieren der Graphen und der Karte
@callback(
    [Output('price-time-series', 'figure'),
     Output('average-price-bar', 'figure'),
     Output('price-heatmap', 'figure'),
     Output('flight-route-map', 'figure'),
     Output('distance-text', 'children'),
     Output('stats-summary', 'children')],
    [Input('Port3', 'value'),
     Input('Port4', 'value'),
     Input('year-slider', 'value')]
)
def update_graph_and_map(selected_origin, selected_destination, selected_year):
    try:
         # Überprüfen, ob die ausgewählten Werte gültig sind
        if not selected_origin or not selected_destination or not selected_year:
            raise ValueError("Einer der Auswahlwerte ist leer. Bitte überprüfen Sie die Auswahl.")
        # Filtern der Daten nach den ausgewählten Werten
        filtered_data = df_cleaned[(df_cleaned['Origin'] == selected_origin) & 
                                   (df_cleaned['Destination'] == selected_destination) &
                                   (df_cleaned['Year'] == selected_year)]
        
        if filtered_data.empty:
            raise ValueError("Die gefilterten Daten sind leer. Überprüfen Sie die Datenquelle oder die Filterkriterien.")
        # Erstellen des Liniendiagramms für die monatlichen Preise
        line_fig = px.line(filtered_data, x='Month', y='$Real', title=f'Monatliche Preise von {selected_origin} nach {selected_destination} in {selected_year}')
        line_fig.update_layout(template='plotly_dark', title='Monatliche Preise', yaxis_title='Preis', xaxis_title='Monat')
        line_fig.update_traces(name='Preis', hovertemplate='Monat: %{x}<br>Preis: %{y} €')
         
        # Erstellen des Balkendiagramms für die durchschnittlichen Preise pro Monat
        avg_prices = filtered_data.groupby('Month')['$Real'].mean().reset_index()
        bar_fig = px.bar(avg_prices, x='Month', y='$Real', title=f'Durchschnittspreise pro Monat von {selected_origin} nach {selected_destination} in {selected_year}')
        bar_fig.update_layout(template='plotly_dark', title='Durchschnittspreise pro Monat', yaxis_title='Preis', xaxis_title='Monat')
        bar_fig.update_traces(name='Preis', hovertemplate='Monat: %{x}<br>Durchschnittspreis: %{y} €')

        # Erstellen der Heatmap für Preisfluktuationen
        heatmap_data = df_cleaned[(df_cleaned['Origin'] == selected_origin) & (df_cleaned['Destination'] == selected_destination)]
        heatmap_fig = px.density_heatmap(heatmap_data, x='Month', y='Year', z='$Real', marginal_x="histogram", marginal_y="histogram", title='Preisfluktuationen über Zeit')
        heatmap_fig.update_layout(template='plotly_dark', title='Preisfluktuationen', yaxis_title='Jahr', xaxis_title='Monat')
        # Berechnung der Distanz zwischen den Flughäfen
        origin_coords = df_airports[df_airports['Airport'] == selected_origin].iloc[0]
        destination_coords = df_airports[df_airports['Airport'] == selected_destination].iloc[0]
        distance_km = haversine(origin_coords['Longitude'], origin_coords['Latitude'], destination_coords['Longitude'], destination_coords['Latitude'])
        # Erstellen der Flugroute auf der Karte
        flight_path = go.Scattergeo(
            lon=[origin_coords['Longitude'], destination_coords['Longitude']],
            lat=[origin_coords['Latitude'], destination_coords['Latitude']],
            mode='lines',
            line=dict(width=2, color='blue'),
            text=f"{selected_origin} nach {selected_destination}",
            hoverinfo='text',
            name="Ausgewählte Route"
        )
        # Erstellen der Markierungen für Flughäfen auf der Karte
        airport_symbols = go.Scattergeo(
            lon=df_airports['Longitude'],
            lat=df_airports['Latitude'],
            mode='markers',
            marker=dict(size=10, color='rgba(0, 0, 0, 0.7)', symbol='circle'),
            hoverinfo='text',
            name="Nicht ausgewählte Flughäfen",
            text=df_airports['Airport']
        )
       # Markierungen für die ausgewählten Flughäfen
        selected_airports = go.Scattergeo(
            lon=[origin_coords['Longitude'], destination_coords['Longitude']],
            lat=[origin_coords['Latitude'], destination_coords['Latitude']],
            mode='markers',
            marker=dict(size=12, color='red', symbol='circle'),
            hoverinfo='text',
            text=[selected_origin, selected_destination],
            name="Abflug und Ziel Flughafen"
        )
        # Erstellen der Karte mit der Flugroute und den Flughäfen
        flight_route_map = go.Figure(data=[flight_path, airport_symbols, selected_airports])
        flight_route_map.update_layout(
    title=f"Flugroute von {selected_origin} nach {selected_destination}",
    template='plotly_dark',
    geo=dict(
        showland=True,
        landcolor="rgb(244, 164, 96)", 
        showocean=True,
        oceancolor="rgb(135, 206, 235)", 
        showcountries=True,
        countrycolor="rgb(32, 32, 32)", 
        showcoastlines=True,
        coastlinecolor="rgb(32, 32, 32)", 
        projection_type='orthographic',
        projection_scale=1,  
        center=dict(lon=0, lat=0)  
    ),
    width=1250,  
    height=1250  


)
        # Erstellen der Karte mit der Flugroute und den Flughäfen
        distance_text = f"Distanz von {selected_origin} nach {selected_destination}: {distance_km:.2f} km"
        
        stats_data = df_cleaned[(df_cleaned['Origin'] == selected_origin) & 
                                (df_cleaned['Destination'] == selected_destination) &
                                (df_cleaned['Year'] == selected_year)]
        summary = stats_data['$Real'].describe()


        # Werte auf zwei Nachkommastellen runden
        summary = summary.round(2)

        summary_table = html.Div([
            dash_table.DataTable(
                data=summary.reset_index().to_dict('records'),
                columns=[{"name": i, "id": i} for i in summary.reset_index().columns],
                style_cell={'textAlign': 'left', 'color': 'white', 'backgroundColor': 'black'},
                style_header={
                    'backgroundColor': 'black',
                    'fontWeight': 'bold',
                    'color': 'white'
                }
            )
        ], style={'width': '70%', 'margin': '0 auto'})

        # Rückgabe der erstellten Diagramme und der berechneten Werte
        return line_fig, bar_fig, heatmap_fig, flight_route_map, distance_text, summary_table
    

        # Fehlerbehandlung und Rückgabe von leeren Diagrammen und einer Fehlermeldung
    except Exception as e:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), f"Fehler: {str(e)}", "Statistische Daten nicht verfügbar"

