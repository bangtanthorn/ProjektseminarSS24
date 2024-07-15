import dash  # Importiere das Dash-Modul
from dash import html, dcc  # Importiere html- und dcc-Komponenten von Dash
import dash_bootstrap_components as dbc  # Importiere Dash-Bootstrap-Komponenten für ein besseres Layout
from dash import callback  # Importiere die Callback-Funktion von Dash
import pandas as pd  # Importiere Pandas für die Datenverarbeitung
from dash.dependencies import Input, Output, State  # Importiere notwendige Dash-Abhängigkeiten für die Callback-Funktionen

# Erstelle eine Dash-App und aktiviere die Verwendung von Seiten (use_pages=True)
app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.DARKLY])

# Definieren von Farben für das Layout
colors = {
    'background': '#000000',  # Hintergrundfarbe
    'text': '#FFFFFF',  # Textfarbe
    "Button": "#4169E1"  # Buttonfarbe
}

# Erstellen der oberen Navigationsleiste
topbar = dbc.Nav(
    [   #Beschriftungen der Seiten
        dbc.NavItem(dbc.NavLink("Visuelle Analyse Seite 1", href="/", style={"font-size": "25px", 'font-family': 'Constantia'})),  # Link zur ersten Analyse-Seite
        dbc.NavItem(dbc.NavLink("Visuelle Analyse Seite 2", href="/zweite-seite", style={"font-size": "25px", 'font-family': 'Constantia'})),  # Link zur zweiten Analyse-Seite
        dbc.NavItem(dbc.NavLink("Prognosen", href="/prognosen", style={"font-size": "25px", 'font-family': 'Constantia'})),  # Link zur Prognosen-Seite

        # Abstand und Label für Abflug-Dropdown
        dbc.Col(html.P(""), width=1),
        dbc.Col(html.P("Abflug:"), style={"font-size": "25px", "margin-left": "800px", 'font-family': 'Constantia'}),
        dbc.Col(
            dcc.Dropdown(
                options=[
                    {"label": "Adelaide", "value": "Adelaide"},
                    {"label": "Albury", "value": "Albury"},
                    {"label": "Alice Springs", "value": "Alice Springs"},
                    {"label": "Armidale", "value": "Armidale"},
                    {"label": "Avalon", "value": "Avalon"},
                    {"label": "Ayers Rock", "value": "Ayers Rock"},
                    {"label": "Ballina", "value": "Ballina"},
                    {"label": "Brisbane", "value": "Brisbane"},
                    {"label": "Broome", "value": "Broome"},
                    {"label": "Bundaberg", "value": "Bundaberg"},
                    {"label": "Cairns", "value": "Cairns"},
                    {"label": "Canberra", "value": "Canberra"},
                    {"label": "Coffs Harbour", "value": "Coffs Harbour"},
                    {"label": "Darwin", "value": "Darwin"},
                    {"label": "Devonport", "value": "Devonport"},
                    {"label": "Dubbo", "value": "Dubbo"},
                    {"label": "Geraldton", "value": "Geraldton"},
                    {"label": "Gold Coast", "value": "Gold Coast"},
                    {"label": "Hamilton Island", "value": "Hamilton Island"},
                    {"label": "Hervey Bay", "value": "Hervey Bay"},
                    {"label": "Hobart", "value": "Hobart"},
                    {"label": "Kalgoorlie", "value": "Kalgoorlie"},
                    {"label": "Karratha", "value": "Karratha"},
                    {"label": "Launceston", "value": "Launceston"},
                    {"label": "Melbourne", "value": "Melbourne"},
                    {"label": "Newcastle", "value": "Newcastle"},
                    {"label": "Paraburdoo", "value": "Paraburdoo"},
                    {"label": "Perth", "value": "Perth"},
                    {"label": "Port Macquarie", "value": "Port Macquarie"},
                    {"label": "Proserpine", "value": "Proserpine"},
                    {"label": "Sunshine Coast", "value": "Sunshine Coast"},
                    {"label": "Sydney", "value": "Sydney"}
                ],
                value="Canberra",  # Standardwert für den Dropdown
                id="Port3",  # ID des Dropdowns für die Abflugorte
                style={'width': '120%',
                       "margin-left": "-10px",
                       "display": "block",
                       "color": "black",
                       'font-family': 'Constantia',
                       "font-size": "20px"},
            ),
        ),
        
        # Label und Dropdown für Ankunft-Dropdown
        dbc.Col(html.P("Ankunft:", style={"font-size": "25px", 'font-family': 'Constantia'}), width=1),
        dbc.Col(
            dcc.Dropdown(
                options=[
                    {"label": "Adelaide", "value": "Adelaide"},
                    {"label": "Albury", "value": "Albury"},
                    {"label": "Alice Springs", "value": "Alice Springs"},
                    {"label": "Armidale", "value": "Armidale"},
                    {"label": "Avalon", "value": "Avalon"},
                    {"label": "Ayers Rock", "value": "Ayers Rock"},
                    {"label": "Ballina", "value": "Ballina"},
                    {"label": "Brisbane", "value": "Brisbane"},
                    {"label": "Broome", "value": "Broome"},
                    {"label": "Bundaberg", "value": "Bundaberg"},
                    {"label": "Cairns", "value": "Cairns"},
                    {"label": "Canberra", "value": "Canberra"},
                    {"label": "Coffs Harbour", "value": "Coffs Harbour"},
                    {"label": "Darwin", "value": "Darwin"},
                    {"label": "Devonport", "value": "Devonport"},
                    {"label": "Dubbo", "value": "Dubbo"},
                    {"label": "Geraldton", "value": "Geraldton"},
                    {"label": "Gold Coast", "value": "Gold Coast"},
                    {"label": "Hamilton Island", "value": "Hamilton Island"},
                    {"label": "Hervey Bay", "value": "Hervey Bay"},
                    {"label": "Hobart", "value": "Hobart"},
                    {"label": "Kalgoorlie", "value": "Kalgoorlie"},
                    {"label": "Karratha", "value": "Karratha"},
                    {"label": "Launceston", "value": "Launceston"},
                    {"label": "Melbourne", "value": "Melbourne"},
                    {"label": "Newcastle", "value": "Newcastle"},
                    {"label": "Paraburdoo", "value": "Paraburdoo"},
                    {"label": "Perth", "value": "Perth"},
                    {"label": "Port Macquarie", "value": "Port Macquarie"},
                    {"label": "Proserpine", "value": "Proserpine"},
                    {"label": "Sunshine Coast", "value": "Sunshine Coast"},
                    {"label": "Sydney", "value": "Sydney"}
                ],
                value="Melbourne",  # Standardwert für den Dropdown
                id="Port4",  # ID des Dropdowns für die Ankunftsorte
                style={'width': '120%',
                       "margin-left": "-15px",
                       "display": "block",
                       "color": "black",
                       'font-family': 'Constantia',
                       "font-size": "20px"},
            ),
        )
    ],
    vertical=False,  # Horizontale Anordnung der Elemente
    pills=True,  # Verwendet Pills-Style für Navigationselemente
    className="navbar navbar-expand-lg navbar-dark bg-secondary",  # Bootstrap-Klassen für Styling
    style={"color": "#A9A9A9"}  # Textfarbe der Navigationsleiste
)

# Layout der App definieren
app.layout = dbc.Container(
    [
        html.Div(
            html.H1("Flugpreisanalyse", style={'fontSize': 70, 'textAlign': 'center', 'color': colors['text'], 'font-family': 'Constantia', 'fontWeight': 'normal'})),  # Überschrift
        html.Hr(),  # Horizontale Linie
        dbc.Row([dbc.Col(topbar)]),  # Einfügen der Navigationsleiste
        dash.page_container,  # Container für Seiteninhalt
        html.Hr(),  # Weitere horizontale Linie
        dbc.Row(dbc.Col([dcc.Store(id="flight_Abflug"), dcc.Store(id="flight_Ankunft")])),  # Speicherung der Flugdaten
        dcc.Store(id='normalized_mae_sarima'),  # Speicherung von MAE-Werten
        dcc.Store(id='normalized_mse_sarima'),  # Speicherung von MSE-Werten
        dcc.Store(id='normalized_rmse_sarima'),  # Speicherung von RMSE-Werten
    ],
    
    style={
        'background-color': "#121212",  # Hintergrundfarbe
        'background-size': '100%',  # Hintergrundgröße
        'position': 'fixed',  # Fixierte Position
        'width': '100%',  # Breite auf 100%
        'height': '100%',  # Höhe auf 100%
        'font-family': 'Rockwell',  # Schriftart
        "display": "block", 
        "margin-left": "auto",
        "margin-right": "auto",
        'textAlign': 'center',  # Zentrierter Text
        "overflow": "scroll"  # Scrollbare Ansicht
    },
    fluid=True  # Flüssiges Layout für die Containergröße
)

# Callback-Funktion zum Aktualisieren der Ankunfts-Dropdown-Optionen basierend auf dem ausgewählten Abflugort
@callback(
    Output("Port4", "options"),
    Input("Port3", "value")
)
def update_port2_options(selected_port1):
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')  # CSV-Datei laden
    df = df[["Year", "Port1", "Port2", "$Value"]]  # Relevante Spalten auswählen
    filtered_df = df[df["Port1"] == selected_port1]  # Daten nach ausgewähltem Abflugort filtern
    port2_options = [{"label": port, "value": port} for port in filtered_df["Port2"].unique()]  # Optionen für Ankunftsort erstellen

    return port2_options

# Callback-Funktion zum Aktualisieren der gespeicherten Flugdaten basierend auf den ausgewählten Abflug- und Ankunftsorten
@callback(
    Output("flight_Abflug", "data"),
    Output("flight_Ankunft", "data"),
    [Input("Port3", "value"),
     Input("Port4", "value")]
)
def update_flight_store(Port3, Port4):
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')  # CSV-Datei laden
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Value", "$Real"]]  # Relevante Spalten auswählen
    df = df[(df["Port1"] == Port3) & (df["Port2"] == Port4)]  # Daten nach ausgewählten Abflug- und Ankunftsorten filtern
    df = df.reset_index(drop=True)  # Index zurücksetzen

    return Port3, Port4

# Starte die Dash-App, falls dieses Skript direkt ausgeführt wird
if __name__ == "__main__":
    app.run(debug=True)  # App im Debug-Modus starten