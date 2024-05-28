import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import callback
import pandas as pd
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash.dash_table.Format import Group

# Erstelle eine Dash-App und aktiviere die Verwendung von Seiten (use_pages=True)
app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.DARKLY])

# Definieren von Farben
colors = {
    'background': '#000000',
    'text': '#FFFFFF',
    "Button": "#4169E1"
}

topbar = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("Fluganalyse1", href="/", style={"font-size": "25px", 'font-family': 'Constantia'})),
        dbc.NavItem(dbc.NavLink("Fluganalyse2", href="/zweite-seite", style={"font-size": "25px", 'font-family': 'Constantia'})),
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
                value="Adelaide",
                id="Port3",
                style={'width': '70%',
                       "margin-left": "-50px",
                       "display": "block",
                       "color": "black",
                       'font-family': 'Constantia',
                       "font-size": "20px"},
            ),
        
        ),
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
                value="Brisbane",
                id="Port4",
                style={'width': '70%',
                       "margin-left": "-15px",
                       "display": "block",
                       "color": "black",
                       'font-family': 'Constantia',
                       "font-size": "20px"},
            ),
        )
    ],
    vertical=False,
    pills=True,
    className="navbar navbar-expand-lg navbar-dark bg-secondary",
    style={"color": "#A9A9A9"}
)

app.layout = dbc.Container([

        html.Div(
        html.H1("Flugpreisanalyse", style={'fontSize': 70, 'textAlign': 'center', 'color': colors['text'], 'font-family': 'Constantia', 'fontWeight': 'normal'})),
        html.Hr(),
        dbc.Row(        
            dbc.Col(
                [
                    topbar  # Fügt oben definierte Bar ein
                ])),
        dash.page_container,  # Fügt definierte Seitennamen und Referenzen ein
        html.Hr(),
            dbc.Row(
        dbc.Col(
             [
                dcc.Store(id="flight_Abflug"),
                dcc.Store(id="flight_Ankunft")  # Speicher des geladenen Aktien Tickers (Informationen über die gewählte Aktie)
             ]
        ))
    ], 
    style={'background-color': "#121212",
          'background-size': '100%',
          'position': 'fixed',
          'width': '100%',
          'height': '100%',
          'font-family': 'Rockwell',
          "display": "block", 
          "margin-left": "auto",
          "margin-right": "auto",
          'textAlign': 'center',
          "overflow": "scroll"
          }
, fluid=True)

@callback(
    Output("Port4", "options"),
    Input("Port3", "value")
)
def update_port2_options(selected_port1):
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year", "Port1", "Port2", "$Value"]]
    filtered_df = df[df["Port1"] == selected_port1]
    port2_options = [{"label": port, "value": port} for port in filtered_df["Port2"].unique()]

    return port2_options

@callback(
    Output("flight_Abflug", "data"),
    Output("flight_Ankunft", "data"),
    [Input("Port3", "value"),
     Input("Port4", "value")]
)
def update_flight_store(Port3, Port4):
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Value", "$Real"]]
    df = df[(df["Port1"] == Port3) & (df["Port2"] == Port4)]
    df = df.reset_index(drop=True)

    return Port3, Port4

# Starte die Dash-App, falls dieses Skript direkt ausgeführt wird
if __name__ == "__main__":
    app.run(debug=True)
