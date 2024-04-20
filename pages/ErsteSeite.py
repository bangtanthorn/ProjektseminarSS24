import csv
import pandas as pd
from dash import html, callback
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash
import seaborn as sns


#Erstellung vom Dataframe und Bereinigung 
df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
df = df[["Year","Month","YearMonth","Port1","Port2","Route","$Value","$Real"]]
#print(df.info())
unique_values = df['Port1'].unique()
print(unique_values)

#df = df[["Year","Month","YearMonth","Port1","Port2","Route","$Value","$Real"]]
#df = df[(df["Port1"] == "Adelaide") & (df["Port2"] == "Sydney")]
#df = df[["$Value"]]
#print(df.to_string(index=False))



dash.register_page(__name__, path='/')



#Style von Buttons
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"


#Style und Farbe für die Tabs
tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '3px solid #4169E1',
    'borderleft': '3px solid #4169E1',
    'borderright': '3px solid #4169E1',
    'borderTop': '3px solid ##4169E1',
    'padding': '6px',
    'fontWeight': 'bold',
    'background-color': "#121212",
    'color': "#4169E1",
}

tab_selected_style = {
    'borderTop': '3px solid #4169E1',
    'borderBottom': '3px solid #4169E1',
    'backgroundColor': '#FFFFFF',
    'color': '#4169E1',
    'padding': '10px'
}

#Farbe für gesamtes Dash
colors = {
    'background': '#121212',
    'text': '#FFFFFF',
    "Button" : "#4169E1"
}


layout = html.Div([
    # Zwei leere Zeilen um Abstand zu bilden
    html.P("Wähle eine Strecke aus:", style={'font-size': '30px'}),
    html.H1("", style={'font-size': '30px', 'color': '#FFFFFF'}),
    html.H1("", style={'font-size': '30px', 'color': '#FFFFFF'}),
    # Layout und style für die Auswahl der Prognosemodelle in Form von einem Dropdown 
    html.Div([
        html.Div([
            html.H1("", style={'font-size': '30px', 'color': '#FFFFFF'}),
            html.P("Abflug:", style={'font-size': '30px'}),
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
                    {"label": "Port Macquari", "value": "Port Macquarie"},
                    {"label": "Proserpine", "value": "Paraburdoo"},
                    {"label": "Sunshine Coast", "value": "Sunshine Coast"},
                    {"label": "Sydney", "value": "Sydney"}
                ],
                value="Adelaide",
                id="Port1",
                style={'width': '30%',
               "margin-left": "auto", 
               "margin-right" : "auto", 
                "display" : "block"},
            ),
            html.P("Ankunft:", style={'font-size': '30px'}),

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
                    {"label": "Port Macquari", "value": "Port Macquarie"},
                    {"label": "Proserpine", "value": "Paraburdoo"},
                    {"label": "Sunshine Coast", "value": "Sunshine Coast"},
                    {"label": "Sydney", "value": "Sydney"}
                ],
           
            value="Brisbane",
            id="Port2",
            style={'width': '30%',
            "margin-left": "auto", 
            "margin-right" : "auto", 
            "display" : "block"},
            ),
            html.H1("", style={'font-size': '30px', 'color': '#FFFFFF'}),
            html.H1("", style={'font-size': '30px', 'color': '#FFFFFF'}),

            dcc.Graph(id="time-series-chart", style = {'width': '50%', "height" : '60%', "margin-left": "auto", "margin-right" : "auto", "display" : "block"}),
            
        ])
    ])
],

    #Weitere Style-Eigenschaften definieren
    style={'background-color': "#D3D3D3",
          'background-size': '100%',
          'width': '100%',
          'height':'95%',
          'font-family': 'Constantia',

          }
    )



@callback(
    Output("Port2", "options"),
    Input("Port1", "value")
)


def update_port2_options(selected_port1):
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year","Port1","Port2","$Value"]]
    filtered_df = df[df["Port1"] == selected_port1]
    port2_options = [{"label": port, "value": port} for port in filtered_df["Port2"].unique()]

    return port2_options




@callback(
    Output(component_id= "time-series-chart", component_property="figure"),
    Input(component_id= "Port1", component_property="value"),
    Input(component_id= "Port2", component_property="value"),
)

def Strecke(Port1,Port2):

    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year","Month","YearMonth","Port1","Port2","Route","$Value","$Real"]]
    df = df[(df["Port1"] == Port1) & (df["Port2"] == Port2)]
    #df = df[["$Value"]]
    df = df.reset_index(drop=True)

    fig = px.bar(df, x="Year", y=df["$Value"],template="none", labels={"Date": "Datum"},color_discrete_sequence=['#4169E1'])
    fig.update_layout(plot_bgcolor='#D3D3D3')
    return fig
