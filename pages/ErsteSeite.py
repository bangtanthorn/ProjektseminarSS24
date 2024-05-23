import csv
import pandas as pd
from dash import html, callback
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash
import seaborn as sns
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import date
from datetime import datetime
import geosphere
from geopy.distance import geodesic
from dash import dash_table



#Erstellung vom Dataframe und Bereinigung 
#df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
#df = df[["Year","Month","YearMonth","Port1","Port2","Route","$Value","$Real"]]
#print(df.info())
#unique_values = df['Port1'].unique()
#print(unique_values)

dash.register_page(__name__, path='/', name = "Fluganalyse1")


df = pd.read_csv('AUS_Fares_March2024.csv')
df_cleaned = df[["Year","Month","YearMonth","Port1","Port2","Route","$Value","$Real"]].copy()



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
    'font_size': '50px',
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
    'background': '#000000',
    'text': '#FFFFFF',
    "Button" : "#4169E1"
}

table_columns = [
    {'name': 'Year', 'id': 'Year', },
    {'name': 'Month', 'id': 'Month'},
    {'name': '$Value', 'id': '$Value'}
]


layout = html.Div([
    html.Div([
        html.Div([
        html.P(""),     
        html.P(""),  
        html.Div([  
        html.P("Wählen Sie einen Zeitraum: ", style={"font-size": "25px","margin-right": "2150px", "margin-top": "100px",'font-family': 'Constantia'}),
        dcc.DatePickerRange(
            id="year-picker",
            min_date_allowed=date(2010, 1, 1),
            max_date_allowed=date.today(),
            start_date=date(2010, 1, 1),
            end_date=date.today(),
            display_format='YYYY',
            style={"textAlign": "center", "vertical-align": "top", "margin-right": "2150px", "height": "40%"}
        ),

        html.Div([
        html.Div([
        html.P("Auswahl: ", style={"font-size": "30px", "margin-top": "500px", 'font-family': 'Constantia'}),
        html.Br(),
        dcc.Checklist(
            id='price-type-checklist',
            options={
        'MAX': 'Maximalpreise',
        'MIN': 'Minimalpreise',
        'DURCH': 'Durchschnittspreise'
        },
        value=['MAX'],
        style={"font-size": "25px", "margin-top": "30px", 'font-family': 'Constantia'}
            )
        ]),
        # html.P("Distanz:", style={'font-size': '20px', 'color': colors['text']}),
        # html.Div([
        #     html.Span("Land: ", style={'font-size': '20px', 'color': colors['text']}),
        #     html.Span(id="Distanz",style={'font-size': '20px','color': colors['Button']})
        # ], style={'display': 'inline-block', 'margin-bottom': '15px'}),
        # html.Br(),
        #html.Div([
        #html.Span("City: ", style={'font-size': '25px', 'color': colors['text']}),
        #html.Span(id="Land", style={'font-size': '25px','color': colors['Button']})
        #], style={'display': 'inline-block', 'margin-bottom': '15px'}),
        ], style={"margin-right": "2200px" })
        ]),


        ]),
        html.P(""),   
        html.P(""),   
        html.Div([
        html.Div([


        

        dcc.Graph(
                id="time-series-chart",
                style={'width': '50%', "height": '80%', "margin-left": "auto",
                       "margin-right": "auto", "color": "#696969", "margin-top": "-850px"}
            ),

        dcc.Tabs(id='tabs', value="Liniendiagramm", children=[
        dcc.Tab(label='Liniendiagramm', value='Liniendiagramm', children=[], style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Scatter-Plot', value='Scatter-Plot', children=[], style=tab_style, selected_style=tab_selected_style)],
        
        style = {'width': '50%',
                 'font-size': '80%',
                 'height': "90%",
                 "margin-left": "auto", 
                 "margin-right" : "auto", 
                 "display" : "block",
                 "margin-top": "20px",
                 #"margin-top": "-117px",
                 'font_size': '20px',
                'font-family': 'Constantia'
                 
                 }),
            dcc.Graph(
                id="Liniendiagramm",
                style={'width': '50%', "height": '80%', "margin-left": "auto",
                       "margin-right": "auto", "color": "#696969", "margin-top": "40px"}
                       #"margin-top": "-10px"
            )
        ], style={"vertical-align": "top", "margin-left": "-100","margin-top": "-600" }),
       dash_table.DataTable(
            id='table', 
            data=[], 
            columns=table_columns, 
            style_cell={
                'textAlign': 'center', 
                'color': '#FFFFFF', 
                'background': colors["background"], 
                'font_size': '15px',
                'font-family': 'Constantia'
            },
            style_header={
                'backgroundColor': '#4169E1',
                'padding': '10px',
                'color': '#FFFFFF', 
                'font-family': 'Constantia'
            },
            style_data_conditional=[
                {
            'if': {'column_id': c},
            'minWidth': '20px',  # Hier können Sie die Breite anpassen
            'maxWidth': '50px',  # Hier können Sie die Breite anpassen
            'width': '30px',     # Hier können Sie die Breite anpassen
             } for c in ['Year', 'Month', '$Value']  # Geben Sie die Spalten an, für die Sie die Breite ändern möchten
            ],
            style_table={
                'maxHeight': '920px',  # Hier können Sie die maximale Höhe der Tabelle festlegen
                'overflowY': 'scroll',
                "margin-top": "-920px",
                "margin-left": "2150px"     # Aktiviert die vertikale Scrollbar
            },
            fill_width=False
            )
            ])
    ])
])
   
#Weitere Style-Eigenschaften definieren
style={'background-color': "#121212",
      'background-size': '100%',
      'width': '100%',
      'height':'95%',
      'font-family': 'Constantia',

      }


# @callback(
#     Output(component_id= "Distanz", component_property="children"),
#     Output(component_id= "Land", component_property="children"),
#     Input(component_id= "flight_Abflug", component_property="data"),
#     Input(component_id= "flight_Ankunft", component_property="data"),
# )


# def Fluginfo(flight_Abflug, flight_Ankunft):

#     # Lese die Flughafendaten ein
#     airports = pd.read_csv("https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat", header=None, sep=',')
#     # Benenne die Spalten um
#     airports.columns = ["ID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude", "Altitude", "Timezone", "DST", "Tz database time zone", "Type", "Source"]
#     # Filtern Sie die Zeile für Gatwick
#     Port_1 = airports[airports["Name"] == flight_Abflug]
#     Port_1 = Port_1.reset_index(drop=True)
#     # Extrahieren Sie die Longitude und Latitude Werte
#     Port_1 = (Port_1["Latitude"].iloc[0], Port_1["Longitude"].iloc[0])
   

#     Port_2 = airports[airports["Name"] == flight_Ankunft]
#     # Extrahieren Sie die Longitude und Latitude Werte
#     Port_2 = (Port_2["Latitude"].iloc[0], Port_2["Longitude"].iloc[0])

#     # Berechne die Entfernung zwischen Gatwick und Heathrow
#     distance = geodesic(Port_2, Port_2).kilometers


#     Land = (Port_2["Land"].iloc[0])

#     return distance, Land



@callback(
    Output(component_id= "time-series-chart", component_property="figure"),
    Input(component_id= "flight_Abflug", component_property="data"),
    Input(component_id= "flight_Ankunft", component_property="data"),
    Input('year-picker',component_property= "start_date"),  
    Input('year-picker',component_property= "end_date"),
    
)

def Strecke(flight_Abflug, flight_Ankunft, start_date, end_date):
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Value", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df.reset_index(drop=True)

    # Extrahiere das Jahr aus den Datumsangaben und wandele sie in Integer um
    start_year = datetime.strptime(start_date, "%Y-%m-%d").year
    end_year = datetime.strptime(end_date, "%Y-%m-%d").year

    filtered_df = df.loc[(df["Year"] >= start_year) & (df["Year"] < end_year)]

    # Gruppiere nach Jahr und erhalte das Maximum für jedes Jahr
    max_values = filtered_df.groupby("Year")["$Value"].max().reset_index()

    fig = px.bar(max_values, x="Year", y="$Value", template="plotly_dark",
        labels={"Year": "Jahr", "$Value": "Wert"}, color_discrete_sequence=["#ff0000"])
    
    return fig


@callback(
    Output(component_id= "Liniendiagramm", component_property="figure"),
    Input(component_id= "flight_Abflug", component_property="data"),
    Input(component_id= "flight_Ankunft", component_property="data"),
    Input('year-picker',component_property= "start_date"),  
    Input('year-picker',component_property= "end_date"),
    Input('price-type-checklist', 'value'),
    Input('tabs', 'value'),
)

def ZweiteStrecke(flight_Abflug, flight_Ankunft, start_date, end_date, selected_Price, tabs):

    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',', dtype={"Year": int})
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Value", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]

    # Extrahiere das Jahr aus den Datumsangaben und wandele sie in Integer um
    start_year = datetime.strptime(start_date, "%Y-%m-%d").year
    end_year = datetime.strptime(end_date, "%Y-%m-%d").year
    
    filtered_df = df.loc[(df["Year"] >= start_year) & (df["Year"] < end_year)]

    # Maximum und Minimum pro Jahr extrahiere
    #min_values = filtered_df.groupby("Year")["$Value"].min()

    #fig = go.Figure([max_line, min_line])

    fig = go.Figure()

    if tabs == "Liniendiagramm": 

        if 'MAX' in selected_Price:
            # Plot für das Maximum erstellen
            max_values = filtered_df.groupby("Year")["$Value"].max()
            max_line = go.Scatter(x=max_values.index, y=max_values.values, mode='lines', name='Maximalwert')
            fig.add_trace(max_line)
    
        if 'MIN' in selected_Price:
            # Plot für das Minimum erstellen
            min_values = filtered_df.groupby("Year")["$Value"].min()
            min_line = go.Scatter(x=min_values.index, y=min_values.values, mode='lines', name='Minimalwert')
            fig.add_trace(min_line)

        if 'DURCH' in selected_Price:
            # Plot für das Minimum erstellen
            mean_values = filtered_df.groupby("Year")["$Value"].mean()
            mean_line = go.Scatter(x=mean_values.index, y=mean_values.values, mode='lines', name='Durchschnittswert')
            fig.add_trace(mean_line)
            # Figur erstellen und Linien hinzufügen
        
    if tabs == "Scatter-Plot": 

        if 'MAX' in selected_Price:
            max_values = filtered_df.groupby("Year")["$Value"].max()
            #fig = px.scatter(x=max_values.index, y=max_values.values)
            fig.add_trace(go.Scatter(x=max_values.index, y=max_values.values, mode='markers', name='Maximalwert', marker=dict(
                    size=10,
                    #color='rgba(255, 0, 0, 0.6)',
                    line=dict(
                    width=2,
                    color='DarkSlateGrey'
                      
                    ))))

        if 'MIN' in selected_Price:
            min_values = filtered_df.groupby("Year")["$Value"].min()
            #fig = px.scatter(x=max_values.index, y=max_values.values)
            fig.add_trace(go.Scatter(x=min_values.index, y=min_values.values, mode='markers', name='Minimalwert', marker=dict(
                    size=10,
                    #color='rgba(255, 0, 0, 0.6)',
                    line=dict(
                    width=2,
                    color='DarkSlateGrey'
                    ))))


        if 'DURCH' in selected_Price:
            mean_values = filtered_df.groupby("Year")["$Value"].mean()
            #fig = px.scatter(x=max_values.index, y=max_values.values)
            fig.add_trace(go.Scatter(x=mean_values.index, y=mean_values.values, mode='markers', name='Durchschnittswerte', marker=dict(
                    size=10,
                    #color='rgba(255, 0, 0, 0.6)',
                    line=dict(
                    width=2,
                    color='DarkSlateGrey'
                    ))))

    fig.update_layout(
    title="Maximum und Minimumpreise für die Strecke: {} & {}".format(flight_Abflug, flight_Ankunft),
    xaxis=dict(title="Year"),
    yaxis=dict(title="$Value"),
    template="plotly_dark"
    )

    return fig

#Funktion/Callback um die Tabelle mit den historischen Daten zu generieren
@callback(Output(component_id='table', component_property='data'),
         Input(component_id= "flight_Abflug", component_property="data"),
         Input(component_id= "flight_Ankunft", component_property="data"),
         Input('year-picker',component_property= "start_date"),  
         Input('year-picker',component_property= "end_date")
        )


def table(flight_Abflug, flight_Ankunft, start_date, end_date):
    monatsnamen = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"]
    
    # Lese die CSV-Datei in ein DataFrame
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',', dtype={"Year": int})
    
    # Auswahl der relevanten Spalten
    df = df[["Year", "Month", "Port1", "Port2", "Route", "$Value"]]
    
    start_year = datetime.strptime(start_date, "%Y-%m-%d").year
    end_year = datetime.strptime(end_date, "%Y-%m-%d").year
    # Konvertiere "start_date" und "end_date" in numerische Werte
    start_date = int(start_year)
    end_date = int(end_year)
    
    # Filtere nach Abflug- und Ankunftsort und Zeitraum
    filtered_df = df.loc[(df["Year"] >= start_date) & (df["Year"] < end_date)]
    
    # Konvertiere numerische Monatswerte in Monatsnamen
    filtered_df["Month"] = filtered_df["Month"].apply(lambda x: monatsnamen[x-1])
    
    return filtered_df.to_dict('records')