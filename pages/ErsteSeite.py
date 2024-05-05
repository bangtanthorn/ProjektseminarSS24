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


#Erstellung vom Dataframe und Bereinigung 
df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
df = df[["Year","Month","YearMonth","Port1","Port2","Route","$Value","$Real"]]
#print(df.info())
unique_values = df['Port1'].unique()
print(unique_values)

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


layout = html.Div([
    # Zwei leere Zeilen um Abstand zu bilden
    
    html.H1("", style={'font-size': '30px', 'color': '#FFFFFF'}),
    html.H1("", style={'font-size': '30px', 'color': '#FFFFFF'}),
    # Layout und style für die Auswahl der Prognosemodelle in Form von einem Dropdown 
    
html.Div([
    dcc.Dropdown(
        id='year-dropdown',
         options=(
            [{'label': 'Insgesamt', 'value': 'Insgesamt'}] + [{'label': str(year), 'value': year} for year in df['Year'].unique()]
        ),
        value='Insgesamt',  # Standardmäßig "Insgesamt" auswählen
        clearable=False,  
        style={'width': '55%', 
               "display" : "block",
               "color": "black",
               'font-family': 'Constantia'},
    )
]),
    
    html.Div([
        html.Div([
            html.H1("", style={'font-size': '30px', 'color': '#FFFFFF'}),
            #html.P("Abflug:", style={'font-size': '30px', 'color': '#FFFFFF'}),
            
          
            html.H1("", style={'font-size': '30px', 'color': '#FFFFFF'}),
            html.H1("", style={'font-size': '30px', 'color': '#FFFFFF'}),

            

            dcc.Graph(id="time-series-chart", style = {'width': '60%', "height" : '70%', "margin-left": "auto", "margin-right" : "auto", "display" : "block", "color":"#696969"}),
           
            html.Div([
               
               
                dcc.Graph(id="Liniendiagramm", style = {'width': '60%', "height" : '70%', "margin-left": "auto", "margin-right" : "auto", "display" : "block", "color":"#696969"})
            ])
        ])
    ])
],

#Weitere Style-Eigenschaften definieren
style={'background-color': "#121212",
      'background-size': '100%',
      'width': '100%',
      'height':'95%',
      'font-family': 'Constantia',

      }
)

#@callback(
#    Output("Port2", "options"),
#    Input("Port1", "value")
#)


#def update_port2_options(selected_port1):
    #df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    #df = df[["Year","Port1","Port2","$Value"]]
    #filtered_df = df[df["Port1"] == selected_port1]
    #port2_options = [{"label": port, "value": port} for port in filtered_df["Port2"].unique()]

    #return port2_options




@callback(
    Output(component_id= "time-series-chart", component_property="figure"),
    Input(component_id= "flight_Abflug", component_property="data"),
    Input(component_id= "flight_Ankunft", component_property="data"),
)

def Strecke(flight_Abflug,flight_Ankunft):

    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year","Month","YearMonth","Port1","Port2","Route","$Value","$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df.reset_index(drop=True)
    fig = px.bar(df, x="Year", y=df["$Value"],template="plotly_dark", labels={"Date": "Datum"},color_discrete_sequence=["#ff0000"])
    #fig.add_trace(go.Scatter(x=df["Year"], y=df["$Value"], mode='lines', line=dict(color='red')))

    return fig


@callback(
    Output(component_id= "Liniendiagramm", component_property="figure"),
    Input(component_id= "flight_Abflug", component_property="data"),
    Input(component_id= "flight_Ankunft", component_property="data"),
    Input('year-dropdown', 'value'),  
)
def Strecke(flight_Abflug, flight_Ankunft, selected_year, ):
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Value", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]


    fig = px.line(df, x="Month", y="$Value", title=f'Jährliche monatliche Preise für {selected_year}',
                  labels={"Month": "Monat", "$Value": "Wert"}, template="plotly_dark")
    return fig