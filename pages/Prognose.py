
import pandas as pd
from dash import html, callback
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash
import seaborn as sns
from dash import dash_table
from pages.LSTM import get_lstm_predictions



dash.register_page(__name__, path='/dritte-seite', name = "Prognose")



layout = html.Div([
       
        html.P(""),     
        html.P(""),  
        dcc.Graph(id="Method-Graph", style={'width': '70%', "height": '60%', "margin-left": "auto", "margin-right": "auto", "display": "block"})

])



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
