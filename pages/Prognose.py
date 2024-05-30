
import pandas as pd
from dash import html, callback
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash
import seaborn as sns
from dash import dash_table


dash.register_page(__name__, path='/dritte-seite', name = "Prognose")



layout = html.Div([
       
        html.P(""),     
        html.P(""),  

])
