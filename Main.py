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

