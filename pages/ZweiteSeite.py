import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
from dash import html, callback


df = pd.read_csv('AUS_Fares_March2024.csv')
df_cleaned = df[['Year', 'Month', 'Route', '$Value', '$Real']].copy()

unique_routes = df_cleaned['Route'].unique()
unique_years = df_cleaned['Year'].unique()

#app = dash.Dash(__name__)

#dash.register_page(__name__, path='/')

dash.register_page(__name__, name = "Zweite Seite")


layout = html.Div([
    html.H1("Flugpreisanalyse-Dashboard"),
    html.Div([
        html.Div([
            html.Label('Wähle eine Route:'),
            dcc.Dropdown(
                id='route-dropdown',
                options=[{'label': route, 'value': route} for route in unique_routes],
                value=unique_routes[0]  # Standardwert ist die erste Route
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('Wähle ein Jahr:'),
            dcc.Slider(
                id='year-slider',
                min=df_cleaned['Year'].min(),
                max=df_cleaned['Year'].max(),
                value=df_cleaned['Year'].min(),
                marks={str(year): str(year) for year in unique_years},
                step=None
            ),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    
    dcc.Graph(id='price-time-series'),
    dcc.Graph(id='average-price-bar')
])

@callback(
    Output('price-time-series', 'figure'),
    Output('average-price-bar', 'figure'),
    [Input('route-dropdown', 'value'),
     Input('year-slider', 'value')]
)


def update_graph(selected_route, selected_year):

    filtered_data = df_cleaned[(df_cleaned['Route'] == selected_route) & (df_cleaned['Year'] == selected_year)]
    

    line_fig = px.line(filtered_data, x='Month', y='$Value', title=f'Monatliche Preise für {selected_route} in {selected_year}')
    

    avg_prices = filtered_data.groupby('Month')['$Value'].mean().reset_index()
    bar_fig = px.bar(avg_prices, x='Month', y='$Value', title=f'Durchschnittspreise pro Monat für {selected_route} in {selected_year}')
    
    return line_fig, bar_fig

#if __name__ == '__main__':
   # app.run_server(debug=True)
