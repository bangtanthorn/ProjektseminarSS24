import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dash import dcc, html, callback, Output, Input
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import itertools
from pages.LSTM import get_lstm_predictions
from pages.LineareRegression import LineareRegression
import dash
from dash import dash_table

dash.register_page(__name__, path='/prognosen', name="Prognosen")

# Daten laden
csv_file_path_fares = 'AUS_Fares_March2024.csv'
df = pd.read_csv(csv_file_path_fares)

# Datenvorbereitung
df['YearMonth'] = pd.to_datetime(df['YearMonth'], format='%Y%m')
df = df.rename(columns={'$Value': 'Value', '$Real': 'Real'})
df = df.drop_duplicates(subset=['YearMonth', 'Route'])

# SARIMA-Modellparameter-Optimierung
def optimize_sarima(endog, seasonal_periods):
    p = range(0, 3)
    d = range(0, 2)
    q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], seasonal_periods) for x in pdq]

    best_aic = float("inf")
    best_params = None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(endog, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (param, param_seasonal)
            except Exception as e:
                print(f"Parameter combination {param} and {param_seasonal} failed with error: {e}")
    return best_params

def prepare_data(df, flight_Abflug, flight_Ankunft):
    route_df = df[(df['Port1'] == flight_Abflug) & (df['Port2'] == flight_Ankunft)].copy()
    if route_df.empty:
        raise ValueError(f"No data found for the route from {flight_Abflug} to {flight_Ankunft}")
    route_df.set_index('YearMonth', inplace=True)
    route_df = route_df[~route_df.index.duplicated(keep='first')]
    if 'Real' not in route_df.columns:
        raise ValueError(f"The 'Real' column is missing in the data for the route from {flight_Abflug} to {flight_Ankunft}")
    route_df = route_df.asfreq('MS').interpolate()
    return route_df

def create_figure(route_df, forecast_df, metrics, flight_Abflug, flight_Ankunft):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=route_df.index, y=route_df['Real'], mode='lines', name='Tatsächliche Preise'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines+markers', name='Prognose', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower CI'], mode='lines', line=dict(color='grey'), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper CI'], mode='lines', line=dict(color='grey'), fill='tonexty', showlegend=False))

    metrics_table = f"<b>Metriken</b><br>MSE: {metrics['mse']:.2f}<br>MAE: {metrics['mae']:.2f}<br>RMSE: {metrics['rmse']:.2f}"
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='rgba(0,0,0,0)'), showlegend=True, name=metrics_table, hoverinfo='none'))
    fig.update_layout(title=f"SARIMA-Prognose für die Strecke: {flight_Abflug} & {flight_Ankunft}", xaxis_title='Jahr', yaxis_title='Preis ($)', template='plotly_dark', legend=dict(x=1, y=1, traceorder='normal', font=dict(size=12, color="white")), margin=dict(r=200))
    return fig

layout = html.Div([
    dcc.Graph(id='price-forecast-graph', style={'width': '70%', 'height': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block'}),
    html.Div(id='error-message', style={'color': 'red'}),
    dcc.Graph(id="Method-Graph", style={'width': '70%', 'height': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block', 'margin-top': '100'}),
    dcc.Graph(id="LineareRegression", style={'width': '70%', 'height': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block', 'margin-top': '100'}),
    dcc.Graph(id="All-Method-Graph", style={'width': '70%', 'height': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block', 'margin-top': '100'})  # Neuer Graph für alle Prognosen
], style={'background-color': "#121212", 'width': '100%', 'height': '95%', 'font-family': 'Constantia', "margin-top": "200px"})

@callback([Output('price-forecast-graph', 'figure'), Output('error-message', 'children')], [Input('Port3', 'value'), Input('Port4', 'value')])
def update_graph(flight_Abflug, flight_Ankunft):
    try:
        route_df = prepare_data(df, flight_Abflug, flight_Ankunft)
        best_params = optimize_sarima(route_df['Real'], 12)
        model = SARIMAX(route_df['Real'], order=best_params[0], seasonal_order=best_params[1])
        results = model.fit()
        forecast = results.get_forecast(steps=12)
        forecast_df = pd.DataFrame({'Forecast': forecast.predicted_mean, 'Lower CI': forecast.conf_int().iloc[:, 0], 'Upper CI': forecast.conf_int().iloc[:, 1]}, index=pd.date_range(start=route_df.index[-1], periods=13, freq='MS')[1:])
        mse = mean_squared_error(route_df['Real'], results.fittedvalues)
        mae = mean_absolute_error(route_df['Real'], results.fittedvalues)
        rmse = np.sqrt(mse)
        metrics = {'mse': mse, 'mae': mae, 'rmse': rmse}
        fig = create_figure(route_df, forecast_df, metrics, flight_Abflug, flight_Ankunft)
        return fig, ""
    except Exception as e:
        error_message = f"Fehler bei der Prognose für die Strecke {flight_Abflug} nach {flight_Ankunft}: {str(e)}"
        fig = go.Figure()
        fig.update_layout(title='Fehler bei der Prognose', xaxis_title='Datum', yaxis_title='Preis ($)', template='plotly_dark')
        return fig, error_message

@callback(Output('Method-Graph', 'figure'), [Input('Port3', 'value'), Input('Port4', 'value')])
def update_lstm_graph(flight_Abflug, flight_Ankunft):
    fig = get_lstm_predictions(flight_Abflug, flight_Ankunft)
    return fig


@callback(Output('LineareRegression', 'figure'), [Input('Port3', 'value'), Input('Port4', 'value')])
def update_LineareRegression(flight_Abflug, flight_Ankunft):
    fig = LineareRegression(flight_Abflug, flight_Ankunft)
    return fig



@callback(Output('All-Method-Graph', 'figure'), [Input('Port3', 'value'), Input('Port4', 'value')])
def update_all_method_graph(flight_Abflug, flight_Ankunft):
    try:
        # Bereite den Gesamtvergleichsgraph vor
        fig = go.Figure()
        # SARIMA Prognose
        sarima_fig = update_graph(flight_Abflug, flight_Ankunft)[0]
        for trace in sarima_fig['data']:
            fig.add_trace(trace)
            fig.add_annotation(
            xref="paper", yref="paper",
            x=0.7, y=0.4,
            text="SARIMA",  # Hier den gewünschten Text einfügen
            showarrow=False,
            font=dict(size=16, color="orange"),  # Hier können Sie Schriftgröße und Farbe anpassen
            bgcolor="rgba(0, 0, 0, 0)"  # Transparenter Hintergrund
            )

                

        # LSTM Prognose
        lstm_fig = update_lstm_graph(flight_Abflug, flight_Ankunft)
        for trace in lstm_fig['data']:
            fig.add_trace(trace)
            fig.add_annotation(
            xref="paper", yref="paper",
            x=0.95, y=0.05,
            text="LSTM",  # Hier den gewünschten Text einfügen
            showarrow=False,
            font=dict(size=16, color="yellow"),  # Hier können Sie Schriftgröße und Farbe anpassen
            bgcolor="rgba(0, 0, 0, 0)"  # Transparenter Hintergrund
            )


        # Lineare Regression Prognose
        lr_fig = update_LineareRegression(flight_Abflug, flight_Ankunft)
        for trace in lr_fig['data']:
            fig.add_trace(trace)
            fig.add_annotation(
            xref="paper", yref="paper",
            x=0.95, y=0.65,
            text="Lineare Regression",  # Hier den gewünschten Text einfügen
            showarrow=False,
            font=dict(size=16, color="red"),  # Hier können Sie Schriftgröße und Farbe anpassen
            bgcolor="rgba(0, 0, 0, 0)"  # Transparenter Hintergrund
            )



        fig.update_layout(title=f'Vergleich der Prognosemethoden für {flight_Abflug}-{flight_Ankunft}', xaxis_title='Datum', yaxis_title='Preis ($)', template='plotly_dark', height=600, showlegend=False) #geändert !!
        fig.update_xaxes(title="Jahr", range=['2022-01-01', '2025-05-01'])
        

        return fig

    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title='Fehler bei der Prognose', xaxis_title='Datum', yaxis_title='Preis ($)', template='plotly_dark')
        return fig












# table_columns = [
#     {'name': 'Monat', 'id': 'Monat'},
#     {'name': 'LSTM', 'id': 'LSTM'},
#     {'name': 'SLR', 'id': 'SLR'},
#     {'name': 'Prognose', 'id': 'Prognose'}
# ]

# csv_file_path_fares = 'AUS_Fares_March2024.csv'
# df = pd.read_csv(csv_file_path_fares)

# # Datenvorbereitung
# df['YearMonth'] = pd.to_datetime(df['YearMonth'], format='%Y%m')
# df = df.rename(columns={'$Value': 'Value', '$Real': 'Real'})
# df = df.drop_duplicates(subset=['YearMonth', 'Route'])

# def prepare_data(df, flight_Abflug, flight_Ankunft):
#     route_df = df[(df['Port1'] == flight_Abflug) & (df['Port2'] == flight_Ankunft)].copy()
#     if route_df.empty:
#         raise ValueError(f"No data found for the route from {flight_Abflug} to {flight_Ankunft}")
#     route_df.set_index('YearMonth', inplace=True)
#     route_df = route_df[~route_df.index.duplicated(keep='first')]
#     if 'Real' not in route_df.columns:
#         raise ValueError(f"The 'Real' column is missing in the data for the route from {flight_Abflug} to {flight_Ankunft}")
#     route_df = route_df.asfreq('MS').interpolate()
#     return route_df

# layout = html.Div([
#     dcc.Graph(id='price-forecast-graph', style={'width': '70%', 'height': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block'}),
#     html.Div(id='error-message', style={'color': 'red'}),
#     dcc.Graph(id="Method-Graph", style={'width': '70%', 'height': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block', 'margin-top': '100'}),
#     dcc.Graph(id="LineareRegression", style={'width': '70%', 'height': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block', 'margin-top': '100'}),
#     dcc.Graph(id="All-Method-Graph", style={'width': '70%', 'height': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block', 'margin-top': '100'}),
#     dash_table.DataTable(
#             id='tablePrognose', 
#             data=[], 
#             columns=table_columns, 
#             style_cell={
#                 'textAlign': 'center', 
#                 'color': '#FFFFFF', 
#                 #'background': colors["background"], 
#                 'font_size': '15px',
#                 'font-family': 'Constantia'
#             }),  # Neuer Graph für alle Prognosen
# ], style={'background-color': "#121212", 'width': '100%', 'height': '95%', 'font-family': 'Constantia', "margin-top": "200px"})

# @callback(Output('Method-Graph', 'figure'), [Input('Port3', 'value'), Input('Port4', 'value')])
# def update_lstm_graph(flight_Abflug, flight_Ankunft):
#     fig = get_lstm_predictions(flight_Abflug, flight_Ankunft)
#     #print(f"LSTM Figure Data for {flight_Abflug}-{flight_Ankunft}: {fig['data']}")
#     return fig

# @callback(Output('LineareRegression', 'figure'), [Input('Port3', 'value'), Input('Port4', 'value')])
# def update_LineareRegression(flight_Abflug, flight_Ankunft):
#     fig = LineareRegression(flight_Abflug, flight_Ankunft)
#     return fig

# @callback(Output('All-Method-Graph', 'figure'), [Input('Port3', 'value'), Input('Port4', 'value')])
# def update_all_method_graph(flight_Abflug, flight_Ankunft):
#     try:
#         # Bereite den Gesamtvergleichsgraph vor
#         fig = go.Figure()
        
#         # LSTM Prognose
#         lstm_fig = update_lstm_graph(flight_Abflug, flight_Ankunft)
#         for trace in lstm_fig['data']:
#             fig.add_trace(trace)
#             fig.add_annotation(
#             xref="paper", yref="paper",
#             x=0.95, y=0.05,
#             text="LSTM",  # Hier den gewünschten Text einfügen
#             showarrow=False,
#             font=dict(size=16, color="yellow"),  # Hier können Sie Schriftgröße und Farbe anpassen
#             bgcolor="rgba(0, 0, 0, 0)"  # Transparenter Hintergrund
#             )
        
#         # Lineare Regression Prognose
#         lr_fig = update_LineareRegression(flight_Abflug, flight_Ankunft)
#         for trace in lr_fig['data']:
#             fig.add_trace(trace)
#             fig.add_trace(trace)
#             fig.add_annotation(
#             xref="paper", yref="paper",
#             x=0.95, y=0.65,
#             text="Lineare Regression",  # Hier den gewünschten Text einfügen
#             showarrow=False,
#             font=dict(size=16, color="red"),  # Hier können Sie Schriftgröße und Farbe anpassen
#             bgcolor="rgba(0, 0, 0, 0)"  # Transparenter Hintergrund
#             )


        
#         fig.update_layout(title=f'Vergleich der Prognosemethoden für {flight_Abflug}-{flight_Ankunft}', xaxis_title='Datum', yaxis_title='Preis ($)', template='plotly_dark', height=600, showlegend=False) #geändert !!
#         fig.update_xaxes(title="Jahr", range=['2022-01-01', '2025-05-01'])
#         #fig.update_yaxes(title="Preis ($)", range=[100, 1000])
       

#         return fig
#     except Exception as e:
#         fig = go.Figure()
#         fig.update_layout(title='Fehler bei der Prognose', xaxis_title='Datum', yaxis_title='Preis ($)', template='plotly_dark') 
#         return fig
    





    
# @callback(
#     Output(component_id='tablePrognose', component_property='data'),
#     Input(component_id='Port3', component_property='value'),
#     Input(component_id='Port4', component_property='value'),
#     allow_duplicate=True
# )
# def Prognosetable(flight_Abflug, flight_Ankunft):
#     try:
#         # Prognosemethoden aktualisieren
#         lstm_fig = update_lstm_graph(flight_Abflug, flight_Ankunft)
#         lr_fig = update_LineareRegression(flight_Abflug, flight_Ankunft)

#         # Extrahiere die Daten aus den Graphen
#         lstm_data = lstm_fig['data'][0]['y']
#         lr_data = lr_fig['data'][0]['y']

#         # Durchschnitt der Prognosen berechnen
#         avg_data = [(lstm_data[i] + lr_data[i]) / 2 for i in range(len(lstm_data))]

#         # Monate extrahieren
#         months = [month.strftime('%Y-%m') for month in df['YearMonth']]

#         # Daten für die Tabelle vorbereiten
#         table_data = []
#         for i in range(len(df)):
#             table_data.append({
#                 'Monat': months[i],
#                 'LSTM': f'${lstm_data[i]:.2f}',
#                 'SLR': f'${lr_data[i]:.2f}',
#                 'Prognose': f'${avg_data[i]:.2f}'
#             })

#         return table_data

#     except Exception as e:
#         print(f'Fehler beim Aktualisieren der Tabelle: {str(e)}')
#         return []



    

