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
from dash import callback_context
import tensorflow as tf

dash.register_page(__name__, path='/prognosen', name="Prognosen")


table_columns = [
    {'name': 'Metrik', 'id': 'Metrik'},
    {'name': 'Long-Short-Term-memory', 'id': 'LSTM'},
    {'name': 'Saisional lineare Regression', 'id': 'SLR'},
    {'name': 'Seasonal-ARIMA', 'id': 'SARIMA'}
]

# Definiere die Zeilen für die Tabelle (Metriken) mit initialen Werten
table_rows = [
    {'Metrik': 'MAE', 'LSTM': '','SLR': '', 'SARIMA': ''},
    {'Metrik': 'MSE', 'LSTM': '','SLR': '',  'SARIMA': ''},
    {'Metrik': 'RMSE', 'LSTM': '','SLR': '',  'SARIMA': ''}
]

layout = html.Div([
    html.Div([
        dash_table.DataTable(
            id='table_Metriken',
            columns=table_columns,
            data=table_rows,
            style_cell={
                'textAlign': 'center',
                'color': '#FFFFFF',
                'backgroundColor': "#121212",
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
                    'if': {'column_id': c['id']},
                    'minWidth': '20px',
                    'maxWidth': '30px',
                    'width': '30px'
                } for c in table_columns
            ],
            style_table={
                'marginTop': '50px',
                'width': '100%',
                'marginLeft': '20px'
            },
            fill_width=False
        ),
    html.Div([
    html.Span("Beste Methode:", style={'font-size': '25px', 'margin-right': '10px'}),
    html.Span(id="BestMethod", style={'font-size': '25px', 'color': 'green', 'padding-left': '15px'})
], style={'display': 'flex', 'align-items': 'center', 'margin-top': '15px'}),
    dcc.Store(id='metrics-store'),
    dcc.Store(id='forecast-store'),
    dash_table.DataTable(
            id='table_BestPrognose',
            columns=[
                {'name': 'Monat,Jahr', 'id': 'Monat,Jahr'},
                {'name': 'Prognostizierte Werte', 'id': 'Prognostizierte Werte in $'}
            ],
            data=[],
            style_cell={
                'textAlign': 'center',
                'color': '#FFFFFF',
                'backgroundColor': "#121212",
                'font_size': '15px',
                'font-family': 'Constantia'
            },
            style_header={
                'backgroundColor': '#4169E1',
                'padding': '10px',
                'color': '#FFFFFF',
                'font-family': 'Constantia'
            },

            style_table={
                'marginTop': '50px',
                'width': '100%',
                'marginLeft': '20px'
            },
            fill_width=False
        ),
    ], style={'display': 'inline-block', 'width': '25%', 'verticalAlign': 'top'}),
    
    html.Div([
        dcc.Graph(id="All-Method-Graph", style={'width': '90%', 'height': '80%', 'marginTop': '50px'}),
    ], style={'display': 'inline-block', 'width': '70%', 'verticalAlign': 'top'}),
    
    html.Div(id='error-message', style={'color': 'red', 'textAlign': 'center', 'font-family': 'Constantia'}),
    
    html.Div([
        dcc.Graph(id="Method-Graph", style={'width': '33%', 'height': '60%', 'display': 'inline-block', 'margin-top': '20px'}),
        dcc.Graph(id="LineareRegression", style={'width': '33%', 'height': '60%', 'display': 'inline-block', 'margin-top': '20px'}),
        dcc.Graph(id='price-forecast-graph', style={'width': '33%', 'height': '60%', 'display': 'inline-block', 'margin-top': '20px'}),
    ], style={'textAlign': 'center'}),
], style={'background-color': "#121212", 'width': '100%', 'height': '100%', 'font-family': 'Constantia'})




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
        #raise ValueError(f"No data found for the route from {flight_Abflug} to {flight_Ankunft}")
        raise ValueError(f"Warten sie bis das Dashboard neu geladen hat für die Strecke {flight_Abflug} nach {flight_Ankunft}")
    route_df.set_index('YearMonth', inplace=True)
    route_df = route_df[~route_df.index.duplicated(keep='first')]
    if 'Real' not in route_df.columns:
        raise ValueError(f"The 'Real' column is missing in the data for the route from {flight_Abflug} to {flight_Ankunft}")
    route_df = route_df.asfreq('MS').interpolate()
    return route_df


def create_figure(route_df, forecast_df, metrics, flight_Abflug, flight_Ankunft):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=route_df.index, y=route_df['Real'], mode='lines', name='Tatsächliche Preise'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines+markers', name='Prognose', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower CI'], mode='lines', line=dict(color='grey'), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper CI'], mode='lines', line=dict(color='grey'), fill='tonexty', showlegend=False))
    # Verbindungslinie zwischen letztem historischen Punkt und erstem Vorhersagepunkt
    fig.add_trace(go.Scatter(
        x=[route_df.index[-1], forecast_df.index[0]],
        y=[route_df['Real'].iloc[-1], forecast_df['Forecast'].iloc[0]],
        mode='lines',
        line=dict(color='orange', dash='dash'),
        showlegend=False 
    ))

    metrics_table = f"<b>Metriken</b><br>MSE: {metrics['normalized_mse']:.2f}<br>MAE: {metrics['normalized_mae']:.2f}<br>RMSE: {metrics['normalized_rmse']:.2f}"
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines+markers', marker=dict(size=10, color='rgba(0,0,0,0)'), showlegend=True, name=metrics_table, hoverinfo='none'))
    fig.update_layout(title=f"SARIMA-Prognose für die Strecke: {flight_Abflug} & {flight_Ankunft}", xaxis_title='Jahr', yaxis_title='Preis ($)', template='plotly_dark', legend=dict(x=1, y=1, traceorder='normal', font=dict(size=12, color="white")), margin=dict(r=200))
    
    return fig





@callback([
    Output('price-forecast-graph', 'figure'), 
    Output('error-message', 'children'),
    Output('metrics-store', 'data'),
    Output('forecast-store', 'data'),
], 
[Input('Port3', 'value'), 
 Input('Port4', 'value')]
)


def update_graph(flight_Abflug, flight_Ankunft):
    try:
        route_df = prepare_data(df, flight_Abflug, flight_Ankunft)
        best_params = optimize_sarima(route_df['Real'], 12)
        model = SARIMAX(route_df['Real'], order=best_params[0], seasonal_order=best_params[1])
        results = model.fit()
        forecast = results.get_forecast(steps=12)
        forecast_df = pd.DataFrame({'Forecast': forecast.predicted_mean, 'Lower CI': forecast.conf_int().iloc[:, 0], 'Upper CI': forecast.conf_int().iloc[:, 1]}, index=pd.date_range(start=route_df.index[-1], periods=13, freq='MS')[1:])
        forecast_df_new = forecast_df["Forecast"]
        print("Forecast SARIMA")
        print(forecast_df_new)

        mse = mean_squared_error(route_df['Real'], results.fittedvalues)
        mae = mean_absolute_error(route_df['Real'], results.fittedvalues)
        rmse = np.sqrt(mse)

        # Normalisierung der Metriken
        max_value = np.max(route_df['Real']) - np.min(route_df['Real'])
        max_mse = max_value ** 2
        max_rmse = np.sqrt(max_mse)

        normalized_mae = mae / max_value if max_value != 0 else 0
        normalized_mse = mse / max_mse if max_mse != 0 else 0
        normalized_rmse = rmse / max_rmse if max_rmse != 0 else 0

        metrics = {'normalized_mse': normalized_mse, 'normalized_mae': normalized_mae, 'normalized_rmse': normalized_rmse}

        fig = create_figure(route_df, forecast_df, metrics, flight_Abflug, flight_Ankunft)
        fig.update_layout(height=600) 

        return fig, "", metrics, forecast_df_new

    except Exception as e:
        error_message = f"Fehler bei der Prognose für die Strecke {flight_Abflug} nach {flight_Ankunft}: {str(e)}"
        fig = go.Figure()
        fig.update_layout(title='Fehler bei der Prognose', xaxis_title='Datum', yaxis_title='Preis ($)', template='plotly_dark')

        return fig, error_message, {}, {}





@callback(Output('Method-Graph', 'figure'), [Input('Port3', 'value'), Input('Port4', 'value')])
def update_lstm_graph(flight_Abflug, flight_Ankunft):
    fig, _, _, _,_ = get_lstm_predictions(flight_Abflug, flight_Ankunft)
 
    return fig



@callback(Output('LineareRegression', 'figure'), [Input('Port3', 'value'), Input('Port4', 'value')])
def update_LineareRegression(flight_Abflug, flight_Ankunft):
    fig, _, _, _,_ = LineareRegression(flight_Abflug, flight_Ankunft)

    return fig



@callback(Output('All-Method-Graph', 'figure'), [Input('Port3', 'value'), Input('Port4', 'value')])
def update_all_method_graph(flight_Abflug, flight_Ankunft):
    try:
        # Bereite den Gesamtvergleichsgraph vor
        fig = go.Figure()
        # SARIMA Prognose
                # SARIMA Prognose
        sarima_fig = update_graph(flight_Abflug, flight_Ankunft)[0]
        for trace in sarima_fig['data']:
            if trace['name'] == 'Prognose':  # Übernehmen Sie nur die Prognosekurve
                fig.add_trace(trace)
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.99, y=0.15,
                    text="SARIMA",
                    showarrow=False,
                    font=dict(size=16, color="orange"),
                    bgcolor="rgba(0, 0, 0, 0)"
                )

                # Verbindungslinie hinzufügen
                if 'x' in trace and 'y' in trace and len(trace['x']) > 0 and len(trace['y']) > 0:
                    fig.add_trace(go.Scatter(
                        x=[sarima_fig['data'][0]['x'][-1], trace['x'][0]],
                        y=[sarima_fig['data'][0]['y'][-1], trace['y'][0]],
                        mode='lines',
                        line=dict(color='orange', dash='dash'),
                        showlegend=False,
                        name='Verbindungslinie'
                    ))

        # LSTM Prognose
        # LSTM Prognose
        lstm_fig = update_lstm_graph(flight_Abflug, flight_Ankunft)
        for trace in lstm_fig['data']:
            if trace['name'] == 'Prognose':  # Nur die Prognosekurve
                # Hier das Datum auf den Anfang des Monats setzen
                trace['x'] = [date.replace(day=1) for date in trace['x']]
            fig.add_trace(trace)
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.99, y=0.05,
                text="LSTM",  # Hier den gewünschten Text einfügen
                showarrow=False,
                font=dict(size=16, color="yellow"),  # Hier können Sie Schriftgröße und Farbe anpassen
                bgcolor="rgba(0, 0, 0, 0)"  # Transparenter Hintergrund
            )

        # Verbindungslinie für LSTM hinzufügen
        if lstm_fig['data']:
            last_historical_date = lstm_fig['data'][0]['x'][-1].replace(day=1)
            first_forecast_date = lstm_fig['data'][1]['x'][0]
            fig.add_trace(go.Scatter(
                x=[last_historical_date, first_forecast_date],
                y=[lstm_fig['data'][0]['y'][-1], lstm_fig['data'][1]['y'][0]],
                mode='lines',
                line=dict(color='yellow', dash='dash'),
                showlegend=False,
                name='Verbindungslinie'
            ))


        # Lineare Regression Prognose
        lr_fig = update_LineareRegression(flight_Abflug, flight_Ankunft)
        for trace in lr_fig['data']:
            fig.add_trace(trace)
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.99, y=0.10,
                text="Lineare Regression",
                showarrow=False,
                font=dict(size=16, color="red"),
                bgcolor="rgba(0, 0, 0, 0)"
            )

        fig.update_layout(title=f'Vergleich der Prognosemethoden für {flight_Abflug}-{flight_Ankunft}', xaxis_title='Datum', yaxis_title='Preis ($)', template='plotly_dark', height=600, showlegend=False)
        fig.update_xaxes(title="Jahr", range=['2024-01-01', '2025-05-01'])

        return fig

    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title='Fehler bei der Prognose', xaxis_title='Datum', yaxis_title='Preis ($)', template='plotly_dark')
        return fig



# Callback für die Tabelle (Metriken)
@callback(
    Output(component_id='table_Metriken', component_property='data'),
    Output("BestMethod", "children"),
    Output(component_id='table_BestPrognose', component_property='data'),
    [
        Input(component_id='flight_Abflug', component_property="data"),
        Input(component_id='flight_Ankunft', component_property="data"),
        Input('metrics-store', 'data'),
        Input('forecast-store', 'data'),
    ]
)
def update_table_Metriken(flight_Abflug, flight_Ankunft, metrics, forecast):

    try:

        # LSTM-Vorhersagen und Metriken abrufen
        _, normalized_mae_lstm, normalized_mse_lstm, normalized_rmse_lstm, lstm_predictions = get_lstm_predictions(flight_Abflug, flight_Ankunft)

        # Lineare Regression Vorhersagen und Metriken abrufen
        _, normalized_mae_lr, normalized_mse_lr, normalized_rmse_lr, lr_predictions = LineareRegression(flight_Abflug, flight_Ankunft)

        # Runden der Metriken
        rounded_mae_lstm = round(normalized_mae_lstm, 2)
        rounded_mse_lstm = round(normalized_mse_lstm, 2)
        rounded_rmse_lstm = round(normalized_rmse_lstm, 2)
        rounded_mae_lr = round(normalized_mae_lr, 2)
        rounded_mse_lr = round(normalized_mse_lr, 2)
        rounded_rmse_lr = round(normalized_rmse_lr, 2)

        # SARIMA Metriken runden
        rounded_mae_sarima = round(metrics['normalized_mae'], 2)
        rounded_mse_sarima = round(metrics['normalized_mse'], 2)
        rounded_rmse_sarima = round(metrics['normalized_rmse'], 2)

        # print("METRIK SARIMA")
        # print(rounded_mae_sarima)
        # print(rounded_mse_sarima)
        # print(rounded_rmse_sarima)

        # Erstellen der Daten für die Tabelle
        metrics_data = [
            {'Metrik': 'MAE:', 'LSTM': rounded_mae_lstm, 'SLR': rounded_mae_lr, 'SARIMA': rounded_mae_sarima},
            {'Metrik': 'MSE:', 'LSTM': rounded_mse_lstm, 'SLR': rounded_mse_lr, 'SARIMA': rounded_mse_sarima},
            {'Metrik': 'RMSE:', 'LSTM': rounded_rmse_lstm, 'SLR': rounded_rmse_lr, 'SARIMA': rounded_rmse_sarima}
        ]

        # Durchschnittliche Metriken für jede Methode berechnen
        avg_mae_lstm = (rounded_mae_lstm + rounded_mse_lstm + rounded_rmse_lstm) / 3
        avg_mae_lr = (rounded_mae_lr + rounded_mse_lr + rounded_rmse_lr) / 3
        avg_mae_sarima = (rounded_mae_sarima + rounded_mse_sarima + rounded_rmse_sarima) / 3 if (rounded_mae_sarima and rounded_mse_sarima and rounded_rmse_sarima) else None

        # Ermittle die beste Methode basierend auf dem Durchschnittswert
        best_method = 'SARIMA' if (avg_mae_sarima and avg_mae_sarima <= avg_mae_lstm and avg_mae_sarima <= avg_mae_lr) else 'Long-Short-Term-Memory' if avg_mae_lstm <= avg_mae_lr else 'Saisionale Lineare Regression'

        # Prepare forecast table data for table_BestPrognose
        table_BestPrognose_data = []
        forecast_months = pd.date_range(start='2024-04-01', periods=5, freq='MS').strftime('%B %Y')

        # print("lstm prediction")
        # print(lstm_predictions)

        print("prediction")
        print(forecast[:5])

        if best_method == 'Long-Short-Term-Memory':
            predictions = lstm_predictions[:7]
        elif best_method == "Saisionale Lineare Regression":
            predictions = lr_predictions[:7]
        elif best_method == 'SARIMA':
            predictions = forecast[:7]
        

        for month, forecast in zip(forecast_months, predictions):
            table_BestPrognose_data.append({'Monat,Jahr': month, 'Prognostizierte Werte in $': round(forecast, 2)})  # forecast is a single-item array

        return metrics_data, best_method, table_BestPrognose_data

    except Exception as e:
        print("Fehler:", e)
        return [], "", []























# table_columns = [
#     {'name': 'Metrik', 'id': 'Metrik'},
#     {'name': 'Long-Short-Term-memory', 'id': 'LSTM'},
#     {'name': 'Saisional lineare Regression', 'id': 'SLR'},
#     {'name': 'Seasonal-ARIMA', 'id': 'SARIMA'}
# ]

# # Definiere die Zeilen für die Tabelle (Metriken) mit initialen Werten
# table_rows = [
#     {'Metrik': 'MAE', 'LSTM': '','SLR': '', 'SARIMA': ''},
#     {'Metrik': 'MSE', 'LSTM': '','SLR': '',  'SARIMA': ''},
#     {'Metrik': 'RMSE', 'LSTM': '','SLR': '',  'SARIMA': ''}
# ]

# table_columns_Best = [
#     {'name': 'Monat', 'id': 'Monat'},
#     {'name': 'Prognosewert', 'id': 'Prognosewert'},

# ]




# # Daten laden
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

# def create_figure(route_df, forecast_df, metrics, flight_Abflug, flight_Ankunft):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=route_df.index, y=route_df['Real'], mode='lines', name='Tatsächliche Preise'))
#     fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines+markers', name='Prognose', line=dict(color='orange', dash='dash')))
#     fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower CI'], mode='lines', line=dict(color='grey'), showlegend=False))
#     fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper CI'], mode='lines', line=dict(color='grey'), fill='tonexty', showlegend=False))

#     metrics_table = f"<b>Metriken</b><br>MSE: {metrics['mse']:.2f}<br>MAE: {metrics['mae']:.2f}<br>RMSE: {metrics['rmse']:.2f}"
#     fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='rgba(0,0,0,0)'), showlegend=True, name=metrics_table, hoverinfo='none'))
#     fig.update_layout(title=f"Prognose für die Strecke: {flight_Abflug} & {flight_Ankunft}", xaxis_title='Jahr', yaxis_title='Preis ($)', template='plotly_dark', legend=dict(x=1, y=1, traceorder='normal', font=dict(size=12, color="white")), margin=dict(r=200))
#     return fig


# layout = html.Div([
#     html.Div([
#         dash_table.DataTable(
#             id='table_Metriken',
#             columns=[
#                 {'name': 'Metrik', 'id': 'Metrik'},
#                 {'name': 'LSTM', 'id': 'LSTM'},
#                 {'name': 'SLR', 'id': 'SLR'},
#                 {'name': 'SARIMA', 'id': 'SARIMA'}
#             ],
#             data=[],
#             style_cell={
#                 'textAlign': 'center',
#                 'color': '#FFFFFF',
#                 'backgroundColor': "#121212",
#                 'font_size': '15px',
#                 'font-family': 'Constantia'
#             },
#             style_header={
#                 'backgroundColor': '#4169E1',
#                 'padding': '10px',
#                 'color': '#FFFFFF',
#                 'font-family': 'Constantia'
#             },
#             style_data_conditional=[
#                 {
#                     'if': {'column_id': c['id']},
#                     'minWidth': '20px',
#                     'maxWidth': '30px',
#                     'width': '30px'
#                 } for c in [
#                     {'name': 'Metrik', 'id': 'Metrik'},
#                     {'name': 'LSTM', 'id': 'LSTM'},
#                     {'name': 'SLR', 'id': 'SLR'},
#                     {'name': 'SARIMA', 'id': 'SARIMA'}
#                 ]
#             ],
#             style_table={
#                 'marginTop': '50px',
#                 'width': '100%',
#                 'marginLeft': '20px'
#             },
#             fill_width=False
#         ),
#         html.Br(),
#         html.Div([
#             html.Span("Beste Methode:", style={'font-size': '25px'}),
#             html.P(""),
#             html.Span(id="BestMethod", style={'font-size': '25px', 'color': 'green'})
#         ], style={'display': 'flex', 'align-items': 'center', 'margin-top': '15px'})
#     ], style={'display': 'inline-block', 'width': '25%', 'verticalAlign': 'top'}),

#     dash_table.DataTable(
#             id='table_BestPrognose',
#             columns=[
#                 {'name': 'Monat', 'id': 'Monat'},
#                 {'name': 'Prognosewert', 'id': 'Prognosewert'}
#             ],
#             data=[],
#             style_cell={
#                 'textAlign': 'center',
#                 'color': '#FFFFFF',
#                 'backgroundColor': "#121212",
#                 'font_size': '15px',
#                 'font-family': 'Constantia'
#             },
#             style_header={
#                 'backgroundColor': '#4169E1',
#                 'padding': '10px',
#                 'color': '#FFFFFF',
#                 'font-family': 'Constantia'
#             },
#             style_data_conditional=[
#                 {
#                     'if': {'column_id': c['id']},
#                     'minWidth': '20px',
#                     'maxWidth': '30px',
#                     'width': '30px'
#                 } for c in [
#                     {'name': 'Metrik', 'id': 'Metrik'},
#                     {'name': 'LSTM', 'id': 'LSTM'},
#                     {'name': 'SLR', 'id': 'SLR'},
#                     {'name': 'SARIMA', 'id': 'SARIMA'}
#                 ]
#             ],
#             style_table={
#                 'marginTop': '50px',
#                 'width': '100%',
#                 'marginLeft': '20px'
#             },
#             fill_width=False
#         ),
    
#     html.Div([
#         dcc.Graph(id="All-Method-Graph", style={'width': '90%', 'height': '80%', 'marginTop': '50px'}),
#     ], style={'display': 'inline-block', 'width': '70%', 'verticalAlign': 'top'}),
    
#     html.Div(id='error-message', style={'color': 'red', 'textAlign': 'center', 'font-family': 'Constantia'}),
    
#     html.Div([
#         dcc.Graph(id="Method-Graph", style={'width': '33%', 'height': '60%', 'display': 'inline-block', 'margin-top': '20px'}),
#         dcc.Graph(id="LineareRegression", style={'width': '33%', 'height': '60%', 'display': 'inline-block', 'margin-top': '20px'}),
#         dcc.Graph(id='price-forecast-graph', style={'width': '33%', 'height': '60%', 'display': 'inline-block', 'margin-top': '20px'}),
#     ], style={'textAlign': 'center'}),
# ], style={'background-color': "#121212", 'width': '100%', 'height': '100%', 'font-family': 'Constantia'})




# @callback(Output('Method-Graph', 'figure'), [Input('Port3', 'value'), Input('Port4', 'value')])
# def update_lstm_graph(flight_Abflug, flight_Ankunft):
#     fig, normalized_mae_lstm, normalized_mse_lstm, normalized_rmse_lstm,_ = get_lstm_predictions(flight_Abflug, flight_Ankunft)
#     print("LSTM aus update")
#     print(normalized_mae_lstm)
#     print(normalized_mse_lstm)
#     print(normalized_rmse_lstm)

#     return fig



# @callback(Output('LineareRegression', 'figure'), [Input('Port3', 'value'), Input('Port4', 'value')])
# def update_LineareRegression(flight_Abflug, flight_Ankunft):
#     fig, _, _, _,_ = LineareRegression(flight_Abflug, flight_Ankunft)

#     return fig





# @callback(Output('All-Method-Graph', 'figure'), [Input('Port3', 'value'), Input('Port4', 'value')])


# def update_all_method_graph(flight_Abflug, flight_Ankunft):
#     try:
#         # Bereite den Gesamtvergleichsgraph vor
#         fig = go.Figure()

#         # # LSTM Prognose
#         lstm_fig = update_lstm_graph(flight_Abflug, flight_Ankunft)
#         for trace in lstm_fig['data']:
#             fig.add_trace(trace)
#             fig.add_annotation(
#                 xref="paper", yref="paper",
#                 x=0.99, y=0.05,
#                 text="LSTM",  # Hier den gewünschten Text einfügen
#                 showarrow=False,
#                 font=dict(size=16, color="yellow"),  # Hier können Sie Schriftgröße und Farbe anpassen
#                 bgcolor="rgba(0, 0, 0, 0)"  # Transparenter Hintergrund
#             )

#         # Lineare Regression Prognose
#         lr_fig = update_LineareRegression(flight_Abflug, flight_Ankunft)
#         for trace in lr_fig['data']:
#             fig.add_trace(trace)
#             fig.add_annotation(
#                 xref="paper", yref="paper",
#                 x=0.99, y=0.10,
#                 text="Lineare Regression",  # Hier den gewünschten Text einfügen
#                 showarrow=False,
#                 font=dict(size=16, color="red"),  # Hier können Sie Schriftgröße und Farbe anpassen
#                 bgcolor="rgba(0, 0, 0, 0)"  # Transparenter Hintergrund
#             )

#         fig.update_layout(xaxis_title='Datum', yaxis_title='Preis ($)', template='plotly_dark', height=600, showlegend=False)
#         fig.update_layout( title=f"Vergleich der Prognosemethoden für: {flight_Abflug} & {flight_Ankunft}")
#         fig.update_xaxes(title="Jahr", range=['2024-01-01', '2025-05-01'])

#         return fig

#     except Exception as e:
#         fig = go.Figure()
#         fig.update_layout(title='Fehler bei der Prognose', xaxis_title='Datum', yaxis_title='Preis ($)', template='plotly_dark')
#         return fig
    



# @callback(
#     Output(component_id='table_Metriken', component_property='data'),
#     Output('BestMethod', 'children'),
#     Output(component_id='table_BestPrognose', component_property='data'),
#     [
#         Input(component_id='flight_Abflug', component_property="data"),
#         Input(component_id='flight_Ankunft', component_property="data")
#     ]
# )
# def update_table_Metriken(flight_Abflug, flight_Ankunft):

#     try:
#         # LSTM predictions and metrics
#         _, normalized_mae_lstm, normalized_mse_lstm, normalized_rmse_lstm, lstm_predictions = get_lstm_predictions(flight_Abflug, flight_Ankunft)

#         # Linear Regression predictions and metrics
#         _, normalized_mae_lr, normalized_mse_lr, normalized_rmse_lr, lr_predictions = LineareRegression(flight_Abflug, flight_Ankunft)

#         # Determine best method based on RMSE
#         metrics_data = [
#             {'Metrik': 'MAE', 'LSTM': round(normalized_mae_lstm, 2), 'SLR': round(normalized_mae_lr, 2), 'SARIMA': ''},
#             {'Metrik': 'MSE', 'LSTM': round(normalized_mse_lstm, 2), 'SLR': round(normalized_mse_lr, 2), 'SARIMA': ''},
#             {'Metrik': 'RMSE', 'LSTM': round(normalized_rmse_lstm, 2), 'SLR': round(normalized_rmse_lr, 2), 'SARIMA': ''}
#         ]

#         best_method = 'LSTM' #if normalized_rmse_lstm < normalized_rmse_lr else 'SLR'

#         # Prepare forecast table data for table_BestPrognose
#         table_BestPrognose_data = []
#         forecast_months = pd.date_range(start='2024-04-01', periods=5, freq='MS').strftime('%B %Y')

#         print("lstm prediction")
#         print(lstm_predictions)

#         if best_method == 'LSTM':
#             predictions = lstm_predictions[:5]
#         else:
#             predictions = lr_predictions[:5]

#         for month, forecast in zip(forecast_months, predictions):
#             table_BestPrognose_data.append({'Monat': month, 'Prognosewert': round(forecast, 2)})

#         # Debugging: Prüfen der Rückgabewerte
#         # print("Metrics data:")
#         # print(metrics_data)
#         # print("Best Method:", best_method)
#         # print("Table Best Prognose data:")
#         # print(table_BestPrognose_data)

#         return metrics_data, best_method, table_BestPrognose_data

#     except Exception as e:
#         print("Fehler:", e)
#         return [], "", []
