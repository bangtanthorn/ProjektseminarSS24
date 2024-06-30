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




df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Real"]]
df = df[(df["Port1"] == "Adelaide") & (df["Port2"] == "Brisbane")]
df = df.reset_index(drop=True)
print(df.tail(20))



dash.register_page(__name__, path='/', name = "Fluganalyse1")





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
    {'name': 'Jahr', 'id': 'Year'},
    {'name': 'Monat', 'id': 'Month'},
    {'name': 'Preis', 'id': '$Real'}
]

# Layout definieren
layout = html.Div([
    html.Div([
        html.Div([
            html.Div([  #div über alle
            html.P("Wählen Sie einen Zeitraum: ", style={"font-size": "25px", "margin-right": "1800px", "margin-top": "100px", 'font-family': 'Constantia'}),
            html.Br(),
            html.Div([
                html.Div([
                    html.P("Von:", style={"font-size": "25px", "margin-right": "700px", "margin-top": "100px", 'font-family': 'Constantia'}),
                    html.Br(),
                    dcc.Dropdown(
                        options=[{"label": str(i), "value": str(i)} for i in range(1, 13)],
                        id='start-month',
                        value="1",
                        clearable=False,
                        style={'width': '20%', "color": "black", 'font-family': 'Constantia', "font-size": "20px"}
                    ),
                    dcc.Dropdown(
                        options=[{"label": str(i), "value": str(i)} for i in range(2010, 2025)],
                        id='start-year',
                        value="2010",
                        clearable=False,
                        style={'width': '30%', "color": "black", 'font-family': 'Constantia', "font-size": "20px"}
                    ),
                ], style={'width': '49%'}),
                html.Br(),
                html.Div([
                    html.P("Bis:", style={"font-size": "25px", "margin-right": "700px", "margin-top": "100px", 'font-family': 'Constantia'}),
                    html.Br(),
                    dcc.Dropdown(
                        options=[{"label": str(i), "value": str(i)} for i in range(1, 13)],
                        id='end-month',
                        value="12",
                         clearable=False,
                        style={'width': '20%', "color": "black", 'font-family': 'Constantia', "font-size": "20px", "margin-right": "1000px"}
                    ),
                    dcc.Dropdown(
                        options=[{"label": str(i), "value": str(i)} for i in range(2010, 2025)],
                        id='end-year',
                        value="2024",
                         clearable=False,
                        style={'width': '30%', "color": "black", 'font-family': 'Constantia', "font-size": "20px", "margin-right": "1000px"}
                    ),
                ], style={'width': '49%'})
            ])
            ],  style={},)
        ]),
    
        html.Div([
            html.P("Auswahl: ", style={"font-size": "30px", "margin-top": "50px", 'font-family': 'Constantia'}),
            html.Br(),
            dcc.Checklist(
                id='price-type-checklist',
                options=[
                    {'label': 'Maximalpreise', 'value': 'MAX'},
                    {'label': 'Minimalpreise', 'value': 'MIN'},
                    {'label': 'Durchschnittspreise', 'value': 'DURCH'}
                ],
                value=['MAX', "MIN", "DURCH"],
                style={"font-size": "25px", "margin-top": "30px", 'font-family': 'Constantia'}
            )
        ], style={"margin-right": "2200px"})
    ]),
    html.P(""),
    html.P(""),
    html.Div([
        html.Div([
            dcc.Graph(
                id="time-series-chart",
                style={'width': '50%', "height": '80%', "margin-left": "auto", "margin-right": "auto", "color": "#696969", "margin-top": "-830px"}
            ),
            dcc.Tabs(
                id='tabs',
                value="Liniendiagramm",
                children=[
                    dcc.Tab(label='Liniendiagramm', value='Liniendiagramm', children=[], style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='Streudiagramm', value='Scatter-Plot', children=[], style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='Flächendiagramm', value='Area Chart', children=[], style=tab_style, selected_style=tab_selected_style)
                ],
                style={'width': '50%', 'font-size': '80%', 'height': "90%", "margin-left": "auto", "margin-right": "auto", "display": "block", "margin-top": "20px", 'font-size': '20px', 'font-family': 'Constantia'}
            ),
            dcc.Graph(
                id="Liniendiagramm",
                style={'width': '50%', "height": '80%', "margin-left": "auto", "margin-right": "auto", "color": "#696969", "margin-top": "40px"}
            ),
            dcc.Graph(
                id="Boxplot",
                style={'width': '50%', "height": '80%', "margin-left": "auto", "margin-right": "auto", "color": "#696969", "margin-top": "-35px"}
            )
        ], style={"vertical-align": "top", "margin-left": "-100", "margin-top": "-800"}),
        
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
                    'minWidth': '20px',
                    'maxWidth': '50px',
                    'width': '30px'
                } for c in ['Year', 'Month', '$Real']
            ],
            style_table={
                'maxHeight': '1340px',
                'overflowY': 'scroll',
                "margin-top": "-1350px",
                "margin-left": "2000px"
            },
            fill_width=False
        )
    ])
])


layout = layout

# Weitere Style-Eigenschaften definieren
tab_style = {
    'padding': '10px',
    'fontWeight': 'bold',
    'font-family': 'Constantia'
}

tab_selected_style = {
    'padding': '10px',
    'fontWeight': 'bold',
    'font-family': 'Constantia',
    'color': '#ffffff',
    'backgroundColor': '#4169E1'
}

@callback(
    Output(component_id="time-series-chart", component_property="figure"),
    Input(component_id="flight_Abflug", component_property="data"),
    Input(component_id="flight_Ankunft", component_property="data"),
    Input('start-month', component_property="value"),
    Input('start-year', component_property="value"),
    Input('end-month', component_property="value"),
    Input('end-year', component_property="value"),
)
def update_time_series_chart(flight_Abflug, flight_Ankunft, start_month, start_year, end_month, end_year):
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df.reset_index(drop=True)

    start_date = datetime(int(start_year), int(start_month), 1)
    end_date = datetime(int(end_year), int(end_month), 1)

    filtered_df = df[(df["YearMonth"] >= int(start_year + start_month.zfill(2))) & (df["YearMonth"] <= int(end_year + end_month.zfill(2)))]

    max_values = filtered_df.groupby("Year")["$Real"].max().reset_index()

    fig = px.bar(max_values, x="Year", y="$Real", template="plotly_dark",
        labels={"Year": "Jahr", "$Real": "Preis"}, color_discrete_sequence=["#ff0000"])
    
    return fig


@callback(
    Output(component_id= "Liniendiagramm", component_property="figure"),
    Input(component_id= "flight_Abflug", component_property="data"),
    Input(component_id= "flight_Ankunft", component_property="data"),
    Input('start-month', component_property="value"),
    Input('start-year', component_property="value"),
    Input('end-month', component_property="value"),
    Input('end-year', component_property="value"),
    Input('price-type-checklist', 'value'),
    Input('tabs', 'value'),
)


def ZweiteStrecke(flight_Abflug, flight_Ankunft, start_month, start_year, end_month, end_year, selected_Price,tabs):

    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',', dtype={"Year": int})
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]

    # Extrahiere das Jahr aus den Datumsangaben und wandele sie in Integer um
    #start_year = datetime.strptime(start_date, "%Y-%m-%d").year
    #end_year = datetime.strptime(end_date, "%Y-%m-%d").year

    start_date = datetime(int(start_year), int(start_month), 1)
    end_date = datetime(int(end_year), int(end_month), 1)

    filtered_df = df[(df["YearMonth"] >= int(start_year + start_month.zfill(2))) & (df["YearMonth"] <= int(end_year + end_month.zfill(2)))]
    
    #filtered_df = df.loc[(df["Year"] >= start_year) & (df["Year"] <= end_year)]


    fig = go.Figure()

    if tabs == "Liniendiagramm": 

        if 'MAX' in selected_Price:
            # Plot für das Maximum erstellen
            max_values = filtered_df.groupby("Year")["$Real"].max()
            max_line = go.Scatter(x=max_values.index, y=max_values.values, mode='lines', name='Maximalwert')
            fig.add_trace(max_line)
    
        if 'MIN' in selected_Price:
            # Plot für das Minimum erstellen
            min_values = filtered_df.groupby("Year")["$Real"].min()
            min_line = go.Scatter(x=min_values.index, y=min_values.values, mode='lines', name='Minimalwert')
            fig.add_trace(min_line)

        if 'DURCH' in selected_Price:
            # Plot für das Minimum erstellen
            mean_values = filtered_df.groupby("Year")["$Real"].mean()
            mean_line = go.Scatter(x=mean_values.index, y=mean_values.values, mode='lines', name='Durchschnittswert')
            fig.add_trace(mean_line)
            # Figur erstellen und Linien hinzufügen
        
    if tabs == "Scatter-Plot": 

        if 'MAX' in selected_Price:
            max_values = filtered_df.groupby("Year")["$Real"].max()
            #fig = px.scatter(x=max_values.index, y=max_values.values)
            fig.add_trace(go.Scatter(x=max_values.index, y=max_values.values, mode='markers', name='Maximalwert', marker=dict(
                    size=10,
                    #color='rgba(255, 0, 0, 0.6)',
                    line=dict(
                    width=2,
                    color='DarkSlateGrey'
                      
                    ))))

        if 'MIN' in selected_Price:
            min_values = filtered_df.groupby("Year")["$Real"].min()
            #fig = px.scatter(x=max_values.index, y=max_values.values)
            fig.add_trace(go.Scatter(x=min_values.index, y=min_values.values, mode='markers', name='Minimalwert', marker=dict(
                    size=10,
                    #color='rgba(255, 0, 0, 0.6)',
                    line=dict(
                    width=2,
                    color='DarkSlateGrey'
                    ))))


        if 'DURCH' in selected_Price:
            mean_values = filtered_df.groupby("Year")["$Real"].mean()
            #fig = px.scatter(x=max_values.index, y=max_values.values)
            fig.add_trace(go.Scatter(x=mean_values.index, y=mean_values.values, mode='markers', name='Durchschnittswerte', marker=dict(
                    size=10,
                    #color='rgba(255, 0, 0, 0.6)',
                    line=dict(
                    width=2,
                    color='DarkSlateGrey'
                    ))))
            

    if tabs == "Area Chart":
        min_values = filtered_df.groupby("Year")["$Real"].min()
        max_values = filtered_df.groupby("Year")["$Real"].max()
        mean_values = filtered_df.groupby("Year")["$Real"].mean()

        if 'MAX' in selected_Price:
            fig.add_trace(go.Scatter(
            x=max_values.index,
            y=max_values.values,
            mode='lines',
            fill='tozeroy',
            line=dict(color='green'),
            name='Maximalwert'
        ))

        if 'MIN' in selected_Price:
            fig.add_trace(go.Scatter(
            x=min_values.index,
            y=min_values.values,
            fill='tonexty', # fill area between this trace and the previous one
            mode='lines',
            line=dict(color='blue'),
            fillcolor='rgba(0, 0, 255, 0.3)', # blue with transparency
            name='Minimalwert'
        ))

        if 'DURCH' in selected_Price:
            fig.add_trace(go.Scatter(
                x=mean_values.index,
                y=mean_values.values,
                fill='tonexty', # fill area between this trace and the previous one
                mode='lines',
                line=dict(color='orange'),
                fillcolor='rgba(255, 165, 0, 0.3)', # orange with transparency
                name='Durchschnittswert'
            ))

    fig.update_layout(
    title="Maximum und Minimumpreise für die Strecke: {} & {}".format(flight_Abflug, flight_Ankunft),
    xaxis=dict(title="Jahr"),
    yaxis=dict(title="Preis"),
    template="plotly_dark"
    )

    return fig


@callback(
    Output(component_id="Boxplot", component_property="figure"),
    Input(component_id='flight_Abflug', component_property='data'),
    Input(component_id='flight_Ankunft', component_property='data'),
    Input('start-month', component_property="value"),
    Input('start-year', component_property="value"),
    Input('end-month', component_property="value"),
    Input('end-year', component_property="value"),
)
def update_boxplot(flight_Abflug, flight_Ankunft,  start_month, start_year, end_month, end_year, ):

    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',', dtype={"Year": int})
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df[["Year", "$Real", "YearMonth"]]
    
    #start_year = datetime.strptime(start_date, "%Y-%m-%d").year
    #end_year = datetime.strptime(end_date, "%Y-%m-%d").year
    
    #filtered_df = df.loc[(df["Year"] >= start_year) & (df["Year"] <= end_year)]

    start_date = datetime(int(start_year), int(start_month), 1)
    end_date = datetime(int(end_year), int(end_month), 1)

    filtered_df = df[(df["YearMonth"] >= int(start_year + start_month.zfill(2))) & (df["YearMonth"] <= int(end_year + end_month.zfill(2)))]

    fig = go.Figure()
    fig.add_trace(go.Box(y=filtered_df['$Real'], name='Boxplot'))

    fig.update_layout(title="Preisverteilung",
                      xaxis_title="Jahr",
                      yaxis_title="Preis",
                      template="plotly_dark")
    return fig


@callback(
    Output(component_id='table', component_property='data'),
    Input(component_id='flight_Abflug', component_property='data'),
    Input(component_id='flight_Ankunft', component_property='data'),
    Input('start-month', component_property="value"),
    Input('start-year', component_property="value"),
    Input('end-month', component_property="value"),
    Input('end-year', component_property="value"),
)
def update_table(flight_Abflug, flight_Ankunft, start_month, start_year, end_month, end_year,):
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',', dtype={"Year": int})
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df[["Year", "Month", "$Real", "YearMonth"]]
    
    # start_year = datetime.strptime(start_date, "%Y-%m-%d").year
    # end_year = datetime.strptime(end_date, "%Y-%m-%d").year
    
    # filtered_df = df.loc[(df["Year"] >= start_year) & (df["Year"] <= end_year)]
    start_date = datetime(int(start_year), int(start_month), 1)
    end_date = datetime(int(end_year), int(end_month), 1)

    filtered_df = df[(df["YearMonth"] >= int(start_year + start_month.zfill(2))) & (df["YearMonth"] <= int(end_year + end_month.zfill(2)))]

    monatsnamen = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"]
    filtered_df["Month"] = filtered_df["Month"].apply(lambda x: monatsnamen[x-1])
    
    return filtered_df.to_dict('records')