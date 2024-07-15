
import pandas as pd
from dash import html, callback
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dash_table


dash.register_page(__name__, path='/', name="Fluganalyse1")

# Definiere das externe CSS-Stylesheet für das Bootstrap-Styling
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"


# Definiere Style- und Farbdefinitionen für die Tabs
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

# Definiere das Farbschema für das gesamte Dashboard
colors = {
    'background': '#000000',
    'text': '#FFFFFF',
    "Button": "#4169E1"
}

# Definiere die Spaltenkonfiguration für die Dash DataTable
table_columns = [
    {'name': 'Jahr', 'id': 'Year'},
    {'name': 'Monat', 'id': 'Month'},
    {'name': 'Preis', 'id': '$Real'}
]

# Definiere die Layout-Struktur unter Verwendung von Dash HTML-Komponenten und Core Components
layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                # Dropdowns für den Zeitraumfilter
                html.P("Wählen Sie einen Zeitraum: ", style={"font-size": "25px", "margin-right": "2150px", "margin-top": "100px", 'font-family': 'Constantia'}),
                html.Div([
                    html.Div([
                        html.P("Von:", style={"font-size": "25px", "margin-right": "2400px", "margin-top": "50px", 'font-family': 'Constantia'}),
                        html.Br(),
                        # Dropdown für Monat von Startdatum
                        html.Div([
                            dcc.Dropdown(
                                options=[{"label": str(i), "value": str(i)} for i in range(1, 13)],
                                id='start-month',
                                value="1",
                                clearable=False,
                                className='custom-dropdownNew',
                                style={'width': '25%', "color": "#000000", 'font-family': 'Constantia', "font-size": "20px", "margin-top": "-20px", "margin-left": "95px"}
                            ),
                            # Dropdown für Jahr von Startdatum
                            dcc.Dropdown(
                                options=[{"label": str(i), "value": str(i)} for i in range(2010, 2025)],
                                id='start-year',
                                value="2010",
                                clearable=False,
                                className='custom-dropdown',
                                style={'width': '28%', "color": "black", 'font-family': 'Constantia', "font-size": "20px", "margin-top": "-38px", "margin-left": "215px"}
                            ),
                        ], style={"margin-left": "-200px"}),
                    ]),
                    html.Br(),
                    html.Div([
                        html.P("Bis:", style={"font-size": "25px", "margin-right": "950px", "margin-top": "15px", 'font-family': 'Constantia'}),
                        html.Br(),
                        # Dropdown für Monat von Enddatum
                        html.Div([
                            dcc.Dropdown(
                                options=[{"label": str(i), "value": str(i)} for i in range(1, 13)],
                                id='end-month',
                                value="12",
                                clearable=False,
                                className='custom-dropdownNew',
                                style={'width': '25%', "color": "black", 'font-family': 'Constantia', "font-size": "20px", "margin-top": "-20px", "margin-left": "93px", "margin-top": "-10px"}
                            ),
                            # Dropdown für Jahr von Enddatum
                            dcc.Dropdown(
                                options=[{"label": str(i), "value": str(i)} for i in range(2010, 2025)],
                                id='end-year',
                                value="2024",
                                clearable=False,
                                className='custom-dropdown',
                                style={'width': '28%', "color": "black", 'font-family': 'Constantia', "font-size": "20px", "margin-left": "215px", "margin-top": "-37px"}
                            ),
                        ], style={"margin-left": "-200px"})
                    ], 
                    ),
                ])
            ], style={'display': 'inline-block', "text-align": "justify"}),
        ]),
    
        html.Div([
            # Checkboxen für Auswahl der Preisarten
            html.P("Auswahl: ", style={"font-size": "30px", "margin-top": "180px", 'font-family': 'Constantia'}),
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
        ], style={"margin-right": "2200px", 'display': 'inline-block', "text-align": "justify"})
    ]),
    html.P(""),
    html.P(""),
    html.Div([
        html.Div([
            # Graph für Zeitreihen-Darstellung
            dcc.Graph(
                id="time-series-chart",
                style={'width': '50%', "height": '80%', "margin-left": "auto", "margin-right": "auto", "color": "#696969", "margin-top": "-790px"}
            ),
            # Tabs für verschiedene Diagrammarten
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
            # Graph für Liniendiagramm
            dcc.Graph(
                id="Liniendiagramm",
                style={'width': '50%', "height": '80%', "margin-left": "auto", "margin-right": "auto", "color": "#696969", "margin-top": "40px"}
            ),
            # Graph für Boxplot
            dcc.Graph(
                id="Boxplot",
                style={'width': '50%', "height": '80%', "margin-left": "auto", "margin-right": "auto", "color": "#696969", "margin-top": "-35px"}
            )
        ], style={"vertical-align": "top", "margin-left": "-100", "margin-top": "-800"}),
        
        # DataTable zur Anzeige von Daten in tabellarischer Form
        dash_table.DataTable(
            # ID des DataTable-Komponenten für die Referenz in Callbacks
            id='table', 
             # Leere Daten für den Anfang, werden durch Callbacks aktualisiert
            data=[], 
            # Spaltenkonfiguration für die DataTable
            columns=table_columns, 
            # Stil der Zellen in der DataTable
            style_cell={
                'textAlign': 'center', 
                'color': '#FFFFFF', 
                'background': colors["background"], 
                'font_size': '15px',
                'font-family': 'Constantia'
            },
            # Stil der Tabellenüberschriften
            style_header={
                'backgroundColor': '#4169E1',
                'padding': '10px',
                'color': '#FFFFFF',
                'font-family': 'Constantia'
            },
            # Bedingte Stilregeln für Datenzellen
            style_data_conditional=[
                {
                    'if': {'column_id': c},
                    'minWidth': '20px',
                    'maxWidth': '50px',
                    'width': '30px'
                } for c in ['Year', 'Month', '$Real']
            ],
            # Stil der gesamten Tabelle
            style_table={
                'maxHeight': '1340px',
                'overflowY': 'scroll',
                "margin-top": "-1350px",
                "margin-left": "2200px"
            },
            # Deaktiviert automatisches Füllen der Tabellenbreite
            fill_width=False
        )
    ])
])

layout = layout

# Weitere Style-Eigenschaften definieren für Tabs
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
    # Daten aus CSV-Datei laden
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    # Daten filtern: Nur relevante Spalten und bestimmte Route
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df.reset_index(drop=True)

    # Daten nach Zeitraum filtern
    filtered_df = df[(df["YearMonth"] >= int(start_year + start_month.zfill(2))) & (df["YearMonth"] <= int(end_year + end_month.zfill(2)))]

    # Maximale Werte pro Jahr berechnen
    max_values = filtered_df.groupby("Year")["$Real"].max().reset_index()

    # Plot mit Plotly Express (Bar Chart) erstellen
    fig = px.bar(max_values, x="Year", y="$Real", template="plotly_dark",
             labels={"Year": "Jahr", "$Real": "Preis"}, color_discrete_sequence=["#ff0000"])
    
    # Anpassung der X-Achse um jedes Jahr anzuzeigen
    fig.update_xaxes(tickvals=max_values["Year"], ticktext=max_values["Year"].astype(str))

    # Anpassung der Y-Achse Tick Labels
    fig.update_yaxes(tickformat=",.0f")

    
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


def ZweiteStrecke(flight_Abflug, flight_Ankunft, start_month, start_year, end_month, end_year, selected_Price, tabs):
    # Daten aus CSV-Datei laden
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',', dtype={"Year": int})
    # Daten filtern: Nur relevante Spalten und bestimmte Route
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route", "$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]

    # Daten nach Zeitraum filtern
    filtered_df = df[(df["YearMonth"] >= int(start_year + start_month.zfill(2))) & (df["YearMonth"] <= int(end_year + end_month.zfill(2)))]

    # Figure-Objekt für das Diagramm erstellen
    fig = go.Figure()

    if tabs == "Liniendiagramm":  # Wenn der aktive Tab "Liniendiagramm" ist

        if 'MAX' in selected_Price:
            # Plot für den Maximalwert erstellen
            max_values = filtered_df.groupby("Year")["$Real"].max()
            max_line = go.Scatter(x=max_values.index, y=max_values.values, mode='lines', name='Maximalwert')
            fig.add_trace(max_line)
    
        if 'MIN' in selected_Price:
            # Plot für den Minimalwert erstellen
            min_values = filtered_df.groupby("Year")["$Real"].min()
            min_line = go.Scatter(x=min_values.index, y=min_values.values, mode='lines', name='Minimalwert')
            fig.add_trace(min_line)

        if 'DURCH' in selected_Price:
            # Plot für den Durchschnittswert erstellen
            mean_values = filtered_df.groupby("Year")["$Real"].mean()
            mean_line = go.Scatter(x=mean_values.index, y=mean_values.values, mode='lines', name='Durchschnittswert')
            fig.add_trace(mean_line)
            
    if tabs == "Scatter-Plot":  # Wenn der aktive Tab "Scatter-Plot" ist

        if 'MAX' in selected_Price:
            # Scatter-Plot für den Maximalwert erstellen
            max_values = filtered_df.groupby("Year")["$Real"].max()
            fig.add_trace(go.Scatter(x=max_values.index, y=max_values.values, mode='markers', name='Maximalwert', marker=dict(
                    size=10,
                    line=dict(
                        width=2,
                        color='DarkSlateGrey'
                    ))))

        if 'MIN' in selected_Price:
            # Scatter-Plot für den Minimalwert erstellen
            min_values = filtered_df.groupby("Year")["$Real"].min()
            fig.add_trace(go.Scatter(x=min_values.index, y=min_values.values, mode='markers', name='Minimalwert', marker=dict(
                    size=10,
                    line=dict(
                        width=2,
                        color='DarkSlateGrey'
                    ))))

        if 'DURCH' in selected_Price:
            # Scatter-Plot für den Durchschnittswert erstellen
            mean_values = filtered_df.groupby("Year")["$Real"].mean()
            fig.add_trace(go.Scatter(x=mean_values.index, y=mean_values.values, mode='markers', name='Durchschnittswerte', marker=dict(
                    size=10,
                    line=dict(
                        width=2,
                        color='DarkSlateGrey'
                    ))))

    if tabs == "Area Chart":  # Wenn der aktive Tab "Area Chart" ist
        # Daten für Minima, Maxima und Durchschnittswerte vorbereiten
        min_values = filtered_df.groupby("Year")["$Real"].min()
        max_values = filtered_df.groupby("Year")["$Real"].max()
        mean_values = filtered_df.groupby("Year")["$Real"].mean()

        if 'MAX' in selected_Price:
            # Area Chart für den Maximalwert erstellen
            fig.add_trace(go.Scatter(
                x=max_values.index,
                y=max_values.values,
                mode='lines',
                fill='tozeroy',
                line=dict(color='green'),
                name='Maximalwert'
            ))

        if 'MIN' in selected_Price:
            # Area Chart für den Minimalwert erstellen
            fig.add_trace(go.Scatter(
                x=min_values.index,
                y=min_values.values,
                fill='tonexty', 
                mode='lines',
                line=dict(color='blue'),
                fillcolor='rgba(0, 0, 255, 0.3)', 
                name='Minimalwert'
            ))

        if 'DURCH' in selected_Price:
            # Area Chart für den Durchschnittswert erstellen
            fig.add_trace(go.Scatter(
                x=mean_values.index,
                y=mean_values.values,
                fill='tonexty', 
                mode='lines',
                line=dict(color='orange'),
                fillcolor='rgba(255, 165, 0, 0.3)', 
                name='Durchschnittswert'
            ))
    # Anpassung des Layouts
    fig.update_layout(
        title="Maximum und Minimumpreise für die Strecke: {} & {}".format(flight_Abflug, flight_Ankunft),
        yaxis=dict(title="Preis"),
        template="plotly_dark"
    )

    # Wenn 'max_values' als lokale Variable definiert ist, dann die X-Achse anpassen
    if 'max_values' in locals():
        fig.update_xaxes(
            title="Jahr",  # Titel für die X-Achse festlegen
            tickvals=max_values.index,  # Wert für die Ticks auf der X-Achse (Jahre)
            ticktext=max_values.index.astype(str)  # Text für die Ticks auf der X-Achse (als Zeichenketten)
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

def update_boxplot(flight_Abflug, flight_Ankunft, start_month, start_year, end_month, end_year):
    # Daten aus CSV-Datei laden
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',', dtype={"Year": int})
    # Daten filtern: Nur relevante Spalten und bestimmte Route
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df[["Year", "$Real", "YearMonth"]]

    # Start- und Endzeitraum definieren
    start_year_month = int(start_year + start_month.zfill(2))
    end_year_month = int(end_year + end_month.zfill(2))

    # Daten nach Zeitraum filtern
    filtered_df = df[(df["YearMonth"] >= start_year_month) & (df["YearMonth"] <= end_year_month)]

    # Figure-Objekt für den Boxplot erstellen
    fig = go.Figure()

    # Für jedes Jahr eine Boxplot-Spur hinzufügen
    years = filtered_df["Year"].unique()
    for year in years:
        year_data = filtered_df[filtered_df["Year"] == year]
        fig.add_trace(go.Box(y=year_data['$Real'], name=str(year)))

    # Layout des Boxplots aktualisieren (Titel, Achsenbeschriftungen, Vorlage)
    fig.update_layout(
        title="Preisverteilung nach Jahr",
        xaxis_title="Jahr",
        yaxis_title="Preis",
        template="plotly_dark"
    )

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

def update_table(flight_Abflug, flight_Ankunft, start_month, start_year, end_month, end_year):
    # Daten aus CSV-Datei laden
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',', dtype={"Year": int})
    # Daten filtern: Nur relevante Spalten und bestimmte Route
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df[["Year", "Month", "$Real", "YearMonth"]]

    # Daten nach Zeitraum filtern
    filtered_df = df[(df["YearMonth"] >= int(start_year + start_month.zfill(2))) & (df["YearMonth"] <= int(end_year + end_month.zfill(2)))]

    # Monatsnamen für die Darstellung anpassen
    monatsnamen = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"]
    filtered_df["Month"] = filtered_df["Month"].apply(lambda x: monatsnamen[x-1])
    
    # Daten als Records-Datenstruktur (Liste von Dictionaries) für die Tabelle zurückgeben
    return filtered_df.to_dict('records')