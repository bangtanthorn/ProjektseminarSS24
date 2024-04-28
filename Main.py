#Imports definieren
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


# Erstelle eine Dash-App und aktiviere die Verwendung von Seiten (use_pages=True)
app = dash.Dash(__name__, use_pages=True, external_stylesheets = [dbc.themes.DARKLY])


#Definieren von Farben
colors = {
    'background': '#000000',
    'text': '#FFFFFF',
    "Button" : "#4169E1"
}


topbar = dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Fluganalyse1", href = "/", style={"font-size": "20px"})),
                dbc.NavItem(dbc.NavLink("Fluganalyse2", href = "/ZweiteSeite", style={"font-size": "20px"})),
                dbc.Col(html.P(""), width = 1), #Leeres Element, damit nächstes Element weiter am Rand steht                                 
            ], vertical = False ,pills = True, className = "navbar navbar-expand-lg navbar-dark bg-secondary", style={"color": "#A9A9A9"} #className legt den Style der einzelnen Komponenten fest, bspw. Bar (hier) oder Cards (kommen noch)
)


app.layout = html.Div(
    [

        html.Div(
        #Überschrift "Flugpreise" in der App
        html.H1("Dashboard: Flugpreise", style={'fontSize':70, 'textAlign':'center', 'color': colors['text'], 'font-family': 'Constantia', 'fontWeight': 'normal'})),
        html.Hr(),
        dbc.Row(        
            dbc.Col(
                [
                    topbar #Fügt oben definierte Bar ein
                ])),
        dash.page_container #Fügt definierte Seitennamen und Referenzen ein
            ,
        #Navigationsleiste mit Links zu verschiedenen Seiten
        #html.Div([
            #dcc.Link("  |  " + page['name']+"  |  ", href=page['path'], style={'fontSize':20, 'textAlign':'center', 'color': colors['text']})
            #for page in dash.page_registry.values()
        #]),
        html.Hr(),

        # Platzhalter für den Inhalt jeder Seite, der dynamisch geändert wird,
        # wenn der Benutzer zwischen den Seiten navigiert
        #dash.page_container
    ], 
    #Style für das komplette Dash definieren
    style={'background-color': "#121212",
          'background-size': '100%',
          'position': 'fixed',
          'width': '100%',
          'height': '100%',
          'font-family': 'Rockwell',
          "display" : "block" , 
          "margin-left": "auto",
          "margin-right": "auto",
          'textAlign': 'center',
          "overflow":"Scroll"

          }
)




#Starte die Dash-App, falls dieses Skript direkt ausgeführt wird
if __name__ == "__main__":

     app.run(debug=True)
