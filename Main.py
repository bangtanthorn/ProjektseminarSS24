#Imports definieren
import dash
from dash import html, dcc


# Erstelle eine Dash-App und aktiviere die Verwendung von Seiten (use_pages=True)
app = dash.Dash(__name__, use_pages=True)


#Definieren von Farben
colors = {
    'background': '#121212',
    'text': '#000000',
    "Button" : "#4169E1"
}


app.layout = html.Div(
    [

        html.Div(
        #Überschrift "Flugpreise" in der App
        html.H1("Flugpreise", style={'fontSize':70, 'textAlign':'center', 'color': colors['text'], 'font-family': 'Constantia', 'fontWeight': 'normal'})),
        #Navigationsleiste mit Links zu verschiedenen Seiten
        html.Div([
            dcc.Link("  |  " + page['name']+"  |  ", href=page['path'], style={'fontSize':20, 'textAlign':'center', 'color': colors['text']})
            for page in dash.page_registry.values()
        ]),
        html.Hr(),

        # Platzhalter für den Inhalt jeder Seite, der dynamisch geändert wird,
        # wenn der Benutzer zwischen den Seiten navigiert
        dash.page_container
    ],
    #Style für das komplette Dash definieren
    style={'background-color': "#D3D3D3",
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
