import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime as dt
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash
import tensorflow as tf



def LineareRegression (flight_Abflug, flight_Ankunft):

    # Setzen vom Seed
    seed = 45
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Laden und Filtern der Daten
    df = pd.read_csv("AUS_Fares_March2024.csv", sep=',')
    df = df[["Year", "Month", "YearMonth", "Port1", "Port2", "Route","$Real"]]
    df = df[(df["Port1"] == flight_Abflug) & (df["Port2"] == flight_Ankunft)]
    df = df.reset_index(drop=True)