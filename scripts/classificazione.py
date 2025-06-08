#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

#%% IMPORT WEATHER DATASET
#Caricamento del dataset contenente i dati metereologici
weather_data = pd.read_csv("../data/raw/weather_prediction_dataset.csv")

#print(weather_data.head())

print(f"Il weather dataset ha {weather_data.shape[0]} record e {weather_data.shape[1]} colonne")

print("Etichette colonne weather_data:")
for etichetta_colonna in weather_data.columns:
    print(etichetta_colonna)
print()

#%% IMPORT BBQ LABEL DATASET
#Caricamento del dataset indicante le registrazioni con meteo da bbq e quelle non
classification_data = pd.read_csv("../data/raw/weather_prediction_bbq_labels.csv")

#print(classification_data.head())
print(f"Il bbq label dataset ha {classification_data.shape[0]} record e {classification_data.shape[1]} colonne")

print("Etichette colonne classification_data:")
for etichetta_colonna in classification_data.columns:
    print(etichetta_colonna)
print()

#%% INDIVIDUAZIONE SOTTODATABASE 
#lista_città = [s.replace("_BBQ_weather", "") for s in classification_data.columns[1:]]
conteggi = {}

for item in weather_data.columns[2:]:
    citta = item.split("_")[0]
    if citta in conteggi:
        conteggi[citta] += 1
    else:
        conteggi[citta] = 1

numero_parametri = max(conteggi.values())
print("Numero parametri meteorologici per città:")
for citta, conteggio in conteggi.items():
    print(f"{citta}: {conteggio}")    
print()
#Roma non è presente nel classification_data
print(f"Numero città weather dataset: {len(conteggi.keys())}")
print(f"Numero città bbq label dataset: {len(classification_data.columns) -1}")

#Selezione città e individuazione indici
citta = "OSLO"
chiavi = list(conteggi.keys())
valori = list(conteggi.values())
indici_citta = [0,0]
indici_citta[0] = sum(valori[:chiavi.index(citta)]) + 2
indici_citta[1] = indici_citta[0] + conteggi[citta]

#print(indici_citta)

#Costruzione dataset città
df = weather_data.iloc[:, indici_citta[0] : indici_citta[1]]
df["MONTH"] = weather_data.iloc[:, 1]
df["BBQ"] = classification_data[citta + "_BBQ_weather"][:]
df["DATE"] = classification_data["DATE"]


#%% PRE-PROCESSING
#Analisi valori nulli
print("Valori NaN")
print(df.isnull().sum())

#Rimozione valori nulli
df.dropna()

print()

#Rimozione duplicati
df.drop_duplicates(inplace=True)

#Impostiamo variabili categoriche quelle che non sono numeriche
df["MONTH"]=df["MONTH"].astype("category")
df["BBQ"]=df["BBQ"].astype("category")

#Rimuoviamo la colonna con le date
df.drop(columns = ["DATE"])

#Rimozione valori fuori soglia




