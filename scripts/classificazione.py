#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#%% IMPORT WEATHER DATASET
#Caricamento del dataset contenente i dati metereologici
weather_data = pd.read_csv("../data/raw/weather_prediction_dataset.csv")

#print(weather_data.head())

print(f"Il weather dataset ha {weather_data.shape[0]} record e {weather_data.shape[1]} colonne")


print("Etichette colonne weather_data:")
print(weather_data.columns)
# for etichetta_colonna in weather_data.columns:
#     print(etichetta_colonna)
print()

#%% IMPORT BBQ LABEL DATASET
#Caricamento del dataset indicante le registrazioni con meteo da bbq e quelle non
classification_data = pd.read_csv("../data/raw/weather_prediction_bbq_labels.csv")

#print(classification_data.head())
print(f"Il bbq label dataset ha {classification_data.shape[0]} record e {classification_data.shape[1]} colonne")

print("Etichette colonne classification_data:")
print(classification_data.columns)
# for etichetta_colonna in classification_data.columns:
#     print(etichetta_colonna)
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
df.to_csv(f"../data/processed/{citta}_weather_dataset.csv")

#%% PRE-PROCESSING
#Analisi valori nulli
print("Valori NaN")
print(df.isnull().sum())

#Rimozione valori nulli
df.dropna(inplace = True)

print()

#Rimozione duplicati
df.drop_duplicates(inplace=True)

#Impostiamo variabili categoriche quelle che non sono numeriche
num_type = ["float64", "int64"]

for col in df.columns:
    print(f"{col} type: {df[col].dtype}.")
    if df[col].dtype not in num_type:
        df[col] = df[col].astype("category")
        print(f"{col} type: {df[col].dtype}.")
    print("-" * 45)

df["MONTH"]=df["MONTH"].astype("category")
df["BBQ"]=df["BBQ"].astype("category")
df["DATE"] = df["DATE"].astype("category")

#Sub-dataset con solo colonne numeriche
colonne_numeriche = df.select_dtypes(include=num_type)

#Rimuoviamo la colonna con le date
df.drop(columns = ["DATE"], inplace=True)

#BoxPlot delle colonne numeriche per individuazione outliers
#Viene scelta una distribuzione su 3 righe e 4 colonne perchè il numero massimo di features metereologiche è 11
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 8))

for i, colonna in enumerate(colonne_numeriche):
    ax = axes[i // 4, i % 4]
    sns.boxplot(data = df[colonna], ax = ax)
    plt.title(f"{colonna}")

# Nascondi gli assi non utilizzati
for j in range(i + 1, 3 * 4):
    fig.delaxes(axes[j // 4, j % 4])
    
plt.tight_layout()
plt.show()

#Valutazione estremi
riassunto = colonne_numeriche.describe()
estremi = riassunto.loc[["min", "max"]]
print(estremi)


# print(f"Intervallo temperatura minima: [{min(df[f"{citta}_temp_min"])}; {max(df[f"{citta}_temp_min"])}]")
# print(f"Intervallo temperatura media: [{min(df[f"{citta}_temp_mean"])}; {max(df[f"{citta}_temp_mean"])}]")
# print(f"Intervallo temperatura massima: [{min(df[f"{citta}_temp_max"])}; {max(df[f"{citta}_temp_max"])}]")


#OSLO è sotto al circolo polare artico, 
#le ore di luce dunque non possono essere più di 20
df = df[df[f"{citta}_sunshine"] < 20]


#Rimozione outliers sospetti (distanza dal centro superiore a 3 IQR)
def calcola_outliers(valori):
    # Converti l'array in un array NumPy
    valori = np.array(valori)

    # Calcola il primo quartile (Q1), il terzo quartile (Q3) e IQR
    Q1 = np.percentile(valori, 25)
    Q3 = np.percentile(valori, 75)
    IQR = Q3 - Q1
    
    # Calcola i limiti per gli outliers
    limite_inferiore = Q1 - 3 * IQR
    limite_superiore = Q3 + 3 * IQR
    
    return limite_inferiore, limite_superiore

#rimozione degli outliers sospetti relativi a 5 parametri metereologici su cui non sono già state operate differenti valutazioni
for colonna in ["_wind_speed", "_wind_gust", "_humidity", "_pressure", "_precipitation"]:
    colonna = f"{citta}" + colonna
    limite_inferiore, limite_superiore = calcola_outliers(df[colonna])
    df = df[(df[colonna] > limite_inferiore) & (df[colonna] < limite_superiore)]

#aggiorno il dataset delle colonne numeriche
colonne_numeriche = df.select_dtypes(include=num_type)

#%% EDA

#Distribuzione valori colonne numeriche
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 8))


for i, colonna in enumerate(colonne_numeriche):
    ax = axes[i // 4, i % 4]
    sns.histplot(df[colonna], kde = True, bins = 20, color="orange", ax = ax)
    ax.set_title(f"{colonna}")

plt.tight_layout()
plt.show()

#Ripartizione dei record nei vari mesi dell'anno
plt.figure(figsize = (5,5))
temp = df["MONTH"].value_counts().sort_index()[::-1]
plt.pie(temp.values,
        labels = temp.index,
        startangle = 90,
        autopct='%.1f%%')
plt.show()


#Percentuale di record con condizioni meteo per bbq
plt.figure(figsize = (5,5))
plt.pie(df["BBQ"].value_counts().sort_index(), 
        explode=[0, 0.1],
        autopct='%.1f%%')
plt.legend(["False", "True"]);
plt.show()

# Condizione meteo rispetto ai parametri metereologici
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 8))
for i, colonna in enumerate(colonne_numeriche):
    ax = axes[i // 4, i % 4]
    sns.kdeplot(data = df, x = colonna, hue = "BBQ", fill = True, ax = ax, warn_singular=False)
    plt.title(f"{colonna} condition for BBQ")

# Nascondi gli assi non utilizzati
for j in range(i + 1, 3 * 4):
    fig.delaxes(axes[j // 4, j % 4])

plt.tight_layout()
plt.show()


#%% SPLITTING
def splitting(random_seed):
    X = colonne_numeriche.values
    print(X)
    y = df["BBQ"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = splitting(42)


#%% ADDESTRAMENTO
# Scelta di alcuni modelli standard per l'addestramento

#SVC
k = "linear"
c = 10
model_SVC = SVC(kernel = k, C = c)

#Logistic Regression
model_logistic_regression = LogisticRegression(solver = "liblinear", max_iter=100)

#SVM poly
k = "poly"
d = 3
model_SVM_poly = SVC(kernel = k, C = c, degree=d)

#SVM rbf
k = "rbf"
g = 1
model_SVM_rbf = SVC(kernel = k, C = c, gamma = g)


modelli = [model_SVC, model_SVM_poly, model_SVM_rbf]

#addestramento dei modelli sul training set
model_logistic_regression.fit(X_train, y_train)

for modello in modelli:
    modello.fit(X_train, y_train)


#%% HYPERPARAMETER TUNING









#%% VALUTAZIONE PERFORMANCE    
# Misuriamo l'accuratezza del modello
y_pred = modello.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)

# Calcolare l'accuratezza sulla validation set
accuracy_val = accuracy_score(y_test, y_pred)
print(f"Accuracy sul validation set: {accuracy_val:.4f}")

# Creazione dell'heatmap della matrice di confusione
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Oranges", cbar=False)
plt.title(f"Modello ...")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()








