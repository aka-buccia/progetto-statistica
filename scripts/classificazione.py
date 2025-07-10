#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning
import warnings
from scipy.stats import t as tt
from scipy.stats import norm
import math
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import shapiro
import scipy.stats as stats
import statsmodels.api as sm

#%% IMPORT DATASET
#%% Import dati metereologici
#Caricamento del dataset contenente i dati metereologici
weather_data = pd.read_csv("../data/raw/weather_prediction_dataset.csv")

print(f"Il weather dataset ha {weather_data.shape[0]} record e {weather_data.shape[1]} colonne")

print("Etichette colonne weather_data:")
print(weather_data.columns)

print()

#%% Import condizioni metereologiche
#Caricamento del dataset indicante le registrazioni con meteo da bbq e quelle non
classification_data = pd.read_csv("../data/raw/weather_prediction_bbq_labels.csv")

print(f"Il classification dataset ha {classification_data.shape[0]} record e {classification_data.shape[1]} colonne")

print("Etichette colonne classification_data:")
print(classification_data.columns)

print()

#%% Analisi struttura dataset
#Calcolo numero di parametri metereologici registrati per città
conteggi = {}

for item in weather_data.columns[2:]:
    citta = item.split("_")[0]
    if citta in conteggi:
        conteggi[citta] += 1
    else:
        conteggi[citta] = 1

#Stampa città e numero di parametri associati
print("Numero parametri meteorologici per città:")
for citta, conteggio in conteggi.items():
    print(f"{citta}: {conteggio}")    
print()

#Numero città registrate nei database
print(f"Numero città weather dataset: {len(conteggi.keys())}")
print(f"Numero città classification dataset: {len(classification_data.columns) -1}")

#%% Selezione città: OSLO
#Selezione città e individuazione indici nel dataset
citta = "OSLO"
chiavi = list(conteggi.keys()) #lista città
valori = list(conteggi.values()) #num parametri associati ad ogni citta
indici_citta = [0,0]
indici_citta[0] = sum(valori[:chiavi.index(citta)]) + 2 #posizione prima colonna della città
indici_citta[1] = indici_citta[0] + conteggi[citta] #posizione ultima colonna della città

#Costruzione dataset città
df = weather_data.iloc[:, indici_citta[0] : indici_citta[1]] #dati metereologici città
df["MONTH"] = weather_data["MONTH"]
df["BBQ"] = classification_data[citta + "_BBQ_weather"] 
df["DATE"] = classification_data["DATE"]
df.to_csv(f"../data/processed/{citta}_weather_dataset.csv") #salvataggio del dataset creato

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
plt.title("Distribuzione mensile delle registrazioni")
plt.show()


#Percentuale di record con meteo per bbq
plt.figure(figsize = (5,5))
plt.pie(df["BBQ"].value_counts().sort_index()[::-1], 
        explode=[0, 0.1],
        autopct='%.1f%%')
plt.legend(["True", "False"]);
plt.title("Distribuzione delle classi")
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

# Boxplot dei parametri rispetto alla condizione meteo
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 8))

for i, colonna in enumerate(colonne_numeriche):
    ax = axes[i // 4, i % 4]
    sns.boxplot(x = "BBQ", y = colonna, data = df, ax = ax)

# Nascondi gli assi non utilizzati
for j in range(i + 1, 3 * 4):
    fig.delaxes(axes[j // 4, j % 4])
    
plt.tight_layout()
plt.show()

# Matrice di correlazione dei parametri metereologici
matrice_correlazione = colonne_numeriche.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(matrice_correlazione, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice di correlazione')
plt.show()


#Analisi bivariate di parametri metereologici con correlazione alta
#Ore di luce - nuvolosità
sns.boxplot(x = f"{citta}_cloud_cover", y = f"{citta}_sunshine", data = df)
plt.show()

#Irraggiamento - ore di luce
plt.scatter(df[f"{citta}_global_radiation"], df[f"{citta}_sunshine"], alpha=0.5, color="red")
plt.title("Scatter plot irraggiamento - ore di luce")
plt.xlabel("Irraggiamento (100 W/m2)")
plt.ylabel("Ore di luce (h)")
plt.show()

#Irraggiamento - temperatura massima
plt.scatter(df[f"{citta}_global_radiation"], df[f"{citta}_temp_max"], alpha=0.5, color="red")
plt.title("Scatter plot irraggiamento - temperatura massima")
plt.xlabel("Irraggiamento (100 W/m2)")
plt.ylabel("Temperatura massima (Celsius)")
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
warnings.filterwarnings("ignore", category=ConvergenceWarning)
model_logistic_regression = LogisticRegression(solver = "liblinear", max_iter=100)

#SVM poly
k = "poly"
d = 3
model_SVM_poly = SVC(kernel = k, C = c, degree=d)

#SVM rbf
k = "rbf"
g = 1
model_SVM_rbf = SVC(kernel = k, C = c, gamma = g)


modelli_SVM = [model_SVC, model_SVM_poly, model_SVM_rbf]

#addestramento dei modelli sul training set
model_logistic_regression.fit(X_train, y_train)

for modello in modelli_SVM:
    modello.fit(X_train, y_train)


#%% HYPERPARAMETER TUNING
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42)

# Definire gli spazi degli iperparametri per SVM e Regressione Logistica
param_grid_SVC = {
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4],  # Solo per il kernel 'poly'
    'gamma': ['scale', 'auto']  # Solo per i kernel 'rbf' e 'poly'
}

param_grid_logistic_regression = {
    'solver' : ['liblinear', 'saga'],
    'C': [0.1, 1, 10, 100]
}

# GridSearchCV per SVM
best_model_SVC = None
accuracy_SVC = 0
best_params_SVC = {}

for modello in modelli_SVM:
    # Tuning degli iperparametri in base all'accuratezza
    grid_search = GridSearchCV(modello, param_grid_SVC, cv=5, scoring='accuracy')
    grid_search.fit(X_val, y_val)
    
    # Determina il miglior modello e i migliori parametri
    if grid_search.best_score_ > accuracy_SVC:
        accuracy_SVC = grid_search.best_score_
        best_model_SVC = grid_search.best_estimator_
        best_params_SVC = grid_search.best_params_

print("Miglior modello SVC:", best_model_SVC)
print("Migliori parametri SVC:", best_params_SVC)
print("Accuratezza SVC sul validation set:", accuracy_SVC)

# GridSearchCV per Logistic Regression
grid_search_logistic = GridSearchCV(model_logistic_regression, param_grid_logistic_regression, cv=5, scoring='accuracy')
grid_search_logistic.fit(X_val, y_val)

# Determina il miglior estimatore e i migliori parametri
best_model_logistic = grid_search_logistic.best_estimator_
best_params_logistic = grid_search_logistic.best_params_

# Prestazioni sui validation set
accuracy_logistic = accuracy_score(y_val, best_model_logistic.predict(X_val))

print("Miglior modello Logistic Regression:", best_model_logistic)
print("Migliori parametri Logistic Regression:", best_params_logistic)
print("Accuratezza Logistic Regression sul validation set:", accuracy_logistic)


# Determina il miglior modello in base alle accuratezze
modello = None
accuratezza_validation = 0
if accuracy_SVC > accuracy_logistic:
    modello = best_model_SVC
    accuratezza_validation = accuracy_SVC
else:
    modello = best_model_logistic
    accuratezza_validation = accuracy_logistic

#%% VALUTAZIONE PERFORMANCE    

# Effettua le predizioni sul testing dataset
y_pred = modello.predict(X_test)

# Calcolo matrice di confusione
matrice_confusione = confusion_matrix(y_test, y_pred)


# Creazione dell'heatmap della matrice di confusione
plt.figure(figsize=(10, 8))
labels = ['False', 'True']
sns.heatmap(matrice_confusione, annot=True, fmt="d", cmap="Oranges", cbar=False, xticklabels=labels, yticklabels=labels)
plt.title("Matrice di confusione")
plt.xlabel("Predetti")
plt.ylabel("Reali")
plt.show()


# Calcolo delle metriche
tn, fp, fn, tp = matrice_confusione.ravel()

accuratezza = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

# Stampa metriche
print(f"Accuratezza: {accuratezza:.4f}")
print(f"Errore di misclassificazione: {(1 - accuratezza):.4f}")
print(f"Sensitività: {sensitivity:.2f}")
print(f"Specificità: {specificity:.2f}")
print(f"Precisione: {precision:.2f}")
print(f"Valore Predittivo Negativo (NPV): {npv:.2f}")

#%% STUDIO STATISTICO SUI RISULTATI DELLA VALUTAZIONE
for k in np.random.randint(low = 0, high = 100, size = 10):

    #Generazione di due nuovi testing dataset e training dataset
    X_train, X_test, y_train, y_test = splitting(k)

    #Training e testing del dataset
    modello.fit(X_train, y_train)
    y_pred = modello.predict(X_test)





