#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
import warnings
from scipy.stats import t
from scipy.stats import norm
import math


#%% IMPORT DATASET

#%% Import dati meteorologici
#Caricamento del dataset contenente i dati meteorologici
weather_data = pd.read_csv("../data/raw/weather_prediction_dataset.csv")

print(f"Il weather dataset ha {weather_data.shape[0]} record e {weather_data.shape[1]} colonne")

print("Etichette colonne weather_data:")
print(weather_data.columns)

print()

#%% Import condizioni meteorologiche
#Caricamento del dataset indicante le registrazioni con meteo da bbq e quelle non
classification_data = pd.read_csv("../data/raw/weather_prediction_bbq_labels.csv")

print(f"Il classification dataset ha {classification_data.shape[0]} record e {classification_data.shape[1]} colonne")

print("Etichette colonne classification_data:")
print(classification_data.columns)

print()

#%% Analisi struttura dataset
#Calcolo numero di parametri meteorologici registrati per città
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
print()

#%% Selezione città: OSLO
#Selezione città e individuazione indici nel dataset
citta = "OSLO"
chiavi = list(conteggi.keys()) #lista città
valori = list(conteggi.values()) #num parametri associati ad ogni citta
indici_citta = [0,0]
indici_citta[0] = sum(valori[:chiavi.index(citta)]) + 2 #posizione prima colonna della città
indici_citta[1] = indici_citta[0] + conteggi[citta] #posizione ultima colonna della città

#Costruzione dataset città
df = weather_data.iloc[:, indici_citta[0] : indici_citta[1]] #dati meteorologici città
df["MONTH"] = weather_data["MONTH"]
df["BBQ"] = classification_data[citta + "_BBQ_weather"] 
df["DATE"] = classification_data["DATE"]
df.to_csv(f"../data/processed/{citta}_weather_dataset.csv") #export del dataset della città creato

#%% PRE-PROCESSING

#%% Rimozione valori nulli e duplicati
#Analisi valori nulli
print("Valori NaN per colonna:")
print(df.isnull().sum())
print()

#Rimozione valori nulli
df.dropna(inplace = True)

#Rimozione duplicati
df.drop_duplicates(inplace=True)

#%% Variabili categoriche e numeriche
#Imposta variabili categoriche quelle non numeriche
num_type = ["float64", "int64"]

print("Feature categoriche e numeriche: ")
for col in df.columns:
    print(f"{col} type: {df[col].dtype}.")
    if df[col].dtype not in num_type:
        df[col] = df[col].astype("category")
        print(f"{col} type: {df[col].dtype}.")
    print("-" * 45)
print()

df["MONTH"]=df["MONTH"].astype("category")
df["BBQ"]=df["BBQ"].astype("category")
df["DATE"] = df["DATE"].astype("category")

#Sub-dataset con solo colonne numeriche
colonne_numeriche = df.select_dtypes(include=num_type)

#Rimozione della colonna con le date
df.drop(columns = ["DATE"], inplace=True)


#%% Outliers

#BoxPlot delle colonne numeriche per individuazione outliers

# Metodo per nascondere gli assi non utilizzati (grafici vuoti)
def rimuovi_assi_superflui():
    for j in range(i + 1, 3 * 4):
        fig.delaxes(axes[j // 4, j % 4])
        
#Viene scelta una distribuzione su 3 righe e 4 colonne perchè il numero massimo di features meteorologiche è 11
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 8))

for i, colonna in enumerate(colonne_numeriche):
    ax = axes[i // 4, i % 4]
    sns.boxplot(data = df[colonna], ax = ax)
    plt.title(f"{colonna}")

rimuovi_assi_superflui()
plt.suptitle("Boxplot colonne numeriche")
plt.tight_layout()
plt.show()

#Intervalli feature numeriche
estremi = colonne_numeriche.describe().loc[["min", "max"]]
print(estremi)

#Rimozione delle giornate con più di 20 ore di luce
df = df[df[f"{citta}_sunshine"] < 20]

#Rimozione delle temperature minime sotto lo 0 se si verificano nei mesi estivi
df = df[~((df[f"{citta}_temp_min"] < 0) & (df['MONTH'].isin([6, 7, 8])))]

#Calcolo outliers sospetti (distanza dal centro superiore a 3 IQR)
def calcola_outliers(valori):
    # Converte in array numpy
    valori = np.array(valori)

    # Calcola il primo quartile (Q1), il terzo quartile (Q3) e IQR
    Q1 = np.percentile(valori, 25)
    Q3 = np.percentile(valori, 75)
    IQR = Q3 - Q1
    
    # Calcola i limiti per gli outliers sospetti
    limite_inferiore = Q1 - 3 * IQR
    limite_superiore = Q3 + 3 * IQR
    
    return limite_inferiore, limite_superiore

#Rimozione degli outliers sospetti
#La rimozione avviene solo per 5 parametri meteorologici su cui non sono già state operate differenti valutazioni
for colonna in ["_wind_speed", "_wind_gust", "_humidity", "_pressure", "_precipitation"]:
    colonna = f"{citta}" + colonna
    limite_inferiore, limite_superiore = calcola_outliers(df[colonna])
    df = df[(df[colonna] > limite_inferiore) & (df[colonna] < limite_superiore)]

#Aggiornamento del sub-dataset con solo colonne numeriche
colonne_numeriche = df.select_dtypes(include=num_type)

#Export dataset ripulito
df.to_csv(f"../data/processed/{citta}_weather_dataset_clean.csv")

#%% EDA

#%% Distribuzione valori colonne numeriche
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 8))


for i, colonna in enumerate(colonne_numeriche):
    ax = axes[i // 4, i % 4]
    sns.histplot(df[colonna], kde = True, bins = 20, color="orange", ax = ax)
    ax.set_title(f"{colonna}")

rimuovi_assi_superflui()
plt.suptitle("Distribuzione valori feature")
plt.tight_layout()
plt.show()

#%% Ripartizione dei record nei vari mesi dell'anno
plt.figure(figsize = (5,5))
temp = df["MONTH"].value_counts().sort_index()[::-1]
plt.pie(temp.values,
        labels = temp.index,
        startangle = 90,
        autopct='%.1f%%')
plt.title("Distribuzione mensile delle registrazioni")
plt.show()

#%% Distribuzione delle classi

#Divisione dei record in base alla condizione meteo
plt.figure(figsize = (5,5))
plt.pie(df["BBQ"].value_counts().sort_index()[::-1],#mette prima i True e poi i False
        explode=[0, 0.1], #distanza delle due fette di torta
        autopct='%.1f%%') #formattazione della percentuale
plt.legend(["True (bel tempo)", "False (brutto tempo)"]);
plt.title("Distribuzione delle classi")
plt.show()

#Distribuzione delle classi su base mensile [grafici torta]
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(8, 6))

for i in range(12):
    ax = axes[i // 4, i % 4]
    mese = df[df["MONTH"] == i+1]
    ax.pie(mese["BBQ"].value_counts().sort_index()[::-1],#mette prima i True e poi i False
        explode=[0, 0.1], #distanza delle due fette di torta
        autopct='%.1f%%', #formattazione della percentuale
        )
    ax.set_title(f"Mese {i+1}")

plt.suptitle("Classificazione su base mensile")
plt.tight_layout()
plt.show()

#%% Classificazione meteo rispetto ai parametri meteorologici

# Condizione meteo rispetto ai parametri meteorologici [kde plot]
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 8))

for i, colonna in enumerate(colonne_numeriche):
    ax = axes[i // 4, i % 4]
    sns.kdeplot(data = df, x = colonna, hue = "BBQ", fill = True, ax = ax, warn_singular=False)

rimuovi_assi_superflui()
plt.suptitle("Distribuzione classi rispetto alle feature")
plt.tight_layout()
plt.show()

# Boxplot dei parametri rispetto alla condizione meteo
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 8))

for i, colonna in enumerate(colonne_numeriche):
    ax = axes[i // 4, i % 4]
    sns.boxplot(x = "BBQ", y = colonna, data = df, ax = ax)

rimuovi_assi_superflui()
plt.suptitle("Condizioni meteo rispetto alle feature")
plt.tight_layout()
plt.show()

#%% Matrice di correlazione dei parametri meteorologici
matrice_correlazione = colonne_numeriche.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(matrice_correlazione, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice di correlazione')
plt.show()

#%% Analisi bivariate di parametri meteorologici con correlazione alta
#Ore di luce - nuvolosità
sns.boxplot(x = f"{citta}_cloud_cover", y = f"{citta}_sunshine", data = df)
plt.title("Nuvolosità - ore di luce")
plt.xlabel("Nuvolosità (okta)")
plt.ylabel("Ore di luce (h)")
plt.show()

#Irraggiamento - ore di luce
plt.scatter(df[f"{citta}_global_radiation"], df[f"{citta}_sunshine"], alpha=0.5, color="red")
plt.title("Irraggiamento - ore di luce")
plt.xlabel("Irraggiamento (100 W/m²)")
plt.ylabel("Ore di luce (h)")
plt.show()

#Irraggiamento - temperatura massima
plt.scatter(df[f"{citta}_global_radiation"], df[f"{citta}_temp_max"], alpha=0.5, color="red")
plt.title("Irraggiamento - temperatura massima")
plt.xlabel("Irraggiamento (100 W/m²)")
plt.ylabel("Temperatura massima (°C)")
plt.show()

#Irraggiamento - umidità
plt.scatter(df[f"{citta}_global_radiation"], df[f"{citta}_humidity"], alpha=0.5, color="red")
plt.title("Irraggiamento - umidità")
plt.xlabel("Irraggiamento (W/m²)")
plt.ylabel("Umidità (%)")
plt.show()

#%% Analisi multivariata
# Distribuzione classi rispetto alla temperatura media nelle giornate senza precipitazioni
precipitazioni_assenti = df[df[f"{citta}_precipitation"] == 0]
sns.kdeplot(data = precipitazioni_assenti, x = f"{citta}_temp_mean", hue = "BBQ", fill = True)
plt.title("Distribuzione classi nei giorni senza precipitazioni")
plt.figure(figsize = (5,5))
plt.show()

#%% SPLITTING

#Funzione per dividere il dataset in training set e testing set (80 - 20)
def splitting(random_seed):
    X = colonne_numeriche.values
    y = df["BBQ"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    return X_train, X_test, y_train, y_test

# Divisione del dataset in train set e test set
X_train, X_test, y_train, y_test = splitting(42)


#%% ADDESTRAMENTO
# Scelta di alcuni modelli standard per l'addestramento
# Vengono scelti alcuni valori standard per gli iperparametri

#Logistic Regression
#Non mostra i warning riguardanti i casi in cui il modello non raggiunge la convergenza
warnings.filterwarnings("ignore", category=ConvergenceWarning)
model_logistic_regression = LogisticRegression(max_iter=100)

#SVC
k = "linear"
c = 10
model_SVC = SVC(kernel = k, C = c)

#SVM poly
k = "poly"
d = 3
model_SVM_poly = SVC(kernel = k, C = c, degree=d)

#SVM rbf
k = "rbf"
g = 1
model_SVM_rbf = SVC(kernel = k, C = c, gamma = g)


modelli_SVM = [model_SVC, model_SVM_poly, model_SVM_rbf]

#Addestramento dei modelli sul training set
model_logistic_regression.fit(X_train, y_train)

for modello in modelli_SVM:
    modello.fit(X_train, y_train)

#%% HYPERPARAMETER TUNING

#%% Divisione del testing set in validation e testing (50 - 50)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42)

#%% Definizione dei valori degli iperparametri da testare
# Definisce alcuni valori degli iperparametri per SVM 
param_grid_SVC = {
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4],  # Solo per il kernel 'poly'
    'gamma': ['scale', 'auto', 1]  # Solo per i kernel 'rbf' e 'poly'
}

# Definisce alcuni valori degli iperparametri per Regressione Logistica
param_grid_logistic_regression = {
    'solver' : ['saga', 'liblinear'],
    'C': [0.1, 1, 10, 100]
}


#%% Hyperparameter tuning su modelli SVM

modello_migliore_SVM = None
accuratezza_SVM = 0
parametri_migliori_SVM = {}

for modello in modelli_SVM:
    
    # Prova ogni modello con tutte le combinazioni di valori definiti per gli iperparametri
    grid_search = GridSearchCV(modello, param_grid_SVC, cv=5, scoring='accuracy')
    grid_search.fit(X_val, y_val)
    
    # Mantiene salvato il modello con accuratezza migliore e i suoi parametri associati
    if grid_search.best_score_ > accuratezza_SVM:
        accuratezza_SVM = grid_search.best_score_
        modello_migliore_SVM = grid_search.best_estimator_
        parametri_migliori_SVM = grid_search.best_params_

#Accuratezza del modello migliore sul validation set
accuratezza_SVM = accuracy_score(y_val, modello_migliore_SVM.predict(X_val))

print("Miglior modello SVC:", modello_migliore_SVM)
print("Migliori parametri SVC:", parametri_migliori_SVM)
print("Accuratezza SVC sul validation set:", accuratezza_SVM)
print()

#%% Hyperparameter tuning su modello Regressione Logistica

# Prova il modello con ogni combinazione di valori definiti per gli iperparametri
grid_search_logistica = GridSearchCV(model_logistic_regression, param_grid_logistic_regression, cv=5, scoring='accuracy')
grid_search_logistica.fit(X_val, y_val)

# Determina il miglior estimatore e i migliori parametri
modello_migliore_logistica = grid_search_logistica.best_estimator_
parametri_migliori_logistica = grid_search_logistica.best_params_

# Accuratezza del modello migliore sul validation set
accuratezza_logistica = accuracy_score(y_val, modello_migliore_logistica.predict(X_val))
accuratezza_logistica = grid_search_logistica.best_score_

print("Miglior modello Regressione Logistica:", modello_migliore_logistica)
print("Migliori parametri Regressione Logistica:", parametri_migliori_logistica)
print("Accuratezza Regressione Logistica sul validation set:", accuratezza_logistica)
print()

#%% Selezione del modello migliore

# Sceglie il miglior modello in base all'accuratezza
modello = None
accuratezza_validation = 0
if accuratezza_SVM > accuratezza_logistica:
    modello = modello_migliore_SVM
    accuratezza_validation = accuratezza_SVM
else:
    modello = modello_migliore_logistica
    accuratezza_validation = accuratezza_logistica

print(f"Modello scelto: {modello}")

#%% VALUTAZIONE PERFORMANCE

#%% Matrice di confusione 

# Effettua le predizioni sul testing set
y_pred = modello.predict(X_test)

#Calcola matrice di confusione
matrice_confusione = confusion_matrix(y_test, y_pred)

# Creazione dell'heatmap della matrice di confusione
plt.figure(figsize=(10, 8))
labels = ['False', 'True']
sns.heatmap(matrice_confusione, annot=True, fmt="d", cmap="Oranges", cbar=False, xticklabels=labels, yticklabels=labels)
plt.title("Matrice di confusione")
plt.xlabel("Predetti")
plt.ylabel("Reali")
plt.show()

#%% Metriche

#Funzione per calcolo delle metriche a partire dalla matrice di confusione
def calcolo_metriche(matrice_confusione):
    
    tn, fp, fn, tp = matrice_confusione.ravel()

    accuratezza = (tp + tn) / (tp + tn + fp + fn)
    sensitivita = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificita = tn / (tn + fp) if (tn + fp) > 0 else 0
    precisione = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return accuratezza, sensitivita, specificita, precisione, npv

# Calcolo delle metriche
accuratezza, sensitivita, specificita, precisione, npv = calcolo_metriche(matrice_confusione)

# Stampa metriche
print("METRICHE MODELLO:")
print(f"Accuratezza: {accuratezza:.4f}")
print(f"MR: {(1 - accuratezza):.4f}")
print(f"Sensitività: {sensitivita:.4f}")
print(f"Specificità: {specificita:.4f}")
print(f"Precisione: {precisione:.4f}")
print(f"Valore Predittivo Negativo (NPV): {npv:.4f}")
print()

#%% Overfitting e underfitting
#Effettua predizione sul training set per valutare overfitting/underfitting
y_pred_train = modello.predict(X_train)

print(f"Accuratezza sul testing set: {accuratezza:.4f}")
print(f"Accuratezza sul validation set: {accuratezza_validation:.4f}")
print(f"Accuratezza sul training set: {(accuracy_score(y_train, y_pred_train)):.4f}")

#%% STUDIO STATISTICO SUI RISULTATI DELLA VALUTAZIONE

#%% Geenerazione dei campioni
# Generazione casuale di k valori, usati come semi generatori per la costruzione di k training e testing set
k = 20
semi_generatori = np.random.randint(low = 0, high = 100, size = k) 

#Dizionario con chiavi le metriche e valori la lista di k valori misurati per ogni metrica
metriche_srs = {
    'accuratezza': [],
    'sensitivita': [],
    'specificita': [],
    'precisione': [],
    'npv': []
}


for seed in semi_generatori:

    #Generazione di un nuovo training set e testing set
    X_train, X_test, y_train, y_test = splitting(seed)

    #Training e testing del modello
    modello.fit(X_train, y_train)
    y_pred = modello.predict(X_test)

    #Calcolo della matrice di confusione
    matrice_confusione = confusion_matrix(y_test, y_pred)

    #Calcolo delle metriche e inserimento dei valori nel dizionario
    metriche = calcolo_metriche(matrice_confusione)
    metriche_srs['accuratezza'].append(metriche[0]) 
    metriche_srs['sensitivita'].append(metriche[1])  
    metriche_srs['specificita'].append(metriche[2])  
    metriche_srs['precisione'].append(metriche[3])  
    metriche_srs['npv'].append(metriche[4])

#%% Stima media e intervallo di confidenza

#Funzione per calcolare l'intervallo di confidenza per la media di un campione di n elementi
def inferenza_media(n, campione, alfa):
    media = np.mean(campione) #calcola la media del campione
    S_2 = campione.var(ddof = 1) #stimatore corretto della varianza
    S = math.sqrt(S_2) #deviazione standard stimata
    limite_superiore = limite_inferiore = 0
    
    if n < 40: #se l'intervallo ha meno di 40 elementi utilizza il quantile della distribuzione t di student
        t_value = t.ppf(1 - alfa/2, n-1)
        limite_inferiore = media - t_value * (S / math.sqrt(n))
        limite_superiore = media + t_value * (S / math.sqrt(n))
    
    else: #se l'intervallo ha 40 o più elementi utilizza il quantile della distribuzione normale
        z_value = norm.ppf(1 - alfa/2)
        limite_inferiore = media - z_value * (S / math.sqrt(n))
        limite_superiore = media + z_value * (S / math.sqrt(n))
        
    print(f"Intervallo di confidenza della media: [{limite_inferiore:.4f}; {limite_superiore:.4f}]")

#%% Descrizione campioni e inferenza della media

print(f"Numero di elementi dei campioni: {k}")
print("Studio statistico sui risultati della valutazione:")
print()

# Ciclo sulle metriche del dizionario
# Per ogni metrica viene effettuata un'analisi descrittiva del campione e viene fatta inferenza sulla media
for metrica in metriche_srs:
    #Selezione del campione, trasformato in array numpy
    metriche_srs[metrica] = np.array(metriche_srs[metrica])
    campione = metriche_srs[metrica]

    #Calcolo delle misure di statistica descrittiva
    print(f"{metrica.upper()}")
    print(f"Media: {(np.mean(campione)):.4f}")
    inferenza_media(len(campione), campione, 0.05) #calcolo dell'intervallo di confidenza della media
    print(f"Mediana: {(np.median(campione)):.4f}")
    
    print(f"Varianza: {(campione.var(ddof = 0)):.4f}")
    print(f"IQR: {(np.percentile(campione, 75) - np.percentile(campione, 25)):.4f}")

    #Stampa dei grafici di distribuzione dei valori
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
    
    sns.histplot(campione, kde = True, bins = k, color="orange", ax = axes[0]) #istogramma
    sns.boxplot(data = campione, ax = axes[1]) #boxplot

    plt.suptitle(f"{metrica.upper()}")
    plt.tight_layout()
    plt.show()
    print()
    
#%%