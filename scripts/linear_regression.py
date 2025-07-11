import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import shapiro
import statsmodels.api as sm


#%% Import database ripulito
citta = "OSLO"
df = pd.read_csv(f"../data/processed/{citta}_weather_dataset_clean.csv")

#%% Addestramento del modello e predizione

# Imposta variabile indipendente e variabile dipendente
X = df[f"{citta}_temp_min"].values.reshape(-1, 1)
Y_osservati = df[f"{citta}_temp_max"]

#Training del modello di regressione lineare
modello_regressione_lineare = LinearRegression()
modello_regressione_lineare.fit(X, Y_osservati)

#Predizione tramite il modello
Y_predetti = modello_regressione_lineare.predict(X)

#%% Coefficienti retta di regressione

intercetta = modello_regressione_lineare.intercept_
coefficiente_angolare = modello_regressione_lineare.coef_[0]
print(f"Intercetta (beta0): {intercetta:.2f}")
print(f"Pendenza (beta1): {coefficiente_angolare:.2f}")

#%% Grafico retta di regressione [scatter plot]

# Disegna i punti osservati (in blu) e la retta di regressione (in rosso)
plt.scatter(X, Y_osservati, color='blue')
plt.plot(X, Y_predetti, color='red')

plt.xlabel("Temperatura minima (°C)")
plt.ylabel('Temperatura massima (°C)')
plt.title('Regressione Lineare semplice')
plt.grid(True) #mostra la griglia per dare visibilità all'intercetta
plt.show()

#%% Metriche di valutazione

print(f"r²: {(r2_score(Y_osservati, Y_predetti)):.4f}")
print(f"MSE: {(mean_squared_error(Y_osservati, Y_predetti)):.4f} °C²")

#%% Analisi di normalità dei residui

#Calcolo residui
residui = Y_osservati - Y_predetti

# Q-Q plot
sm.qqplot(residui, line='s')
plt.title('Q-Q Plot dei Residui')
plt.xlabel('Quantili Teorici')
plt.ylabel('Quantili dei Residui')
plt.show()

## Verifica normalità dei residui (test di Shapiro-Wilk)
shapiro_test = shapiro(residui)
print(f"Test di Shapiro-Wilk p-value: {(shapiro_test.pvalue):.4e}")


#Istogramma per osservare la distribuzione dei residui [istogramma]
sns.histplot(residui, kde = True, color="orange")
plt.xlabel("Residui")
plt.ylabel("Numero residui")
plt.title("Distribuzione residui")

#Valore medio dei residui
print(f"Media dei residui: {(np.mean(residui)):.4e}")

#Outliers residui [boxplot]
plt.figure(figsize = (5, 5))
sns.boxplot(residui)
plt.title('Boxplot dei Residui')
plt.ylabel("Dimensione residui (°C)")
plt.show()
