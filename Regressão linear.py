import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Faz a  leitura do arquivo CSV usando pandas.
dataset = pd.read_csv('aptos-metro-valor.csv')

#Cria  uma base contendo as variáveis independentes e uma base contendo a variável dependente.
x = dataset.iloc[:, 0].values 
y = dataset.iloc[:, 1].values 

#Separa a base em duas partes: uma para treinamento e outra para testes. Use 85% das instâncias para o treinamento.
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.15, random_state=0)

#Constrói um modelo de regressão linear simples em função dos dados de treinamento.
linearRegression = LinearRegression()
linearRegression.fit(x_treinamento, y_treinamento)

#Exibe a reta obtida e os dados de treinamento em um mesmo gráfico. Incluaa equação obtida no título do gráfico.
plt.scatter(x_treinamento, y_treinamento, color="red")
plt.plot(x_treinamento,linearRegression.predict(x_treinamento),color="blue")
plt.title("metros x valor (Treinamento)")
plt.xlabel("valor por metro quadrado")
plt.ylabel("metros")
plt.show()


#Exibe a reta obtida e os dados de teste em um mesmo gráfico.Inclua aequação obtida no título do gráfico.
plt.scatter(x_teste, y_teste, color="red")
plt.plot(x_treinamento,linearRegression.predict(x_treinamento),color="blue")
plt.title("metros x valor (teste)")
plt.xlabel("valor por metro quadrado")
plt.ylabel("metros")
plt.show()