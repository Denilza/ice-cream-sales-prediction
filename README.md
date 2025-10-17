# 🍦 Prevendo Vendas de Sorvete com Machine Learning  

<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/869/869869.png" width="120px" alt="ice cream icon"/>
</p>

## 🎯 Objetivo  

Este projeto tem como objetivo **prever as vendas diárias de sorvetes com base na temperatura ambiente**, utilizando um modelo de **regressão linear**.  
A proposta é auxiliar donos de sorveterias na **tomada de decisão sobre a produção**, reduzindo desperdícios e maximizando lucros.  

---

## 🧩 Cenário  

A sorveteria fictícia **Gelato Mágico**, localizada em uma cidade litorânea, percebeu que as vendas de sorvete variam conforme a temperatura.  
Para otimizar a produção, foi desenvolvido um modelo de *Machine Learning* capaz de prever o número de sorvetes vendidos em função da temperatura.

<p align="center">
  <img src="https://cdn.pixabay.com/photo/2016/11/29/09/08/ice-cream-1869739_960_720.jpg" width="600px"/>
</p>

---

## 🧠 Tecnologias Utilizadas  

- **Python 3.10+**  
- **Pandas** – Manipulação de dados  
- **Scikit-learn** – Criação e treino do modelo de regressão  
- **MLflow** – Rastreamento de experimentos e registro de modelos  
- **Matplotlib** – Visualização dos dados  
- **Git/GitHub** – Controle de versão e portfólio  

---

---

## 📊 Dataset  

O conjunto de dados contém duas variáveis simples:  

| temperatura (°C) | vendas (unidades) |
|------------------|-------------------|
| 20 | 45 |
| 22 | 49 |
| 24 | 58 |
| 26 | 63 |
| 28 | 72 |
| 30 | 80 |
| 32 | 88 |
| 34 | 95 |
| 36 | 100 |
| 38 | 105 |
| 40 | 110 |

📁 [Baixar dataset_vendas_sorvete.csv](./inputs/dataset_vendas_sorvete.csv)

---

## ⚙️ Treinamento do Modelo  

O modelo foi treinado usando **Regressão Linear** com a biblioteca `scikit-learn`.  
Durante o processo, foram registradas métricas e parâmetros com o **MLflow** para garantir rastreabilidade e reprodutibilidade.

```python
from sklearn.linear_model import LinearRegression
import pandas as pd
import mlflow

data = pd.read_csv("inputs/dataset_vendas_sorvete.csv")
X = data[['temperatura']]
y = data['vendas']

mlflow.set_experiment("Ice Cream Sales Prediction")

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X, y)
    mlflow.sklearn.log_model(model, "modelo_icecream")


