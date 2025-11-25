# 🏭 Manutenção Preditiva com Inteligência Artificial

Este projeto aplica **Machine Learning** para prever falhas em máquinas industriais, utilizando dados simulados de sensores.  
A manutenção preditiva é um pilar da **Indústria 4.0**, reduzindo custos, aumentando a segurança e evitando paradas inesperadas.

---

## 📌 Objetivos do Projeto

- Criar um modelo que prevê falhas em máquinas usando parâmetros como:
  - Vibração  
  - Temperatura  
  - Pressão  
- Treinar um algoritmo Random Forest.
- Avaliar o desempenho usando:
  - Matriz de Confusão
  - Relatório de Classificação
  - Importância das Features
- Fazer previsões em novos dados reais/simulados.

---

## 📂 Estrutura do Repositório
predictive_maintenance/
│ README.md
│ requirements.txt
│
├── data/
│ └── dataset_simulado.csv
│
├── notebooks/
│ └── maintenance.ipynb
│
└── src/
└── model.py

---

## 🧠 Tecnologias Usadas

- Python 3
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

## Melhorias Futuras

- Dashboard em tempo real (Streamlit)
- API com FastAPI
- Dados reais de sensores industriais
- Rede neural para detecção de anomalias

---

## Autor

Projeto criado por Guilherme Gomes

Código abaixo:

# PROJETO: MANUTENÇÃO PREDITIVA INDUSTRIAL
# ============================================

# 1. Importação
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Dataset Simulado
np.random.seed(42)
data = pd.DataFrame({
    'vibration': np.random.normal(0, 1, 1000),
    'temperature': np.random.normal(70, 5, 1000),
    'pressure': np.random.normal(30, 2, 1000),
    'failure': np.random.choice([0,1], 1000, p=[0.95,0.05])
})

data.to_csv("/content/dataset_simulado.csv", index=False)

data.head()

# 3. Treino/Teste
X = data.drop('failure', axis=1)
y = data['failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modelo
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# 5. Avaliação
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predição")
plt.ylabel("Real")
plt.show()

# 7. Importância das Features
importances = model.feature_importances_
plt.bar(X.columns, importances)
plt.title("Importância das Features")
plt.show()

# 8. Predição Exemplo
novo = pd.DataFrame({
    "vibration":[0.5],
    "temperature":[72],
    "pressure":[29]
})

print("Previsão:", model.predict(novo))





