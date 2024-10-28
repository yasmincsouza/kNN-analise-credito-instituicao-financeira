# KNeighborsClassifier para Análise de Concessão de Crédito em Instituições Financeiras
# Importando as bibliotecas necessárias
import pandas as pd 
import matplotlib.pyplot as plt
import pre_processamento

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

dataP = pre_processamento

# Divide os dados em conjuntos de treinamento (80%) e teste (20%).
X_train, X_test, y_train, y_test = train_test_split(dataP.X, dataP.y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Considera 6 vizinhos mais próximos.
KNC = KNeighborsClassifier(n_neighbors=6)
KNC.fit(X_train, y_train)

y_pred = KNC.predict(X_test)
y_prob = KNC.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred) # Porcentagem de previsões corretas
auc = roc_auc_score(y_test, y_prob) # Métrica de desempenho
report = classification_report(y_test, y_pred) # Relatório com métricas como precisão, recall e F1-score.
    
print(report)

print("\n")
print("######################################## Input de Novos Clientes ########################################")
# Lendo o arquivo credit_risk_dataset.csv utilizando a biblioteca pandas
try:
    dados_novos = pd.read_csv('dados/novos_clientes.csv')
    print("Arquivo CSV carregado com sucesso!")
except FileNotFoundError:
    print("Arquivo CSV não encontrado!")

# Pré-processamento dos novos dados
dados_novos = dados_novos.drop(dados_novos[dados_novos['person_age'] > 100].index)

ord_enc = OrdinalEncoder()

dados_novos['person_home_ownership'] = ord_enc.fit_transform(dados_novos[['person_home_ownership']])
dados_novos['loan_intent'] = ord_enc.fit_transform(dados_novos[['loan_intent']])
dados_novos['loan_grade'] = ord_enc.fit_transform(dados_novos[['loan_grade']])
dados_novos['cb_person_default_on_file'] = ord_enc.fit_transform(dados_novos[['cb_person_default_on_file']])
dados_novos.drop_duplicates(inplace=True)
        
# Imputação e escalonamento
dados_novos['person_emp_length'] = dados_novos['person_emp_length'].fillna(dados_novos['person_emp_length'].mean())
dados_novos_imp = pre_processamento.imp.transform(dados_novos)
dados_novos_scaled = scaler.transform(dados_novos_imp)
        
# Previsão usando o modelo KNN
previsoes = KNC.predict(dados_novos_scaled)
probabilidades = KNC.predict_proba(dados_novos_scaled)[:, 1]
        
# Exibir resultados
for i, pred in enumerate(previsoes):
    print(f"Cliente {i+1}:")
    print(f"  Previsão: {'Aprovado' if pred == 1 else 'Negado'}")
    print(f"  Probabilidade de aprovação: {int(probabilidades[i] * 100)}%")