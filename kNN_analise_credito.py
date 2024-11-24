# KNeighborsClassifier para Análise de Concessão de Crédito em Instituições Financeiras
# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.impute import KNNImputer

try:
    dados = pd.read_csv('dataset/credit_risk_dataset.csv')
    print("Arquivo CSV carregado com sucesso!")
except FileNotFoundError:
    print("Arquivo CSV não encontrado!")

# Verifica como estão estruturado os dados das primeiras entradas de cada coluna
print("\n")
print("#### Dados das primeiras entradas de cada coluna ####")
print(dados.head())

# Tipos de dados e contagem de entradas.
print("\n")
print("#### Visão Geral dos Dados ####")
print(dados.info())

print("\n")
print("#### Verificar valores ausentes ####")
print(dados.isna().sum())

print("\n")
print("#### Verificar valores duplicados ####")
print(dados.duplicated().sum())
print("\n")
      
## Algumas colunas contêm dados categóricos, então é preciso fazer uma limpeza e tranformação desses dados
# Remove todas as linhas do dataframe onde a idade é maior que 100
dados = dados.drop(dados[dados['person_age'] > 100].index)

ord_enc = OrdinalEncoder()

dados['person_home_ownership'] = ord_enc.fit_transform(dados[['person_home_ownership']])
dados['loan_intent'] = ord_enc.fit_transform(dados[['loan_intent']])
dados['loan_grade'] = ord_enc.fit_transform(dados[['loan_grade']])
dados['cb_person_default_on_file'] = ord_enc.fit_transform(dados[['cb_person_default_on_file']])
dados.drop_duplicates(inplace=True) # Remove duplicatas

# Cria uma copia dos dados e preenche os valores ausentes com a média da coluna person_emp_length
df1 = dados.copy()
df1['person_emp_length'] = df1['person_emp_length'].fillna(df1['person_emp_length'].mean())

dados = df1

X = dados.drop(columns='loan_status')
y = dados['loan_status']

# Método de imputação que usa o algoritmo K-Nearest Neighbors para preencher valores ausentes
# 5 vizinhos mais próximos
imp = KNNImputer(n_neighbors=5)
X_imp = imp.fit_transform(X)
X1 = pd.DataFrame(X_imp, columns=X.columns)

# Garante que apenas os valores ausentes foram preenchidos
X1.shape
X.shape
X = X1

def graficosDados():
    plt.figure(figsize=(10,6))
    sns.barplot(x=dados['person_age'].unique(), y=dados['person_age'].value_counts())
    plt.xticks(rotation=90)
    # plt.title("Idade")
    plt.ylabel("Quantidade")
    plt.xlabel("Idade")
    plt.show()

    legenda_loan_intent = {'0.0': 'Pessoal', '1.0': 'Educação', '2.0': 'Médico', '3.0': 'Empreendimento', '4.0': 'Reforma Casa', '5.0': 'Consolidação Dívidas'}
    plt.figure(figsize=(10,6))
    sns.countplot(data = dados, x = 'loan_intent', hue = 'loan_status')
    plt.xticks(rotation=90)
    plt.ylabel("Quantidade")
    plt.xlabel("Objetivo do empréstimo")
    # Alterar os rótulos do eixo x
    plt.xticks(ticks=range(len(legenda_loan_intent)), labels=list(legenda_loan_intent.values()), rotation=0)
    # Alterar o título da legenda (loan_status)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Adimplente' if label == '0' else 'Inadimplente' for label in labels], title='Situação do Empréstimo')
    #Remove a legenda
    plt.legend([], [], frameon=False)
    plt.show()

    legenda_loan_grade = {'0.0': 'A', '1.0': 'B', '2.0': 'C', '3.0': 'D', '4.0': 'E', '5.0': 'F', '6.0': 'G'}
    plt.figure(figsize=(10,6))
    sns.countplot(data = dados, x='loan_grade', hue = 'loan_status')
    plt.xticks(rotation=90)
    plt.xlabel("Nota de Crédito")
    plt.ylabel("Quantidade")
    # Alterar os rótulos do eixo x
    plt.xticks(ticks=range(len(legenda_loan_grade)), labels=list(legenda_loan_grade.values()), rotation=0)
    # Alterar o título da legenda (loan_status)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Adimplente' if label == '0' else 'Inadimplente' for label in labels], title='Situação do Empréstimo')
    plt.show()

graficosDados()

# Divide os dados em conjuntos de treinamento (80%) e teste (20%).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
report = classification_report(y_test, y_pred) # Relatório com métricas como precision, recall e F1-score.

print(f"Accuracy {accuracy:.2f}")
print(f"Curva AUC-ROC: {auc:.2f}")
print("######################################## Relatório com métricas como precision, recall e F1-score ########################################")
print(report)

print("\n")
print("######################################## Input de Novos Clientes ########################################")
# Lendo o arquivo credit_risk_dataset.csv utilizando a biblioteca pandas
try:
    dados_novos = pd.read_csv('dataset/novos_clientes.csv')
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
dados_novos_imp = imp.transform(dados_novos)
dados_novos_scaled = scaler.transform(dados_novos_imp)
        
# Previsão usando o modelo KNN
previsoes = KNC.predict(dados_novos_scaled)
probabilidades = KNC.predict_proba(dados_novos_scaled)[:, 1]
        
# Exibir resultados
for i, pred in enumerate(previsoes):
    print(f"Cliente {i+1}:")
    print(f"  Previsão: {'Aprovado' if pred == 1 else 'Negado'}")
    print(f"  Probabilidade de aprovação: {int(probabilidades[i] * 100)}%")