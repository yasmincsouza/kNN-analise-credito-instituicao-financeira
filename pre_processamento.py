# KNeighborsClassifier para Análise de Concessão de Crédito em Instituições Financeiras
# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

# Lendo o arquivo credit_risk_dataset.csv utilizando a biblioteca pandas
try:
    # dados = pd.read_csv('analise_credito.csv')
    dados = pd.read_csv('dataset/credit_risk_dataset.csv')
    print("Arquivo CSV carregado com sucesso!")
except FileNotFoundError:
    print("Arquivo CSV não encontrado!")

# Verificar como estão os dados das primeiras entradas de cada coluna
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

# Exibe estatísticas descritivas (média, desvio padrão, valores mínimos e máximos)
print("\n")
print(X.isna().sum())
print("\n")
print(X.describe())
print("\n")

def graficosDados():
    plt.figure(figsize=(10,6))
    sns.barplot(x=dados['person_age'].unique(), y=dados['person_age'].value_counts())
    plt.xticks(rotation=90)
    plt.title('Idade')
    plt.show()

    legenda_loan_grade = {'0.0': 'A', '1.0': 'B', '2.0': 'C', '3.0': 'D', '4.0': 'E', '5.0': 'F', '6.0': 'G'}
    plt.figure(figsize=(10,6))
    sns.barplot(x=dados['loan_grade'].unique(), y=dados['loan_grade'].value_counts())
    plt.title('Grau de Empréstimo')
    # Alterar os rótulos do eixo x
    plt.xticks(ticks=range(len(legenda_loan_grade)), labels=list(legenda_loan_grade.values()))
    plt.show()

    legenda_loan_intent = {'0.0': 'Pessoal', '1.0': 'Educação', '2.0': 'Médico', '3.0': 'Empreendimento', '4.0': 'Reforma Casa', '5.0': 'Consolidação Dívidas'}
    plt.figure(figsize=(10,6))
    sns.countplot(data = dados, x = 'loan_intent', hue = 'loan_status')
    plt.xticks(rotation=90)
    plt.title("Relação entre objetivos de empréstimo e status de empréstimo")
    plt.xlabel("Objetivos de empréstimo (loan_intent)")
    # Alterar os rótulos do eixo x
    plt.xticks(ticks=range(len(legenda_loan_intent)), labels=list(legenda_loan_intent.values()), rotation=0)
    # Alterar o título da legenda (loan_status)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Adimplente' if label == '0' else 'Inadimplente' for label in labels], title='Situação do Empréstimo')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.pie(dados['loan_status'].value_counts(), labels=['Negado', 'Aprovado'], autopct='%1.1f%%')
    plt.title("Proporção de empréstimos aprovados e negados")
    plt.show()

    # legenda_loan_grade = {'0.0': 'A', '1.0': 'B', '2.0': 'C', '3.0': 'D', '4.0': 'E', '5.0': 'F', '6.0': 'G'}
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='loan_grade', y='loan_percent_income', hue='loan_status', data=dados, palette=['#4C72B0', '#C44E52'])
    plt.title("Distribuição do Percentual da Renda por Nota de Crédito e Status do Empréstimo")
    plt.xlabel("Grau de Empréstimo (loan_grade)")
    plt.ylabel("Percentual da Renda (loan_percent_income)")
    # Alterar os rótulos do eixo x
    plt.xticks(ticks=range(len(legenda_loan_grade)), labels=list(legenda_loan_grade.values()))
    # Alterar o título da legenda (loan_status)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Adimplente' if label == '0' else 'Inadimplente' for label in labels], title='Situação do Empréstimo')
    plt.show()

graficosDados()