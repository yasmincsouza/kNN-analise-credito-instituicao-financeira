# KNeighborsClassifier para Análise de Concessão de Crédito em Instituições Financeiras
# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier

# Lendo o arquivo credit_risk_dataset.csv utilizando a biblioteca pandas
try:
    # dados = pd.read_csv('analise_credito.csv')
    dados = pd.read_csv('dados/credit_risk_dataset.csv')
    print("Arquivo CSV carregado com sucesso!")
except FileNotFoundError:
    print("Arquivo CSV não encontrado!")

# Verificar como estão os dados das primeiras entradas de cada coluna
print("#### Dados das primeiras entradas de cada coluna ####")
print(dados.head())

# Resumo visão geral dos dados e compreender suas características.
print("####################")
print(dados.info())

## Algumas colunas contêm dados categóricos. Para treinar ainda mais o modelo, precisamos nos livrar delas.
print("##########Propriedade da casa##########")
print(dados['person_home_ownership'].value_counts())
print("##########Intenção de empréstimo##########")
print(dados['loan_intent'].value_counts())
print("##########Grau de Empréstimo##########")
print(dados['loan_grade'].value_counts())
print("##########Inadimplência histórica##########")
print(dados['cb_person_default_on_file'].value_counts())

def visualizaoDados():
    plt.figure(figsize=(10,6))
    sns.barplot(x=dados['person_age'].unique(), y=dados['person_age'].value_counts())
    plt.xticks(rotation=90)
    plt.title('Age')
    plt.show()

    plt.figure(figsize=(10,6))
    sns.barplot(x=dados['person_home_ownership'].unique(), y=dados['person_home_ownership'].value_counts())
    plt.title('Availability of housing')
    plt.show()

    plt.figure(figsize=(10,6))
    sns.barplot(x=dados['loan_grade'].unique(), y=dados['loan_grade'].value_counts())
    plt.title('Loan grade')
    plt.show()

    plt.figure(figsize=(10,6))
    sns.barplot(x=dados["person_income"],y=dados["loan_intent"])
    plt.title("Relationship between loan goals and person income")
    plt.show()

    plt.figure(figsize=(10,6))
    sns.barplot(x=dados["loan_amnt"],y=dados["loan_intent"])
    plt.title("Relationship between loan goals and loan amount")
    plt.show()

    plt.figure(figsize=(10,6))
    sns.countplot(data = dados, x = 'loan_grade', hue = 'loan_status')
    plt.title("Relationship between loan grade and loan status")
    plt.show()

    plt.figure(figsize=(10,6))
    sns.countplot(data = dados, x = 'person_home_ownership', hue = 'loan_status')
    plt.title("Relationship between home ownership and loan status")
    plt.show()

    plt.figure(figsize=(10,6))
    sns.countplot(data = dados, x = 'loan_intent', hue = 'loan_status')
    plt.xticks(rotation=90)
    plt.title("Relationship between loan goals and loan status")
    plt.show()

    plt.figure(figsize=(10,6))
    plt.pie(dados['loan_status'].value_counts(), labels=['N', 'Y'], autopct='%1.1f%%')
    plt.title("Loan status")
    plt.show()

print("####################")
print(dados.isna().sum())

print("####################")
print(dados.duplicated().sum())

dados = dados.drop(dados[dados['person_age'] > 100].index)

ord_enc = OrdinalEncoder()

dados['person_home_ownership'] = ord_enc.fit_transform(dados[['person_home_ownership']])
dados['loan_intent'] = ord_enc.fit_transform(dados[['loan_intent']])
dados['loan_grade'] = ord_enc.fit_transform(dados[['loan_grade']])
dados['cb_person_default_on_file'] = ord_enc.fit_transform(dados[['cb_person_default_on_file']])
dados.drop_duplicates(inplace=True)

df1 = dados.copy()
df1['person_emp_length'] = df1['person_emp_length'].fillna(df1['person_emp_length'].mean())

fig, axes = plt.subplots(2, 2,figsize=(16,10))

axes[0,0].set_title('Employment length boxplot')
axes[0,1].set_title('Employment length distribution plot')
axes[1,0].set_title('Employment length boxplot after imputation')
axes[1,1].set_title('Employment length distribution plot after imputation')

sns.boxplot(ax=axes[0,0], data=dados['person_emp_length'], orient='h')
sns.histplot(ax=axes[0,1], data=dados['person_emp_length'])
sns.boxplot(ax=axes[1,0], data=df1['person_emp_length'], orient='h')
sns.histplot(ax=axes[1,1], data=df1['person_emp_length'])

# plt.show()

dados = df1

X = dados.drop(columns='loan_status')
y = dados['loan_status']

# Método de imputação que usa o algoritmo K-Nearest Neighbors para preencher valores ausentes
# 5 vizinhos mais próximos
imp = KNNImputer(n_neighbors=5)
X_imp = imp.fit_transform(X)

X1 = pd.DataFrame(X_imp, columns=X.columns)
X1.shape
X.shape

X = X1
print(X.isna().sum())
print(X.describe())