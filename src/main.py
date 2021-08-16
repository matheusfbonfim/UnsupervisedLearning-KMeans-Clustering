##########################
# IMPORTANDO BIBLIOTECAS

# Bibliotecas básicas
import numpy as np
import pandas as pd

# Gráficos e tabelas
import matplotlib.pyplot as plt
import matplotlib.colors

# Algoritmo K-means e Elbow
from sklearn.cluster import KMeans

import seaborn as sns

# -----------------------------------------------------------
# -----------------------------------------------------------

def main():
    ##########################
    # PRE PROCESSAMENTO DE DADOS (Importando o csv para o dataframe)

    # Configurando e lendo o arquivo csv
    df = pd.read_csv('../database/alunos_engcomp.csv', delimiter = ';', decimal = ',')

    # Verificando as 5 primeiras linhas do dataframe
    print("="*50)
    print("\tPRE VISUALIZACAO DA BASE DE DADOS")
    print("="*50)
    print(df.head(),'\n')

    # Número de linhas e colunas
    num_linhas = df.shape[0]
    num_colunas = df.shape[1]
    print(f"Numero de linhas: {num_linhas} e colunas: {num_colunas}\n")


    ############################
    # ELIMINANDO DADOS INCONSISTENTES OU COLUNAS QUE NÃO AGREGAM INFORMAÇÃO

    # Elimine as linhas onde pelo menos um elemento está faltando.
    df = df.dropna()

    # Elimine as linhas onde a nota no ENEM é zero
    index_value_error = df[df['ENEM'] == '0'].index
    df = df.drop(index=index_value_error)

    # Elimine a linha onde aparece o #VALUE
    index_value_error = df[df['ENEM'] == '#VALUE!'].index
    df = df.drop(index=index_value_error)

    # Eliminando colunas de coeficiente e periodo
    del df['Coeficiente']
    del df['Período']

    ############################
    # ALTERANDO OS ROTULOS

    # Change textual labels 'M' to 0 and 'F' to 1
    df["Sexo"] = df["Sexo"].apply(lambda sexo: 0 if sexo == 'M' else 1)

    # Change textual labels 'Publica' to 0 and 'Particular' to 1
    df["Escola"] = df["Escola"].apply(lambda escola: 0 if escola == 'Pública' else 1)

    # Alterando o ponto decimal ',' para '.'
    df["ENEM"] = df["ENEM"].str.replace(',','.')

    # Alterando a nota do ENEM de str para float
    df['ENEM'] = pd.to_numeric(df["ENEM"])


    ############################
    ## MÉTODO DO COTOVELO(ELBOW METHOD)
        # Utilize para justificar a quantidade de agrupamentos.

    wcss = [] # Within cluster sum of squares
    for i in range(1, 12):
        # Chama-se o contrutor do KMeans
            # Varia o numero de clusters no loop
            # inicializa com 'k-means++' pois aumenta a probabilidade
            # de escolher centroides longes um do outro
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)

        # Treinando
        kmeans.fit(df[['Escola', 'ENEM']])

        # Guardando no vetor a inercia produzida
        wcss.append(kmeans.inertia_)

    # Plotando o Elbow Method
    plt.plot(range(1,12), wcss)
    plt.title('Metodo do cotovelo')
    plt.xlabel('Numero de Clusters')
    plt.ylabel('WCSS')
    plt.show()


    ############################
    ## KMEANS
    # Com analise do metodo do cotovelo foi escolhido n_clusters = 3

    # Modelo
    kmeans = KMeans(n_clusters = 3, init="k-means++")

    # Treinando o kmeans
    kmeans.fit(df[['Escola', 'ENEM']])

    # Vendo as labels criadas
    print(f"Labels: {kmeans.labels_}\n")
    print(f"Cluster Centers: {kmeans.cluster_centers_}\n")

    ############################
    ## VISUALIZANDO OS CLUTERS

    print("=" * 50)
    print("\t\tVISUALIZANDO OS CLUTERS")
    print("=" * 50)

    for label in np.unique(kmeans.labels_):
        print('LABEL: ', label)
        print(df.iloc[kmeans.labels_ == label])
        print('\n')


    # Incluindo a column clusters
    df['Clusters'] = kmeans.labels_
    print("=" * 50)
    print("\t\tVISUALIZACAO - ADD CLUSTERS")
    print("=" * 50)
    print(df.head(), '\n')


    # Plotando o clustering
    cmap = matplotlib.colors.ListedColormap(['red', 'blue', 'green'], "")
    norm = matplotlib.colors.BoundaryNorm(boundaries=[0, 1, 2, 3], ncolors=3, clip=True)
    sns.scatterplot(x='ENEM', y = 'Escola', hue = 'Clusters', data = df, cmap = cmap, norm = norm)


    plt.show()



# -----------------------------------------------------------
# Executando a Kmeans
main()