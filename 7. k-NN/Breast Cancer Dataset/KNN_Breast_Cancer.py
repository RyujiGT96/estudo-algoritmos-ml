# Importação dos pacotes a serem utilizados
import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

#Dataset: https://archive.ics.uci.edu/ml/datasets/breast+cancer+ wisconsin+(original)

# Função para implementação do algoritmo
def k_nearest_neighbors(data, predict, k=3):
    # Wraning caso o valor de k não seja coerente
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting objects!')
    distances = [] # Lista para armazenar valores de distância entre o dado de predição e todos os dados de treino
    # Cálculo das distâncias Euclideanas entre o dado de predição e todos os dados de treino
    for group in data:
        for feature in data[group]:
            euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict)) # Cálculo de distância Euclideana
            distances.append([euclidean_distance, group])
    # Computação dos votos dos k dados mais próximos do dado de predição
    votes = [object[1] for object in sorted(distances)[:k]]
    # Classe resultante da votação
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

# Importação dos dados
df = pd.read_csv('breast_cancer_wisconsin_dataset.csv')
df.replace('?', -99999, inplace=True) # Torna os dados com informações desconhecida em outliers
df.drop(['id'], axis=1, inplace=True) # Remove a coluna de id
full_data = df.astype(float).values.tolist() # Converte a tabela panda para uma lista de listas
# .astype converte os dados para float 
# .values converte os valores para uma numpy array
# .tolist() retorna um lista com os valores

# Embaralha os dados
random.shuffle(full_data)

# Split de dados de treino e dados de teste
test_size = 0.2 # 20% dos dados será utlizado para teste
train_set = {2:[], 4:[]} # Dados de treino
test_set = {2:[], 4:[]} # Dados de teste
train_data = full_data[:-int(test_size*len(full_data))] # Todos os dados menos os últimos 20% 
test_data = full_data[-int(test_size*len(full_data)):] # Os últimos 20% dos dados

# Alocando os valores de atributos e classes na base de dados de treino
for object in train_data:
        train_set[object[-1]].append(object[:-1])
# Alocando os valores de atributos e classes na base de dados de tetse
for object in test_data:
        test_set[object[-1]].append(object[:-1])        

# Variáveis de contagem para cálculo da precisão
correct = 0
total = 0
# Cálculo da precisão
for group in test_set:
        for data in test_set[group]:
                vote = k_nearest_neighbors(train_set, data, k=5)
                if vote == group:
                        correct += 1
                total += 1
print('Accuracy:', correct/total*100, '%')