# Dataset: http://archive.ics.uci.edu/ml/datasets/banknote+authentication

# Pacotes a serem utilizados no código
from random import seed
from random import randrange
from csv import reader

# Função para importação dos dados
def load_csv(filename):
    file = open(filename)
    lines = reader(file)
    dataset = list(lines)
    return dataset

# Conversão de coluna string para float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Divisão dos dados em k partições para cross validation
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Função para cálculo de porcentagem de precisão
def accuracy_metric(actual, predicted):
    correct = 0 
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) *100

# Função para avaliar o algoritmo utilizando cross validation
def evaluate_algorithm(dataset, algorithm, n_folds, *args): # *args é usado caso possa adicionar mais parâmetros na função
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


## Cálculo do coeficiente Gini
# Função para cálculo de coeficiente Gini para um conjunto de sub-nós (grupos) e valores de classe conhecidos
def gini_index(groups, classes):
    # Contagem do total de amostras no ponto de divisão (split point)
    n_instances = float(sum([len(group) for group in groups]))
    # Soma dos coeficientes Gini ponderados para cada grupo
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # Condicional de segurança contra divisão por zero
        if size == 0:
            continue
        score = 0 
        # Cálculo da frequência de cada valor de classe (p) e do score p^2 
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size # .count() retorna o número de ocorrências de class_val 
            score += p*p # soma dos quadrados das frequências de cada classe no grupo
        # Ponderar o score do grupo pelo seu tamanho relativo
        gini += (1.0 - score)*(size / n_instances) # Quantifica o ganho gerado pela divisão
    return gini 

## Divisão (particionamento) dos dados
''' A divisão dos dados para CART significa os dados em duas listas de linhas dado o índice do atributo e 
    o valor condicional de divisão para este atributo. '''
# Função para o split dos dados
def test_split(index, value, dataset):
    left, right = list(), list() # Lista para armazenamento dos dados dos sub-nós
    for row in dataset:
        if row[index] < value: # Critério de divisão
            left.append(row)
        else:
            right.append(row)
    return left, right # Retorna as listas de dados para cada sub-nó gerado

# Avaliar de todas as divisões possíveis
''' Com a função para o cálculo do coeficiente the Gini e a função de split test agora temos o necessário para avaliar
    as divisões. Dado a base de dados, devemos checar todo valor em cada atributo como um candidato de divisão, avaliar 
    o custo da divisão e determinar a melhor divisão possível que se pode realizar. Uma vez que a melhor divisão é 
    determinada, podemos utilizá-la na árvore de decisão. '''
# Função para determinação da melhor divisão
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset)) # set() não permite valores repetidos
    b_index, b_value, b_score, b_groups = 9999, 9999, 9999, None
    for index in range(len(dataset[0])-1): # Não contar a coluna de classes
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            #print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
            if gini < b_score: # Quanto menor o gini melhor
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

## Contrução da árvore de decisão
''' Para gerar o nó raiz da árvore basta aplicarmos get_split() para o dataset original. Já para gerar os sub-nós 
    subsequentes é necessário aplicar get_split() de forma recursiva para cada sub-nó gerado. '''
# Nós terminais. É necessário determinar o momento de interromper o crescimento da árvore.
''' Para isso iremos utilizar o Maximum Tree Depth (máximo número de nós permitidos) na árvore e o Minimum Node
    Records (número mínimo de dados presentes em um nó). Uma outra condição que se deve levar em conta também 
    é o caso em que todos os dados em um nó pertencem à mesma classe e nó não pode ser mais dividido. Quando 
    interrompemos o crescimento da árvore em um dado nó, esse nó é chamado de nó terminal e ele é utilizado para 
    realizar a predição final. Isso é feito de forma a escolher a classe mais frequente no grupo de dados no nó. '''
# Função para seleção de uma classe em um nó terminal. Retorna a classe mais comum no nó
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count) # Retorna o item mais comum na lista

# Função a realização do split recursivo dos nós. Cria sub-nós filhos de um nó ou declara o nó como terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # Checa se algum dos nós criados está vazio
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # Checa se max_depth foi atingida
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # Split do nó esquerdo (left child)
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # Split do nó direito (right child)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)
    
# Função para construção da árvore
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Função para print da árvore
def print_tree(node, depth=0):
    if isinstance(node, dict): # Checa se node é um dicionário
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

## Predição de classificação
# Função que faz predição de um dado com a árvore de decisão gerada
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Função contendo o algoritmo CART 
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions

## ___Main()___
# Teste do algoritmo CART para dados de autentiação de banco
seed(1)
# Importação e preparação dos dados
filename = 'data_banknote_authentication_dataset.csv'
dataset = load_csv(filename)
# Conversão de atributos string para float
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# Avaliação da performance do algoritmo
n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))



