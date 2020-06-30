####################################
# Importação dos módulos de função #
####################################

from __future__ import division  # Permite utilizar fucionalidades futuras do python
import math # Funções matemáticas
import operator # Funções intrínsecas de operações matemáticas
import copy # Possibilita a realização de cópias de objetos
import csv # Possibilita a manipulação de dados no formato .csv
import time # Funções de tempo/data
import random # Funções de randomização
from collections import Counter # Cria um dicionário com a frequência dos elementos de um objeto

#####################################################
# Classe csvdata para o armazenamento de dados .csv #
#####################################################
class csvdata():
    def __init__(self, classifier):
        self.rows = []
        self.attributes = []
        self.attribute_types = []
        self.classifier = classifier
        self.class_col_index = None

################################################################
# Classe decisionTreeNode para construção da árvore de decisão #
################################################################
class decisionTreeNode():
    def __init__(self, is_leaf_node, classification, attribute_split_index, attribute_split_value, parent, left_child, right_child, height):
        self.is_leaf_node = True
        self.classification = None
        self.attribute_split = None
        self.attribute_split_index = None
        self.attribute_split_value = None
        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.height = None

###############################
# Pré-processamento dos dados #
###############################
def preprocessing(dataset):
    # Conversão dos atributos numéricos em float. 'True' = Numérico e 'False' = Discreto
    for example in dataset.rows:
        for x in range(len(dataset.rows[0])):
            if dataset.attributes[x] == 'True':
                example[x] = float(example[x])

######################################################
# Construção da árvore de decisão de forma recursiva #
######################################################
def compute_decision_tree(dataset, parent_node, classifier):
    # Primeiro criar um nó da árvore
    node = decisionTreeNode(True, None, None, None, parent_node, None, None, 0)
    # Cálculo da altura da árvore
    if (parent_node == None):
        node.height = 0
    else:
        node.height = node.parent.height + 1
    # Checar se os dados do nó são puros
    ones = count_positives(dataset.rows, dataset.attributes, classifier) # count_positives() irá contar o número de exemplos (rows) com classificação '1'
    if (len(dataset.rows) == ones):
        node.classification = 1
        node.is_leaf_node = True
        return node
    elif (ones == 0):
        node.classification = 0
        node.is_leaf_node = True
        return node
    else:
        node.is_leaf_node = False

    # Definir o melhor atributo para o split
    splitting_attribute = None
    # O ganho de informação fornecido pelo melhor atributo
    maximum_info_gain = 0
    # Limite condicional
    split_val = None
    # O mínimo valor de ganho de informação permitido
    minimum_info_gain = 0.01
    # Cálculo da entropia do dataset
    entropy = calculate_entropy(dataset, classifier)
    for attr_index in range(len(dataset.rows[0])):
        if (dataset.attributes[attr_index] != classifier):
            local_max_gain = 0
            local_split_val = None
            attr_value_list = [example[attr_index] for example in dataset.rows] # Dados que serão splitados
            attr_value_list = list(set(attr_value_list)) # Remove valores duplicados de atributos
            # Caso o atributo for numérico, definir as condições limite para o split
            if (len(attr_value_list) > 100):
                attr_value_list = sorted(attr_value_list)
                total = len(attr_value_list)
                ten_percentile = int(total/10)
                new_list = []
                for x in range(1, 10):
                    new_list.append(attr_value_list[x*ten_percentile])
                attr_value_list = new_list
            # Definição do melhor valor de val
            for val in attr_value_list:
                # Calcular o valor de ganho utilizando este valor de limite
                # Se for maior que local_split_val, salvar este valor na mesma variável
                current_gain = calculate_information_gain(attr_index, dataset, val, entropy)
                if (current_gain > local_max_gain):
                    local_max_gain = current_gain
                    local_split_val = val
            # Definição do melhor atributo
            if (local_max_gain > maximum_info_gain):
                maximum_info_gain = local_max_gain
                split_val = local_split_val
                splitting_attribute = attr_index

    # Classificação do nó para casos quase puros
    if (maximum_info_gain <= minimum_info_gain or node.height > 20):
        node.is_leaf_node = True
        node.classification = classify_leaf(dataset, classifier)
        return node
    
    # Informações do nó (leaf) formado
    node.attribute_split_index = splitting_attribute
    node.attribute_split = dataset.attributes[splitting_attribute]
    node.attribute_split_value = split_val

    # Construção dos ramos após o split
    left_dataset = csvdata(classifier)
    right_dataset = csvdata(classifier)
    left_dataset.attributes = dataset.attributes
    right_dataset.attributes = dataset.attributes
    left_dataset.attribute_types = dataset.attribute_types
    right_dataset.attribute_types = dataset.attribute_types

    # Alocação dos dados para cada ramo criado
    for row in dataset.rows:
        if (splitting_attribute is not None and row[splitting_attribute] >= split_val):
            left_dataset.rows.append(row)
        elif (splitting_attribute is not None and row[splitting_attribute] < split_val):
            right_dataset.rows.append(row)

    # Recursion
    node.left_child = compute_decision_tree(left_dataset, node, classifier)
    node.right_child = compute_decision_tree(right_dataset, node, classifier)

    return node

###########################################
# Função para classificação da folha (nó) #
###########################################
def classify_leaf(dataset, classifier):
    ones = count_positives(dataset.rows, dataset.attributes, classifier)
    total = len(dataset.rows)
    zeroes = total - ones
    if (ones >= zeroes):
        return 1
    else:
        return 0

#############################   
# Avaliação final dos dados #
#############################
def get_classification(example, node, class_col_index):
    if (node.is_leaf_node == True):
        return node.classification
    else:
        if (example[node.attribute_split_index] >= node.attribute_split_value):
            return get_classification(example, node.left_child, class_col_index)
        else:
            return get_classification(example, node.right_child, class_col_index)               

##################################
# Cálculo da entropia do dataset #
##################################
def calculate_entropy(dataset, classifier):

    # Contagem de rows com classificação '1'
    ones = count_positives(dataset.rows, dataset.attributes, classifier)
    # Cálculo do número de rows
    total_rows = len(dataset.rows)
    # A entropia é calculada pela fórmula somatória de -p*log2(p), onde p é a probabilidade de certa classificação
    entropy = 0
    # Probabilidade p de classificação '1' no dataset total
    p = ones/total_rows
    if (p != 0):
        entropy += p*math.log(p, 2)
    # Probabilidade p de classificação '0' no dataset total
    p = (total_rows - ones)/total_rows
    if (p != 0):
        entropy += p*math.log(p, 2)
    entropy = -entropy
    return entropy

##################################
# Cálculo de ganho de informação #
##################################
def calculate_information_gain(attr_index, dataset, val, entropy):
    classifier = dataset.attributes[attr_index]
    attr_entropy = 0
    total_rows = len(dataset.rows)
    #criando dois possíveis ramos da árvore 
    gain_upper_dataset = csvdata(classifier)
    gain_lower_dataset = csvdata(classifier)
    gain_upper_dataset.attributes = dataset.attributes
    gain_lower_dataset.attributes = dataset.attributes
    gain_upper_dataset.attribute_types = dataset.attribute_types
    gain_lower_dataset.attribute_types = dataset.attribute_types
    # split de acordo com val
    for example in dataset.rows:
        if (example[attr_index] >= val):
            gain_upper_dataset.rows.append(example)
        elif (example[attr_index] < val):
            gain_lower_dataset.rows.append(example)
        
    if (len(gain_upper_dataset.rows) == 0 or len(gain_lower_dataset.rows) == 0):
        return -1

    # Cálculo da entropia do atributo utilizado
    attr_entropy += calculate_entropy(gain_upper_dataset, classifier)*len(gain_upper_dataset.rows)/total_rows
    attr_entropy += calculate_entropy(gain_lower_dataset, classifier)*len(gain_lower_dataset.rows)/total_rows

    return entropy - attr_entropy

##########################################
# Contador de rows com classificação '1' #
##########################################
def count_positives(instances, attributes, classifier):
    count = 0
    class_col_index = None
    # Achar o índice do classificador
    for a in range(len(attributes)):
        if attributes[a] == classifier:
            class_col_index = a
        else:
            class_col_index = len(attributes) - 1
    # Contagem de '1's
    for i in instances:
        if i[class_col_index] == '1':
            count += 1
    return count

#######################
# Validação da árvore #
#######################
def validate_tree(node, dataset):
    total = len(dataset.rows)
    correct = 0
    for row in dataset.rows:
        #Validação de exemplo (row)
        correct += validate_row(node, row)
    return correct/total

##############################
# Validação do exemplo (row) #
##############################
# Para achar o melhor score antes de podar a árvore
def validate_row(node, row):
    if (node.is_leaf_node == True):
        projected = node.classification
        actual = int(row[-1])
        if (projected == actual):
            return 1
        else:
            return 0
    value = row[node.attribute_split_index]
    if (value >= node.attribute_split_value):
        return validate_row(node.left_child, row)
    else:
        return validate_row(node.right_child, row)

##########################
# Poda (Prune) da árvore #
##########################
def prune_tree(root, node, validate_set, best_score):
    # Se o nó for uma folha
    if (node.is_leaf_node == True):
        # classification = node.classification
        node.parent.is_leaf_node = True
        node.parent.classification = node.classification
        if (node.height < 20):
            new_score = validate_tree(root, validate_set)
        else:
            new_score = 0
        if (new_score >= best_score):
            return new_score
        else:
            node.parent.is_leaf_node = False
            node.parent.classification = None
            return best_score
    # Se o nó não for uma folha
    else:
        new_score = prune_tree(root, node.left_child, validate_set, best_score)
        if (node.is_leaf_node == True):
            return new_score
        new_score = prune_tree(root, node.right_child, validate_set, new_score)
        if (node.is_leaf_node == True):
            return new_score
        return new_score

##########################################
# Programa principal para rodar a árvore #
##########################################
def run_decision_tree():

    # Dados a serem utilizados
    dataset = csvdata('')
    training_set = csvdata('')
    test_set = csvdata('')

    # Carregar o dados
    f = open('wine_quality_dataset.csv')
    original_file = f.read()
    # Tratar os dados
    rowsplit_data = original_file.splitlines()
    dataset.rows = [rows.split(',') for rows in rowsplit_data]
    dataset.attributes = dataset.rows.pop(0)
    # Printar atributos
    print("Attributes:")
    print(dataset.attributes)

    # Definição dos tipos de atributos (Numérico == 'true' e Nominal == 'false')
    # Para cada caso deve ser alterado
    dataset.attribute_types = ['true', 'true', 'true', 'true', 'true', 'true', 'true', 'true', 'true', 'true', 'true', 'false']

    # Definir a classe
    classifier = dataset.attributes[-1]
    dataset.classifier = classifier

    # Achar o índice da classe
    for a in range(len(dataset.attributes)):
        if (dataset.attributes[a] == dataset.classifier):
            dataset.class_col_index = a
        else:
            dataset.class_col_index = len(dataset.attributes) - 1
    
    # Printar qual é a classe
    print(f'Classifier is {dataset.attributes[dataset.class_col_index]} (Index: {dataset.class_col_index})')

    # Pré-processamento dos dados
    preprocessing(dataset)

    # Dados para treinamento, teste e validação
    training_set = copy.deepcopy(dataset)
    training_set.rows = []
    test_set = copy.deepcopy(dataset)
    test_set.rows = []
    validate_set = copy.deepcopy(dataset)
    validate_set.rows = []

    # Caso realizar poda (prunning), ativar código abaixo
    # Criar o dataset para validação para pós poda (post pruning)
    # dataset.rows = [x for i, x in enumerate(dataset.rows) if i % 10 != 9]
    #validate_set.rows = [x for i, x in enumerate(dataset.rows) if i % 10 == 9]

    # Número de runs a serem realizadas
    K = 10
    # Armazenar a precisão (accuracy) das 10 runs
    accuracy = []
    start = time.clock()

    for k in range(K):
        print('Doing fold', k)
        # Parece estar criando novos datasets para treino e teste
        training_set.rows = [x for i, x in enumerate(dataset.rows) if i % K != k]
        test_set.rows = [x for i, x in enumerate(dataset.rows) if i % K == k]
        # Printar quantos exemplos para treino e teste são criados
        print("Number of training records: %d" % len(training_set.rows))
        print("Number of test records: %d" % len(test_set.rows))

        # Construção da árvore
        root = compute_decision_tree(training_set, None, classifier)

        # Teste da árvore
        # Classificar os dados de teste usando a árvore construída
        results = []
        for instance in test_set.rows:
            result = get_classification(instance, root, test_set.class_col_index)
            results.append(str(result) == str(instance[-1]))

        # Cálculo da precisão (Accuracy)
        acc = float(results.count(True))/float(len(results))
        print("Accuracy: %.4f" % acc)

        # Se desejar, ativar o código de poda abaixo.
        # best_score = validate_tree(root, validate_set)
        # post_prune_accuracy = 100*prune_tree(root, root, validate_set, best_score)
        # print ("Post-pruning score on validation set: " + str(post_prune_accuracy) + "%")

        accuracy.append(acc)
        del root
    
    mean_accuracy = math.fsum(accuracy)/K
    print('Final results:')
    print("Accuracy  %f " % (mean_accuracy))
    print("Took %f secs" % (time.clock() - start))

    # Cria um arquivo de resultados
    f = open("result.txt", "w")
    f.write("accuracy: %.4f" % mean_accuracy)
    f.close()

###################
# Rodar algoritmo #
###################

if __name__ == "__main__":
    run_decision_tree()

    



        
        


    



