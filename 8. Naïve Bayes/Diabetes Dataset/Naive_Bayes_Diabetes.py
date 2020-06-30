import csv
import random
import math
# dataset: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
# Informação sobre o dataset: https://www.andreagrandi.it/2018/04/14/machine-learning-pima-indians-diabetes/

## Preparação dos dados
# Função para importação dos dados
def loadCSV(filename):
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

# Função para os split dos dados em dados de treinamento e dados de teste
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset)*splitRatio)
    trainSet = [] # Lista com dados de treinamento
    testSet = list(dataset) # Lista com dados de teste
    while len(trainSet) < trainSize:
        index = random.randrange(len(testSet)) # Gera um valor de índice randomicamente
        trainSet.append(testSet.pop(index)) # Remove um dado da lista de teste e o adiciona para a lista de treinamento
    return [trainSet, testSet]

## Sumarizar as informações dos dados para realizar predições utilizando o algoritmo Naive Bayes
''' A sumarização das informações dos dados consiste no cálculo, por classe, da média e do desvio padrão para cada 
    atributo. Portanto, como a base de dados analisada possui 2 valores de classe e 7 atributos, será necessário
    o cálculo da média e do desvio padrão dos 7 atributos para cada uma das 2 classes, ou seja, serão necessários 
    14 sumários dos atributos (2 para cada atributo).''' 

# Função para determinar os valores de classe presente na base de dados e separar os dados de acordo com as classes
def separateByClass(dataset):
    classes = {}
    for data in dataset:
        if data[-1] not in classes:
            classes[data[-1]] = []
        classes[data[-1]].append(data)
    return classes

# Função para cálculo da média dos valores dos atributos
def mean(attribute_values):
    return sum(attribute_values) / float(len(attribute_values))

# Função para cálculo do desvio padrão 
def stdev(attribute_values):
    avg = mean(attribute_values)
    variance = sum([pow(x - avg, 2) for x in attribute_values]) / float(len(attribute_values)-1)
    return math.sqrt(variance)

# Função de sumarização dos valores dos atributos
def summarize(dataset):
    summaries = [(mean(attribute_values), stdev(attribute_values)) for attribute_values in zip(*dataset)] 
    # zip(*dataset) agrupa os valores das colunas em tuplas
    del summaries[-1] # Não é necessário sumarizar os valores de classe
    return summaries # Retorna um lista de tuplas com valores de média e desvio padrão para cada atributo

# Função para sumarização por classe
def summarizeByClass(dataset):
    classes = separateByClass(dataset)
    summariesByClass = {}
    for classValue, instances in classes.items():
        summariesByClass[classValue] = summarize(instances)
    return summariesByClass # Retorna os valores de sumarização para cada classe

# Função para sumarização dos dados em geral (média e desvio padrão dos dados originais sem levar em conta a classe)
def generalSummarize(dataset):
    generalSummaries = summarize(dataset)
    return generalSummaries

## Realização de predições
''' Realizar predições consiste no cálculo da probabilidade de um certo dado pertencer a cada classe. A classe que 
    apresentar maior probabilidade será escolhida para a predição. Podemos usar a função Gaussiana para estimar a 
    probabilidade de um dado valor de atributo, dado que são conhecidos sua média e desvio padrão para o atributo.
    for the attribute estimated from the training data. Dado que os sumários dos atributos foram gerados para cada 
    atributo e classe, o resultado é uma probabilidade condicional de um dado atributo dado um valor de classe.'''

# Função para cálculo de probabilidade
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2) / (2*math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi)*stdev))*exponent

# Função para cálculo de probabilidade de uma classe em relação à todos os dados (Prior Probability)
def classesProbabilities(dataset):
    classesProb = {}
    for data in dataset:
        if data[-1] not in classesProb:
            classesProb[data[-1]] = 1 / float(len(dataset))
        else:
            classesProb[data[-1]] += 1 / float(len(dataset))
    return classesProb

# Função para o cálculo da probabilidade de um dado pertencer a uma classe. 
# Basta multiplicar as probabilidades de cada atributo pertencer à classe.
def calculateClassProbabilities(summariesByClass, generalSummaries, inputVector, classesProb):
    probabilities = {} # P(Class|X) = P(X|Class)*P(Class) / P(X)
    # Cálculo de P(X|Class)*P(Class)
    for classValue, classSummaries in summariesByClass.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
        probabilities[classValue] *= classesProb[classValue] # P(Class)
    # Cálculo de P(X)
    normalizingConstantProb = 1
    for j in range(len(generalSummaries)):
        g_mean, g_stdev = generalSummaries[j]
        x = inputVector[j]
        normalizingConstantProb *= calculateProbability(x, g_mean, g_stdev)
    # Cálculo de P(X|Class)*P(Class)/P(X)
    for classValue, value in probabilities.items():
        probabilities[classValue] = value / normalizingConstantProb
    return probabilities

# Função para realizar a predição. Escolhe a classe com maior probabilidade.
def predict(summariesByClass, generalSummaries, inputVector, classesProb):
    probabilities = calculateClassProbabilities(summariesByClass, generalSummaries, inputVector, classesProb)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

## Predições para os dados de teste
# Função que retorna uma lista de predições para os dados de teste
def getPredictions(summariesByClass, generalSummaries, testSet, classesProb):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summariesByClass, generalSummaries, testSet[i], classesProb)
        predictions.append(result)
    return predictions

## Cálculo da precisão do modelo. Compara as predições com as classes dos dados de teste
# Função para cálculo de precisão
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

## Main()
def main():
    splitRatio = 0.67
    dataset = loadCSV('pima_indians_diabetes_dataset.csv')
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print(f'Split {len(dataset)} rows into train={len(trainingSet)} and test={len(testSet)} rows')
    # Preparação do modelo
    summariesByClass = summarizeByClass(trainingSet)
    generalSummaries = generalSummarize(trainingSet)
    classesProb = classesProbabilities(trainingSet)
    # Teste do modelo
    predictions = getPredictions(summariesByClass, generalSummaries, testSet, classesProb)
    accuracy = getAccuracy(testSet, predictions)
    print(f'Accuracy: {accuracy}%')
main()
