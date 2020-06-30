import os
import numpy as np
import sklearn.datasets

from support_vector_clustering import SupportVectorClustering

if __name__ == '__main__':

    # importação dos dados
    f = open('caldeira_dataset.csv')
    data = f.read()
    f.close()
    # conversão dos dados para matriz
    lines = data.split('\n')
    num_rows = len(lines)
    num_cols = len(lines[0].split(';'))
    float_data = np.zeros((num_rows, num_cols-2))
    labels = np.zeros(len(float_data))
    for i, line in enumerate(lines):
        values = [float(value) for value in line.split(';')[1:num_cols-1]]
        float_data[i,:] = values
        labels[i] = float(line[-1])

    # dados de treinamento
    normal_data = float_data[:100]
    fault_data = float_data[2882:2982]
    training_data = np.concatenate((normal_data, fault_data), axis = 0)

    # rótulos dos dados de treinamento
    normal_labels = labels[:100]
    fault_labels = labels[2882:2982]
    training_labels = np.concatenate((normal_labels, fault_labels), axis = 0)
    
    # dados de teste
    test_normal_data = float_data[1440:1540]
    test_fault_data = float_data[4322:4422]
    test_data = np.concatenate((test_normal_data, test_fault_data), axis = 0)

    # rótulos dos dados de teste
    test_normal_labels = labels[1440:1540]
    test_fault_labels = labels[4322:4422]
    test_data_labels = np.concatenate((test_normal_labels, test_fault_labels), axis = 0)
    print(test_data_labels)

    # normalização dos dados 
    mean = training_data.mean(axis=0)
    std = training_data.std(axis=0)
    training_data -= mean
    training_data /= std
    test_data -= mean
    test_data /= std

    # algoritmo SVC
    svc = SupportVectorClustering()
    svc.dataset(training_data, training_labels) # database
    svc.parameters(p=0.005, q=0.24) # parâmetros # p = 0.005 q = 0.24 funcionou
    svc.kernel_matrix() # matriz kernel
    svc.find_beta() # solução problema de otimização
    svc.cluster() # matriz de adjacência
    svc.return_clusters() # define clusters
    svc.results()
    svc.clusters_centroids()
    svc.test_clustering(test_data, test_data_labels)
    
  