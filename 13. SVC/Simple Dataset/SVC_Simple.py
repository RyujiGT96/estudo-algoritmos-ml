import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import time
plt.style.use('seaborn-whitegrid')

from support_vector_clustering import SupportVectorClustering
# importação dos dados
def define_data(REBUILD_DATA = False, N_SAMPLES = 50):
    if REBUILD_DATA == True:
        ms = sklearn.datasets.make_moons(n_samples=N_SAMPLES,noise=0.1)[0]
        np.save('simple_dataset.npy', ms)
    X = np.load('simple_dataset.npy')
    return X

if __name__ == '__main__':
    # define database
    X = define_data(REBUILD_DATA = False, N_SAMPLES=50)
    # iniciação do algoritmo
    start_time = time.time()
    svc = SupportVectorClustering()
    svc.dataset(X) # database
    svc.parameters(p=0.002, q=6.5) # define parâmetros
    svc.kernel_matrix() # cálculo matriz kernel
    svc.find_beta() # solução problema de otimização
    svc.cluster() # cálculo matriz adjacência
    svc.return_clusters() # define clusters
    print('\n')
    print(f'Processing Time: {time.time() - start_time} seconds')
    svc.plot_clusters() # plot

  