import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from tqdm import tqdm
plt.style.use('seaborn-whitegrid')

class SupportVectorClustering():

    def __init__(self):
        pass

    def dataset(self, xs, xs_labels):
        self.xs = xs # dataset
        self.xs_labels = xs_labels
        self.N = len(xs) # number de instâncias

    def parameters(self, p=0.1, q=1):
        self.q = q # parâmetro kernel width 
        self.p = p # fração de bounded support vectors (BSVs) 
        self.C = 1/(self.N*p) # constante de penalização (1/C >= 1)
    
    def kernel(self, x1, x2):
        return np.exp(-self.q*np.sum((x1-x2)**2, axis=-1)) # gaussian kernel 

    def kernel_matrix(self):
        self.km = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.km[i,j] = self.kernel(self.xs[i], self.xs[j])

    # método de otimização
    def find_beta(self): 
        beta = cvx.Variable(self.N) # vetor de N dimensões
        objective = cvx.Maximize(cvx.sum(beta) - cvx.quad_form(beta, self.km))  # função objetivo 
        constraints = [0 <= beta, beta <= self.C, cvx.sum(beta) == 1] # restrições 
        prob = cvx.Problem(objective, constraints) # definição do problema de otimização
        prob.solve() # solução do problema de otimização
        self.beta = beta.value # valores ótimos das variáveis beta 

    # cálculo do raio da hiperesfera
    def r_func(self, x):
        return self.kernel(x, x) - 2*np.sum([self.beta[i]*self.kernel(self.xs[i],x) for i in range(self.N)]) + self.beta.T@self.km@self.beta
        # python > 3.5 @ matrix multiplication

    # amostragem de segmentos entre dois pontos
    def sample_segment(self,x1,x2,r,n=10):
        adj = True
        for i in range(n):
            x = x1 + (x2-x1)*i/(n+1) 
            if self.r_func(x) > r:
                adj = False
                return adj
        return adj
    
    # definição da matriz de adjacência
    def cluster(self):
        print('Calculating adjacency matrix... \n')
        svs_tmp = np.array(self.beta < self.C)*np.array(self.beta > 10**-8) # svs: 0 < beta < C
        self.svs = np.where(svs_tmp == True)[0] # índice support vectors 
        bsvs_tmp = np.array(self.beta >= self.C) # bsvs: beta == C ??
        self.bsvs = np.where(bsvs_tmp == True)[0] # índice bounded support vectors
        self.r = np.mean([self.r_func(self.xs[i]) for i in self.svs[:5]]) # why 5??
        self.adj = np.zeros((self.N, self.N)) # matriz adjacência
        # checar adjacência entre pontos 
        for i in tqdm(range(self.N)):
            if i not in self.bsvs:
                for j in range(i, self.N):
                    if j not in self.bsvs:
                        self.adj[i,j] = self.adj[j,i] = self.sample_segment(self.xs[i],self.xs[j],self.r)
    
    # definição dos clusters
    def return_clusters(self):
        ids = list(range(self.N))
        self.clusters = {}
        num_clusters = 0
        while ids:
            num_clusters += 1
            self.clusters[num_clusters] = []
            curr_id = ids.pop(0)
            queue = [curr_id]
            while queue:
                cid = queue.pop(0)
                for i in ids:
                    if self.adj[i,cid]:
                        queue.append(i)
                        ids.remove(i)
                self.clusters[num_clusters].append(cid)
        print('\n')
        print(f'The number of clusters is {num_clusters}')

    def results(self):
        clusters_results = np.zeros((len(self.clusters.keys()), 2))
        for i in self.clusters.keys():
            normal = 0
            fault = 0
            for j in self.clusters[i]:
                if self.xs_labels[j] == 0:
                    normal += 1
                else:
                    fault += 1   
            clusters_results[i-1][0] = normal
            clusters_results[i-1][1] = fault
        print([0, 1])
        print(clusters_results)
        print(len(clusters_results))
    
    # centróides dos clusters
    def clusters_centroids(self):
        self.centroids = dict()
        for key, values in self.clusters.items(): 
            sum_cluster_data = np.zeros(len(self.xs[0]))
            for j in values:
                sum_cluster_data += self.xs[j]
            centroid = sum_cluster_data / len(values)
            self.centroids[key] = centroid
        print(self.centroids)

    # teste do modelo
    def test_clustering(self, xs_test, xs_labels_test):
        self.xs_test = xs_test
        self.xs_labels_test = xs_labels_test
        clusters_similarity = np.zeros((len(xs_test),len(self.centroids.keys())))
        for i, x in enumerate(xs_test):
            for key, centroid in self.centroids.items():
                clusters_similarity[i,key-1] = np.linalg.norm(x-centroid, 2)
        cluster_assignment = np.argmin(clusters_similarity, axis = 1)
        print(cluster_assignment)
        
        if len(self.centroids.keys()) == 2:
            false_alarm = 0
            warn_miss = 0
            for i in range(len(self.xs_labels_test)):
                if (cluster_assignment[i] != self.xs_labels_test[i]):
                    if ((cluster_assignment[i] == 1) and (self.xs_labels_test[i] == 0)):
                        false_alarm += 1
                    else:
                        warn_miss += 1
            print(f'False Alarm {false_alarm*100/len(self.xs_labels_test)}%')
            print(f'Missed Warning {warn_miss*100/len(self.xs_labels_test)}%')
            print(f'Right Diagnostic {(len(self.xs_labels_test) - false_alarm - warn_miss)*100/len(self.xs_labels_test)}%')
        else:
            error = 0
            for i in range(len(self.xs_labels_test)):
                if cluster_assignment[i] != self.xs_labels_test[i]:
                    error += 1
            print(f'Right Diagnostic {(len(self.xs_labels_test) - error)*100/len(self.xs_labels_test)}%')
            print(f'Wrong Diagnostic {(error)*100/len(self.xs_labels_test)}%')

    def plot_clusters(self):
        colors = {1:'r',2:'b',3:'g',4:'c',5:'m',6:'y',7:'k',8:'b',9:'c'}
        for num_cluster, samples in self.clusters.items():
            cluster = np.empty((len(samples), 2))
            for idx, sample in enumerate(samples):
                cluster[idx] = self.xs[sample]
            plt.scatter(cluster[:,0], cluster[:,1], c = colors[num_cluster], label = num_cluster)
        plt.legend()
        plt.show()






               

    

    
   