import math
import random
import copy
import matplotlib.pyplot as plt

# Função do algoritmo K-Means
# Retorna os clusters obtidos
def k_means(points, k):
    if len(points) < k:
        return -1
    centroids = init_centroids(points, k)
    #print(centroids)
    stop = False
    while stop == False:
        clusters = assign_points(points, k, centroids)
        #print(clusters)
        old_centroids = copy.deepcopy(centroids)
        centroids = update_centroids(clusters)
        #print(centroids)
        stop = compare_centroids(centroids, old_centroids, 0.01)
    return clusters

# Função para a escolha de k cluster centroids iniciais. 
# Retorna uma matriz contendo os pontos escolhidos como centróides
def init_centroids(points, k):
    aux_points = copy.deepcopy(points)
    centroids = [0]*k
    centroid_0 = random.choice(aux_points)
    centroids[0] = centroid_0
    aux_points.remove(centroid_0)
    for i in range (1, k):
        dist = []
        for j in range(0, len(aux_points)):
                dist.append(math.sqrt(math.pow(aux_points[j][0] - centroid_0[0], 2) + math.pow(aux_points[j][1] - centroid_0[1], 2)))
        max_dist = None
        new_centroid = None
        for index, value in enumerate(dist):
            if max_dist == None:
                max_dist = value
                new_centroid = aux_points[index]
            else:
                if value > max_dist:
                    max_dist = value
                    new_centroid = aux_points[index]
        centroids[i] = new_centroid
        aux_points.remove(new_centroid)
    return centroids

# Função para atribuição dos pontos para os centróides mais próximos 
# Retorna um dicionário  com os clusters
def assign_points(points, k, centroids):
    clusters = {}
    for point in points:
        dist = []
        for centroid in centroids:
            dist.append(math.sqrt(math.pow(point[0] - centroid[0], 2) + math.pow(point[1] - centroid[1], 2)))
        index = 0
        min_dist = dist[0]
        for i, d in enumerate(dist):
            if d < min_dist:
                min_dist = d
                index = i
        clusters.setdefault(index,[]).append(point)
    return clusters

# Função para a alocação dos centroids dos clusters para as médias dos seus pontos
# Retorna uma matriz com os novos centroids
def update_centroids(clusters):
    means = [0]*len(clusters.keys())
    for key, cluster in clusters.items():
        mean_point = [0,0]
        counter = 0

        for point in cluster:
            mean_point[0] += point[0]
            mean_point[1] += point[1]
            counter += 1
        mean_point[0] = mean_point[0]/counter
        mean_point[1] = mean_point[1]/counter
        means[key] = mean_point
    return means

# Função para verificar convergência do algoritmo
# Retorna um Boolenano
def compare_centroids(centroids, old_centroids, threshold):
    for i in range(len(centroids)):
        new = centroids[i]
        old = old_centroids[i]
        if math.sqrt(math.pow(new[0] - old[0], 2) + math.pow(new[1] - old[1], 2)) > threshold:
            return False
    return True

# Função para print dos clusters
def print_clusters(clusters):
    for key in range(0, len(clusters.keys())):
        print(f'Points in cluster #{key + 1}')
        for point in clusters.get(key):
            print(f'Point ({point[0]},{point[1]})')

    
# Função para plotar os clusters em um gráfico
def plot_clusters(clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ['o', 'd', 'x', 'h', 'H', 7, 4, 5, 6, '8', 'p', ',', '+', '.', 's', '*', 3, 0, 1, 2]
    colors = ['r', 'k', 'b', 'c', 'm', 'g', 'y', [0,1,1], [1,0,0], [1,0,1], [1,1,0]]
    cnt = 0
    for cluster in clusters.values():
        x = []
        y = []
        for point in cluster:
            x.append(point[0])
            y.append(point[1])
        ax.scatter(x, y, s = 60, c = colors[cnt], marker = markers[cnt])
        cnt += 1
    plt.show()

# Função Main()

# Pré-processando os dados
points = []
f = open('data.csv', 'r')
data = f.read()
rows = data.strip().split('\n')
for row in rows:
    split_row = row.split(',')
    points.append(split_row)
for point in points:
   point[0] = float(point[0].strip())
   point[1] = float(point[1].strip())

# Rodando o algoritmo K-Means
clusters = k_means(points, 6)
print_clusters(clusters)
plot_clusters(clusters)