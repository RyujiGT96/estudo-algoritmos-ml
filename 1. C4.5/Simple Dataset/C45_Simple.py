import numpy as np

# Dados
x1  = [0, 1, 1, 2, 2, 2]
x2 = [0, 0, 1, 1, 1, 0]
y = np.array([0, 0, 0, 1, 1, 0])

# Função para split dos dados
def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}

# Função para o cálculo do entropia
def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts = True)
    freqs = counts.astype('float')/len(s)
    for p in freqs:
        if p != 0.0:
            res -= p*np.log2(p)
    return res

# Função para o cálculo de ganho de informação (information gain) 
def mutual_information(y, x):
    res = entropy(y)
    # Particionamos x de acordo com os valores dos atrinutos x_i
    val, counts = np.unique(x, return_counts = True)
    freqs = counts.astype('float')/len(x)
    # Calculamos uma média ponderada da entropia
    for p, v in zip(freqs, val):
        res -= p*entropy(y[x == v])
    return res

from pprint import pprint

def is_pure(s):
    return len(set(s)) == 1

def recursive_split(x, y):
    # Se não for possível realizar o split, retornar o set original
    if is_pure(y) or len(y) == 0:
        return y
    # Escolher o atributo que fornece o maior ganho de informação
    gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)
    # Se não houver ganho, nada deve ser feito e deve-se retornar o set original
    if np.all(gain<1e-6):
        return y
    # O split é realizado utilizando o atributo selecionado
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis = 0)
        x_subset = x.take(v, axis = 0)

        res["x_%d = %d" % (selected_attr + 1, k)] = recursive_split(x_subset, y_subset)

    return res

X = np.array([x1,x2]).T
pprint(recursive_split(X, y))

