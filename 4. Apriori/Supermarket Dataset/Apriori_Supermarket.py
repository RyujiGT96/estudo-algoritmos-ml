# Importação dos pacotes utilizados no algoritmo
from itertools import chain, combinations
import operator

# Função para obtenção de todas as combinações de items
def subsets(itemset):
    return chain(*[combinations(itemset, i + 1) for i, a in enumerate(itemset)])

# Função  do algortimo Apriori
def apriori(data, min_support, min_confidence):
    # Lista de item sets e transações
    itemset, transaction_list = itemset_from_data(data)
    print('\n')   
    print(f'Item Sets: \n \n{list(itemset)}')
    print('\n')   
    print(f'Transactions: \n \n{list(transaction_list)}')

    # Gerar candidatos
    candidates = get_candidates(transaction_list, itemset, min_support)
   
    rules = list()
    for sets in candidates.keys():
        if len(sets) > 1:
            for subset in subsets(sets):
                item = sets.difference(subset)
                if item: # If not None
                    subset = frozenset(subset)
                    subset_item = subset | item  # União de sets
                    confidence = float(candidates[subset_item]) / candidates[subset]
                    if confidence >= min_confidence:
                        rules.append((subset, item, confidence))
    return rules, candidates

# Função para obtenção de combinações de k-itens
def joinset(itemset, k):
    joint_set = set()
    for i in itemset:
        for j in itemset:
            if len(i.union(j)) == k:
                joint_set.add(i.union(j))
    return joint_set

# Função para determinar os candidatos à itemsets frequentes
def get_candidates(transaction_list, itemset, min_support):
    candidates = dict()
    k = 1
    k_itemset = get_freq_itemset(transaction_list, itemset, min_support)
    candidates.update(k_itemset)
    k += 1
    while True:
        itemset = joinset(k_itemset, k)
        k_itemset = get_freq_itemset(transaction_list, itemset, min_support)
        if not k_itemset: # If None
            break
        candidates.update(k_itemset)
        k += 1
    return candidates

# Função para determinar os itens mais frequentes de acordo com o valor de suporte
def get_freq_itemset(transaction_list, itemset, min_support):
    len_transaction_list = len(transaction_list)
    freq_itemsets = dict()
    for item in itemset:
        freq_itemsets[item] = 0
        for row in transaction_list:
            if item.issubset(row):
                freq_itemsets[item] += 1
        freq_itemsets[item] = freq_itemsets[item] / len_transaction_list
    relevant_itemsets = dict()
    for item, support in freq_itemsets.items():
        if support >= min_support:
            relevant_itemsets[item] = support
    return relevant_itemsets

# Constrói lista de itemsets e transações
def itemset_from_data(data):
    itemset = set()
    transaction_list = list()
    for row in data:
        transaction_list.append(frozenset(row))
        for item in row:
            if item not in itemset:
                itemset.add(frozenset([item]))
    return itemset, transaction_list

# Função para imprimir resultados
def print_report(rules, candidates):
    print('\n')
    print('---Frequent Itemsets---')
    print('[Itemset] | [Support]')
    sorted_candidates = sorted(candidates.items(), key=operator.itemgetter(1))
    for candidate in sorted_candidates:
        print(f'{tuple(candidate[0])} : {round(candidate[1], 4)}')

    print('\n')
    print('---Rules---')
    sorted_rules = sorted(rules, key=lambda s : s[2])
    print('[Rule] | [Confidence]')
    for rule in sorted_rules:
         print(f'{tuple(rule[0])} => {tuple(rule[1])} : {round(rule[2], 4)}')

# Função para leitura de dados csv
def get_csv_data(filename):
    data = []
    f = open(filename, 'r')
    csv_data = f.read()
    rows = csv_data.strip().split('\n')
    for row in rows:
        split_row = row.strip().split(',')
        data.append(split_row)
    return data

# Main()
data = get_csv_data('supermarket_data.csv')
print('\n')
#print('Leitura dos dados:')
#print(data)

print('\n')
min_support = float(input('Minimum Support: '))
min_confidence = float(input('Minimum Confindence: '))

rules, candidates = apriori(data, min_support, min_confidence)
print_report(rules, candidates)




