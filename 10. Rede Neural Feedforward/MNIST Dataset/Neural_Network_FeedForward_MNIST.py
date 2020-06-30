# importação das bibliotecas a serem utilizadas
# pytorch: biblioteca para implementação de modelos de deep learning / redes neurais
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# coleta dos dados: serão analisados amaostras da base de dados MNIST para reconhecimento de dígitos manuscritos
# o pytorch já oferece funções para a coleta desses dados
# dados originais: http://yann.lecun.com/exdb/mnist/

# coleta de dados para treinamento
train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
# coleta de dados para teste
test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# preparação dos dados: dados serão analisados em bateladas de 10 amostras
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

# implementação do modelo / arquitetura de rede neural a ser utilizado
# optou-se por uma rede neural feedforward  de 4 camadas de unidades de processamento
class Net(nn.Module):
    # definição da estrutura das camadas
    def __init__(self):
        super().__init__()
        # imagens analisadas possuem dimensão 28x28 pixels
        # logo os dados de entrada são vetores de dimensão (1, 28x28) = flatten image
        self.fc1 = nn.Linear(28*28, 64) # camada oculta 1: 64 unidades de processamento
        self.fc2 = nn.Linear(64, 64) # camada oculta 2: 64 unidades de processamento
        self.fc3 = nn.Linear(64, 64) # camada oculta 3: 64 unidades de processamento
        self.fc4 = nn.Linear(64, 10) # camada de saída: 10 unidades de processamento
        # amotras são classificadas por meio de 10 classes (0,1,2,3,4,5,6,7,8,9) 
        # o mesmo número de unidades de saída

    # implementação da propagação forward  
    def forward(self, x):
    	# amostra x é propagada de camada em camada
    	# para 3 primeiras camadas ocultas a função de ativação ReLU é utilizada
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # para a camada de saída utiliza-se a função de ativação LogSoftmax, comum em problemas de classificação
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

    # pytorch: valores dos parâmetros são iniciados randomicamente utilizando distribuições uniformes

# iniciação do modelo de rede neural implementado
net = Net()

# escolha do algoritmo de treinamento
# opta-se pelo uso do algoritmo de otimização Adam para o algoritmo de backprogation
optimizer = optim.Adam(net.parameters(), lr=0.001) # taxa de aprendizado lr = 0.001

# processo de treinamento 
loss_log = []
for epoch in range(3): # 3 passagens sobre todas as amostras de treinamento são realizadas
	total_loss = torch.tensor([[0]])
    for data in trainset: # iteração da propagação de amostras de treinamento
        X, y = data # amostras de treinamento 
        net.zero_grad() # zera os valores de gradiente da rede neural
        # propagação forward
        output = net(X.view(-1, 28*28)) 
        # cálculo do erro das repostas geradas na propagação forward
        loss = F.nll_loss(output, y) # opta-se pelo uso da função erro Negative Log Likelihood
        							 # para problemas de classificação
        # propagação backward
        # pytroch: gradientes são calculados automaticamente durante a propagação forward
        loss.backward() 
        # ajuste dos valores dos parâmetros pesos e bias
        optimizer.step() # algoritmo Adam otimiza os valores dos parâmetros pela propagação dos gradientes da rede 
        total_loss += loss
    # valores de erro
    loss_log.append(total_loss.item())

# processo de teste
correct = 0
total = 0
# desativa o cálculo dos gradientes
with torch.no_grad():
    for data in testset: # iteração da propagação das amostras de teste
        X, y = data # amostras de teste
        output = net(X.view(-1, 28*28)) # respostas geradas pela rede neural
        # checagem de reposta gerada está correta
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
# Acurácia:  
print("Accuracy: ", round(correct/total, 3))