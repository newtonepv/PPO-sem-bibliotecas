import torch
import torch.nn as nn
import random
from copy import deepcopy

def pegar_index_da_acao_tomada(lista: list[torch.Tensor]) -> int:
    # Calculate total weight
    total = sum(v.item() for v in lista)
    
    # Generate random value between 0 and total
    r = random.uniform(0, total)
    
    # Find the index
    acumulado = 0
    for i, v in enumerate(lista):
        acumulado += v.item()
        if acumulado > r:
            return i
            
    # Fallback to last index (in case of floating point rounding issues)
    return len(lista) - 1

def limitar(con: torch.Tensor, limite_concordancia) -> torch.Tensor:
    con_value = con.item()
    if con_value > 1 + limite_concordancia:
        return torch.tensor(1 + limite_concordancia, requires_grad=True)
    elif con_value < 1 - limite_concordancia:
        return torch.tensor(1 - limite_concordancia, requires_grad=True)
    return con

def softmax(valores: list[torch.Tensor]) -> list[torch.Tensor]:
    # Convert list to tensor
    valores_tensor = torch.stack(valores)
    # Apply softmax
    softmax_fn = nn.Softmax(dim=0)
    resultado = softmax_fn(valores_tensor)
    # Convert back to list
    return [v for v in resultado]

class MLP(nn.Module):
    def __init__(self, camadas: list, fun_ativacao: list):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(camadas) - 1):
            self.layers.append(nn.Linear(camadas[i], camadas[i + 1]))
            
        self.activation_fns = []
        for fn_type in fun_ativacao:
            if fn_type == 1:  # ReLU
                self.activation_fns.append(nn.ReLU())
            elif fn_type == 2:  # Sigmoid
                self.activation_fns.append(nn.Sigmoid())
                
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        for i, (layer, activation) in enumerate(zip(self.layers, self.activation_fns)):
            x = activation(layer(x))
        return [x[i] for i in range(x.size(0))]

class Politica():
    def __init__(self, camadas: list, diminuicao_nota: float, qtd_acoes_afetadas_pela_recompensa: int, 
                 vel_aprendizagem: float, limite_concordancia: float):
        self.delta_recompensas: list[float] = []
        self.concordancia: list[torch.Tensor] = []
        self.diminuicao_nota = diminuicao_nota
        self.vel_aprendizagem = vel_aprendizagem
        self.qtd_acoes_afetadas_pela_recompensa = qtd_acoes_afetadas_pela_recompensa
        self.limite_concordancia = limite_concordancia

        fun_ativacao: list[int] = [1 for _ in range(len(camadas)-2)]
        fun_ativacao.append(2)  # last one is sigmoid
        
        self.pol_atual = MLP(camadas, fun_ativacao)
        self.pol_antiga = deepcopy(self.pol_atual)
        self.optimizer = torch.optim.SGD(self.pol_atual.parameters(), lr=vel_aprendizagem)

    def __call__(self, estado):
        with torch.no_grad():
            resposta_pol_nova = softmax(self.pol_atual(estado))
            resposta_pol_velha = softmax(self.pol_antiga(estado))
            
        index_acao = pegar_index_da_acao_tomada(resposta_pol_nova)
        chance_da_nova_tomar_acao = resposta_pol_nova[index_acao]
        chance_da_velha_tomar_acao = resposta_pol_velha[index_acao]

        concordancia = chance_da_nova_tomar_acao/chance_da_velha_tomar_acao
        
        self.concordancia.append(limitar(concordancia, self.limite_concordancia))
        self.delta_recompensas.append(0)
        return index_acao
    
    def premiar(self, premio: float, index_acao_a_premiar: int = -1, premio_esperado: int = 0):
        if index_acao_a_premiar == -1:
            index_acao_a_premiar = len(self.delta_recompensas)-1
    
        for i in range(self.qtd_acoes_afetadas_pela_recompensa):
            index = index_acao_a_premiar-i
            if index < 0:
                break
            self.delta_recompensas[index] += (premio-premio_esperado) * (self.diminuicao_nota**i)

    def atualizar(self):
        self.pol_antiga = deepcopy(self.pol_atual)
        assert len(self.delta_recompensas) == len(self.concordancia)
        
        self.optimizer.zero_grad()
        perda_acumulada = torch.tensor(0.0, requires_grad=True)

        for dr, con in zip(self.delta_recompensas, self.concordancia):
            perda_acumulada = perda_acumulada + dr * con

        perda_acumulada.backward()
        self.optimizer.step()
        
        self.concordancia.clear()
        self.delta_recompensas.clear()

    def get_index_ultima_acao(self) -> int:
        return len(self.delta_recompensas)-1