from motor_autograd import MLP , Valor
from copy import deepcopy
import random

def pegar_index_da_acao_tomada(lista: list[Valor]) -> int:
    # Calculate total weight
    total = sum(v.valor_numerico for v in lista)
    
    # Generate random value between 0 and total
    r = random.uniform(0, total)
    
    # Find the index
    acumulado = 0
    for i, v in enumerate(lista):
        acumulado += v.valor_numerico
        if acumulado > r:
            return i
            
    # Fallback to last index (in case of floating point rounding issues)
    return len(lista) - 1

def limitar(con:Valor,limite_concordancia)->Valor:
    if(con.valor_numerico > 1+limite_concordancia):
        con = con - (con.valor_numerico-(1+limite_concordancia))
    elif(con.valor_numerico<1-limite_concordancia):
        con = con+((1-limite_concordancia)-con.valor_numerico)
    return con

def softmax(valores:list[Valor])->list[Valor]:
    soma = Valor(0)
    for v in valores:
        soma += v
    saida:list[Valor] = [] 
    for v in valores:
        saida.append(v/soma)
    return saida

class Politica():
    def __init__(self, camadas:list, diminuicao_nota:float, qtd_acoes_afetadas_pela_recompensa:int, vel_aprendizagem:float, limite_concordancia:float):
        "camadas tem que estar no formato aceito pela classe MLP de motor_autograd"
        self.delta_recompensas:list[float] = []
        self.concordancia:list[Valor] = []
        self.diminuicao_nota = diminuicao_nota
        self.vel_aprendizagem = vel_aprendizagem
        self.qtd_acoes_afetadas_pela_recompensa = qtd_acoes_afetadas_pela_recompensa
        self.limite_concordancia = limite_concordancia

        fun_ativacao:list[int] = [1 for _ in range(len(camadas)-2)]#a primeira é entrada e a ultima vai de sigmoid
        fun_ativacao.append(2)#a ultima é sigmoid
        self.pol_atual = MLP(camadas, fun_ativacao)
        self.pol_antiga = deepcopy(self.pol_atual)

    def __call__(self, estado):

        resposta_pol_nova = softmax(self.pol_atual(estado))
        resposta_pol_velha = softmax(self.pol_antiga(estado))
        index_acao = pegar_index_da_acao_tomada(resposta_pol_nova)
        chance_da_nova_tomar_acao = resposta_pol_nova[index_acao]
        chance_da_velha_tomar_acao = resposta_pol_velha[index_acao]

        concordancia = chance_da_nova_tomar_acao/chance_da_velha_tomar_acao
        
        self.concordancia.append(limitar(concordancia, self.limite_concordancia))
        self.delta_recompensas.append(0)
        return index_acao
    
    def premiar(self, premio:float, index_acao_a_premiar:int = -1, premio_esperado:int = 0):
        
        if(index_acao_a_premiar==-1):
            index_acao_a_premiar=len(self.delta_recompensas)-1
    
        for i in range(self.qtd_acoes_afetadas_pela_recompensa):
            index = index_acao_a_premiar-i
            if(index<0):
                break
            self.delta_recompensas[index] += (premio-premio_esperado) * (self.diminuicao_nota**i)

    def atualizar(self):
        self.pol_antiga = deepcopy(self.pol_atual)
        assert(len(self.delta_recompensas)==len(self.concordancia))
        perda_acumulada = Valor(0)

        for dr, con in zip(self.delta_recompensas,self.concordancia):
            perda_acumulada += dr*con

        perda_acumulada.derivadas()
        for p in self.pol_atual.parametros():
            p.valor_numerico+=self.vel_aprendizagem*p.gradiente 
            p.gradiente = 0
        self.concordancia.clear()
        self.delta_recompensas.clear()

    def get_index_ultima_acao(self)-> int:
        return len(self.delta_recompensas)-1

    

    

    
