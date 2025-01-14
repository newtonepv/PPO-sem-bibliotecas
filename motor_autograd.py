import math
class Valor:
    def __init__(self, valor_numerico: float, variaveis:set = set(), operador: str = "",ajustavel:bool = False, gradiente:float = 0,calc_der:bool=True) -> None:
        self.ajustavel = ajustavel
        self.valor_numerico = valor_numerico
        self.variaveis = variaveis
        self.op = operador
        self.gradiente = gradiente
        self._setar_grad_anteriores=lambda: None
        self.calc_der = calc_der

    def __str__(self)->str:
        return "Valor(data="+str(self.valor_numerico)+", operador="+str(self.op)+", ajustavel="+str(self.ajustavel)+")"

    def __add__(self, other)->'Valor':
        if( not isinstance(other,Valor)):#se other nao for do tipo Valor
            
            other = Valor(other) #se other nao for int ou float ele da erro, e não é ajustavel pq é um float

        valorResultante = Valor(self.valor_numerico+other.valor_numerico,variaveis=(self,other),operador="+")#nao é austavel, nao é um peso

        def setar_grad_anteriores():

            self.gradiente += valorResultante.gradiente if self.calc_der else 0
            other.gradiente += valorResultante.gradiente if other.calc_der else 0

        valorResultante._setar_grad_anteriores=setar_grad_anteriores

        return valorResultante


    def __mul__(self, other)->'Valor':

        if( not isinstance(other,Valor)):#se other nao for do tipo Valor
            other = Valor(other) #se other nao for int ou float ele da erro


        saida = Valor(self.valor_numerico*other.valor_numerico,variaveis=(self,other),operador="*")
        
        def setar_grad_anteriores():
            self.gradiente += (saida.gradiente*other.valor_numerico) if self.calc_der else 0
            other.gradiente += (saida.gradiente*self.valor_numerico) if other.calc_der else 0
        
        saida._setar_grad_anteriores= setar_grad_anteriores
        
        return saida

    def __pow__(self,other)-> 'Valor':
        assert(isinstance(other, (int, float)))
        saida = Valor(self.valor_numerico**other, (self,), "**")

        def setar_grad_anteriores():
            self.gradiente += (other * self.valor_numerico**(other-1) * saida.gradiente) if self.calc_der else 0

        saida._setar_grad_anteriores = setar_grad_anteriores
        return saida

    def tanh(self)->'Valor':
        x=self.valor_numerico
        tanhx = math.tanh(x)

        saida = Valor(tanhx,(self, ),"tanh")

        def setar_grad_anteriores():
            derivada = 1-tanhx**2 #derivada da unica entrada em relacao ao valor de saida
            self.gradiente += (derivada * saida.gradiente) if self.calc_der else 0
            

        saida._setar_grad_anteriores = setar_grad_anteriores
        return saida

    def exp(self) -> 'Valor':
        x=self.valor_numerico
        saida = Valor(math.exp(x), (self,), 'exp')

        def setar_grad_anteriores():
            self.gradiente +=(saida.valor_numerico*saida.gradiente) if self.calc_der else 0
        
        saida._setar_grad_anteriores = setar_grad_anteriores
        return saida
    
    def sigmoid(self):
        x = self.valor_numerico
        if x > 15:  # Increased threshold for better stability
            sigx = 0.99999
        elif x < -15:
            sigx = 0.00001
        else:
            sigx = 1/(1 + math.exp(-x))
        saida = Valor(sigx, (self,), 'sig')
        def setar_grad_anteriores():
            self.gradiente+=sigx*(1-sigx)*saida.gradiente
        saida._setar_grad_anteriores = setar_grad_anteriores
        return saida

    def __neg__(self)->'Valor': # -self
        return self * -1

    def __radd__(self, other)->'Valor': # other + self
        return self + other

    def __sub__(self, other)->'Valor': # self - other
        return self + (-other)

    def __rsub__(self, other)->'Valor': # other - self
        return other + (-self)

    def __rmul__(self, other)->'Valor': # other * self
        return self * other

    def __truediv__(self, other)->'Valor': # self / other
        return self * other**-1

    def __rtruediv__(self, other)->'Valor': # other / self
        return other * self**-1

    def derivadas(self):
        # Ordenação topológica dos nós
        ordem_topo = []
        visitados = set()
        
        def construir_ordem_topo(vertice:Valor):
            if vertice not in visitados:
                visitados.add(vertice)
                for var in vertice.variaveis:
                    construir_ordem_topo(var)
                ordem_topo.append(vertice)
        
        construir_ordem_topo(self)
        
        # Inicializar gradiente do nó final
        self.gradiente = 1
        
        # Percorrer nós em ordem reversa
        for vertice in reversed(ordem_topo):
            vertice._setar_grad_anteriores()
        
        return self.gradiente
    

import random
from typing import Union, List# assim o codigo é usavel no python >3.10

class No:
    def __init__(self, qtd_entradas:int, fun_atv:int):
        "na funcao de ativacao use 0-nenhuma | 1-tanh | 2-sigmoid"
        self.fun_atv = fun_atv
        self.pesos = [Valor(random.uniform(-1,1),ajustavel=True) for _ in range(0,qtd_entradas)] #esta dentro de [] pq é uma lista, pra poder por um for no meio
        self.vies = Valor(random.uniform(-1,1),ajustavel=True)

    def __call__(self, entradas:List[Union[float,int,Valor]]) -> Valor:

        assert(isinstance(entradas, list))
        assert(all(isinstance(x,(int,float,Valor)) for x in entradas))
        assert(len(entradas)==len(self.pesos))

        out = sum((e1*p1 for (e1, p1) in zip(self.pesos, entradas)))
        out += self.vies

        if(self.fun_atv==1):
            out = out.tanh()
        if(self.fun_atv==2):
            out = out.sigmoid()

        return out
    
    def parametros(self)->List[Valor]:
        return self.pesos+[self.vies]


class Camada:
    def __init__(self, qtd_entradas:int, qtd_nos:int, fun_atv:int):
        "na funcao de ativacao use 0 para nenhuma 1 para tanh 2 para sigmoid"
        self.fun_atv = fun_atv
        self.qtd_nos = qtd_nos
        self.qtd_entradas = qtd_entradas
        self.nos = [No(qtd_entradas, fun_atv) for _ in range(qtd_nos)]
    
    def __call__(self, entradas:List[Union[float,int,Valor]]) -> List[Valor]:
        saidas = [ no(entradas) for no in self.nos]
        return saidas


    def parametros(self)->List[Valor]:
        parametros = []
        #return [no.parametros() for no in self.nos] nao daria certo pois seria uma lista de listas
        for no in self.nos:
            for p in no.parametros():#recebe um iteravel e aumenta a lista
                parametros.append(p)
        return parametros
        

class MLP:
    def __init__(self, qtd_neuronios_em_cada_camada:list[int], fun_atv:list[int]):
        '''coloque na primeira posição do vetor a quantidade de entradas da rede neural
            nas funcoes de ativacao use 0 para nenhuma 1 para tanh 2 para sigmoid, nao pode por na entrada,
            ou seja, o tamanho desse array+1 é o tamanho do qtd_neuronios_em_cada_camada'''
        assert(len(qtd_neuronios_em_cada_camada)-1 == len(fun_atv))

        if not all(isinstance(x, int) and x > 0 for x in qtd_neuronios_em_cada_camada):
            raise ValueError("Todos os valores devem ser inteiros positivos")

        self.qtd_neuronios_em_cada_camada=qtd_neuronios_em_cada_camada

        self.camadas:List[Camada] = []

        camadas_mais_entrada = len(qtd_neuronios_em_cada_camada)

        if(camadas_mais_entrada<2):
            raise ValueError("A rede neural precisa ter no mínimo 1 camada, lembre que o primeiro valor do array é a qtd de entradas")


        for i in range(1, camadas_mais_entrada):
            self.camadas.append(Camada(qtd_neuronios_em_cada_camada[i-1],
                                        qtd_neuronios_em_cada_camada[i],
                                        fun_atv[i-1]))
  

    def __call__(self, entradas:List[Union[float,int,Valor]])->List[Valor]:
        saidas = entradas
        for c in self.camadas:
            saidas = c(saidas)
        return saidas

    def parametros(self)->List[Valor]:
        parametros = []
        for c in self.camadas:
            for p in c.parametros():
                parametros.append(p)
        return parametros
        
    def zerar_gradientes(self):
        for p in self.parametros():
            p.gradiente = 0

'''mlp = MLP([3, 4,  4, 1])

entradas_exemplo = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 5.0],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, 1.0]
]
saidas_verdade = [1.0, -1.0, -1.0, 1.0]

#primeira rodada
while(True):
    saidas = [mlp(x) for x in entradas_exemplo]
    perdas = [((saida[0]-saida_verdade)**2 )for saida, saida_verdade in zip(saidas , saidas_verdade)]
    perda_total = sum(perdas)

    print("")
    print("perda total = "+ str(perda_total))

    input()

    perda_total.derivadas()
    for p in mlp.parametros():
        p.valor_numerico+=0.01*(-p.gradiente)
        p.variaveis = ()
        p.gradiente = 0
    
    

#

saidas = [mlp(x) for x in entradas_exemplo]
perdas = [((saida[0]-saida_verdade)**2 )for saida, saida_verdade in zip(saidas , saidas_verdade)]
perda_total = sum(perdas)

print("")
print("perda total = "+ str(perda_total))'''