import torch
def perda_limitada(vantagem, chance_nova, chance_velha):
    ratio = torch.clip(torch.exp(chance_nova-chance_velha),1-config.limitador_epsilon,1+config.limitador_epsilon)
    perda = -ratio*vantagem
    return perda

def meq(valor, valor_real):#minimo erro quadrado
    return (valor-valor_real)**2

class Bloco_Com_Residual(torch.nn.Module):
    def __init__(self, entradas:int, dropout:int = 0.2):
        super().__init__()#nsei qual a diff de fazer isso a so super()
        self.bloco = torch.nn.Sequential(
            torch.nn.LayerNorm(entradas),
            torch.nn.GELU(),
            torch.nn.Linear(entradas,entradas*4),
            torch.nn.GELU(),
            torch.nn.Linear(entradas*4, entradas),
            torch.nn.Dropout(dropout)
        )
    def forward(self, entradas:torch.Tensor):
        return entradas + self.bloco(entradas)

class Agente(torch.nn.Module):
    def __init__(self, observacoes:int, acoes:int, dimensoes:int=64, blocos:int=2):
        super().__init__()
        self.num_observacoes = observacoes
        self.base=torch.nn.Sequential(
            torch.nn.Linear(observacoes,dimensoes),
            *[Bloco_Com_Residual(dimensoes) for i in range(blocos)]
        )

        self.valor_do_estado = torch.nn.Linear(dimensoes,1)

        self.politica = torch.nn.Linear(dimensoes,acoes)
        torch.nn.init.orthogonal_(self.politica.weight, 0.01)

    def rodar_valor(self, observacoes:torch.Tensor):
        '''print("observacoes.shape", flush=True)
        print(observacoes.shape, flush=True)
        print("observacoes", flush=True)
        print(self.num_observacoes, flush=True)'''
        return self.valor_do_estado(self.base(observacoes))
    
    def rodar_politica(self, observacoes:torch.Tensor, acao:int = None)-> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            acao: 
                Tensor(int)
            chances.log_probAcao: 
                Tensor(float)
            chances.entropy: 
                Tensor(float)
            self.valor_do_estado base: 
                Tensor[float]
        """
        base = self.base(observacoes)
        logits = self.politica(base)
        chances = torch.distributions.Categorical(logits=logits)
        if(acao==None):
            acao = chances.sample()
        return acao, chances.log_prob(acao), chances.entropy(), self.valor_do_estado(base) 
    

import gym
from configuracoes import config

env = gym.make(config.nome_id_ambiente)
num_acoes = env.action_space.n
num_obser = env.observation_space.shape[0]


agente = Agente(num_obser, num_acoes)
otimizador = torch.optim.Adam(agente.parameters(),config.vel_aprendizado)

for o in range(config.num_atualizacoes):
    observacoes = []
    acoes = []
    chances = []
    recompensas = []
    valores_estimados = []

    recompensa_total_primeiro_amb = 0
    with torch.no_grad():
        cont_ambientes=0
        for _ in range(config.num_ambientes):
            _observacoes = []
            _acoes = []
            _chances = []
            _recompensas = []
            _valores = []

            cont_ambientes+=1
            cont_passos=0

            obs_bruta = env.reset()
            observacao = torch.Tensor(obs_bruta[0])
            terminou=False
            truncou=False
            while (not terminou and not truncou) and not (cont_passos>config.max_passos_por_atualizacao):
                
                _observacoes.append(observacao)

                a = agente.rodar_politica(observacao)
                tensor_acao, chance_tensor, _, valor_do_estado_tensor = a
                acao = torch.Tensor.item(tensor_acao)

                observacao,recompensa,terminou,truncou,info = env.step(acao)
                observacao = torch.Tensor(observacao)

                print(recompensa)

                _acoes.append(tensor_acao)
                _chances.append(chance_tensor)
                if(_ ==0):
                    recompensa_total_primeiro_amb+=recompensa
                _recompensas.append(torch.Tensor([recompensa]))
                _valores.append(valor_do_estado_tensor)

                cont_passos+=1
            observacoes.append(_observacoes)
            acoes.append(_acoes)
            chances.append(_chances)
            recompensas.append(_recompensas)
            valores_estimados.append(_valores)
    """atualmente temos
        observacoes[num_ambientes][o_numero_de_steps_do_ambiente]"""

    vantagens = []
    valores_reais = []
    for i in range(config.num_ambientes):
        vantagens.append([])
        valores_reais.append([])

        for j in range(len(observacoes[i])):
            vantagens[i].append(0.0)
            valores_reais[i].append(0.0)

        valor_s_m_1 = agente.rodar_valor(observacoes[i][len(observacoes[i])-1])
        vantagem_s_m_1 = 0
        for j in reversed(range(len(observacoes[i]))):
            delta = (recompensas[i][j]+config.gamma*valor_s_m_1) - valores_estimados[i][j]
            vantagens[i][j] = delta + config.gamma * config.evg_lambda * vantagem_s_m_1
            vantagem_s_m_1 = vantagens[i][j]
            valor_s_m_1 = valores_estimados[i][j]
        
        for j in range(len(vantagens[i])):
            valores_reais[i][j] = vantagens[i][j] + valores_estimados[i][j]


    observacoes_aux = []
    acoes_aux = []
    chances_aux = []
    recompensas_aux = []
    valores_estimados_aux = []
    vantagens_aux = []
    valores_reais_aux = []

    for i in range(config.num_ambientes):
        for j in range(len(observacoes[i])):
            observacoes_aux.append(observacoes[i][j])

        for j in range(len(acoes[i])):
            acoes_aux.append(acoes[i][j])
        #print(len(acoes))

        for j in range(len(chances[i])):
            chances_aux.append(chances[i][j])

        for j in range(len(valores_estimados[i])):
            valores_estimados_aux.append(valores_estimados[i][j])

        for j in range(len(vantagens[i])):
            vantagens_aux.append(vantagens[i][j])

        for j in range(len(valores_reais[i])):
            valores_reais_aux.append(valores_reais[i][j])

    observacoes = observacoes_aux
    acoes = acoes_aux
    chances = chances_aux
    recompensas = recompensas_aux
    valores_estimados = valores_estimados_aux
    vantagens = vantagens_aux
    valores_reais = valores_reais_aux

    cont_acoes = 0
    while(cont_acoes<len(acoes)):#enquanto ainda tiver dado
        perda_politica = 0
        perda_valor = 0
        perda_entropia = 0

        cont_minibatch = 0
        while( cont_minibatch < (config.num_ambientes*config.max_passos_por_atualizacao)//config.qtd_divisoes_da_experiencia ):
            if(cont_acoes>=len(acoes)):
                break#faz o update e ja volta a fazer o rollout

            _, chance_nova, entropia, valor = agente.rodar_politica(observacoes[cont_acoes],acoes[cont_acoes])

            perda_politica += perda_limitada(vantagens[cont_acoes], chance_nova, chances[cont_acoes])
            perda_valor += meq(valor, valores_reais[cont_acoes])
            perda_entropia += entropia #maximizar a entropia

            cont_acoes+=1
            cont_minibatch+=1

        perda_total = -config.peso_entropia*perda_entropia + config.peso_valor*perda_valor + perda_politica
        otimizador.zero_grad()
        perda_total.backward()
        torch.nn.utils.clip_grad_norm_(agente.parameters(),config.gradiente_max)
        otimizador.step()
