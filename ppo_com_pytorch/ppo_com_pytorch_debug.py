import gym
import torch
from typing import List,Callable, Tuple
from configuracoes import config
import numpy


processamento = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def env_creator_function(nome_ambiente:str,indx: int, semente:int,nome_video:str, record_video:bool=False):
    def create_env():
        ambiente = gym.make(nome_ambiente, render_mode="rgb_array")
        ambiente = gym.wrappers.RecordEpisodeStatistics(ambiente)
        if(record_video):
            if(indx==0):#gravar so o primeiro
                ambiente = gym.wrappers.RecordVideo(ambiente,f'videos/{nome_video}')
        ambiente.action_space.seed(semente)
        ambiente.observation_space.seed(semente)
        return ambiente
    return create_env

funcoes_criar_ambiente: List[Callable] = [env_creator_function(config.nome_id_ambiente,
                                                               i,
                                                               config.semente+i,
                                                               config.nome_video) for i in range(config.num_ambientes)]

ambientes = gym.vector.SyncVectorEnv(funcoes_criar_ambiente)

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
        return acao, chances.log_prob(acao), chances.entropy(), self.valor_do_estado(base) 
        """
        base = self.base(observacoes)
        logits = self.politica(base)
        chances = torch.distributions.Categorical(logits=logits)
        if(acao==None):
            acao = chances.sample()
        return acao, chances.log_prob(acao), chances.entropy(), self.valor_do_estado(base) 

def gerar_experiencia(agente:Agente,ambientes:gym.vector.SyncVectorEnv, quantidade_max_passos:int, observacao_atual:torch.Tensor, acabou_atual:torch.Tensor, recompensa_somada:List[float]):
    '''
    retorno = observacoes, acoes, recompensas, probabilidades, valor_dos_estados,observacao_atual, acabou_atual, recompensa_somada, acabou_vetor
    '''
    observacoes:torch.Tensor = torch.zeros(
        (config.num_ambientes, quantidade_max_passos)+ambientes.single_observation_space.shape
    ).to(processamento)

    acabou_vetor:torch.Tensor = torch.zeros(
        (config.num_ambientes, quantidade_max_passos)
    ).to(processamento)

    recompensas:torch.Tensor = torch.zeros(
        (config.num_ambientes, quantidade_max_passos)
    ).to(processamento)

    valor_dos_estados:torch.Tensor = torch.zeros(
        (config.num_ambientes, quantidade_max_passos)
    ).to(processamento)

    probabilidades:torch.Tensor = torch.zeros(
        (config.num_ambientes, quantidade_max_passos)+ambientes.single_action_space.shape
    ).to(processamento)

    acoes:torch.Tensor = torch.zeros(
        (config.num_ambientes, quantidade_max_passos)
    ).to(processamento)
    count_infos =0
    count_desinfos = 0
    for i in range(quantidade_max_passos):
        observacoes[:,i] = observacao_atual
        acabou_vetor[:,i] = acabou_atual

        with torch.no_grad():
            acao, probabilidade, _, valor=agente.rodar_politica(observacao_atual)# nao guardamos a entropia pq a gente usa ela na funcao de perda so na hora de atualizar o modelo, quando a gente pede novas probabilidades pros ratios
        
        acoes[:,i] = acao
        probabilidades[:,i] = probabilidade
        valor_dos_estados[:,i] = valor.flatten()

        observacao_atual,recompensa_atual,acabou_atual,_,info=ambientes.step(acao.cpu().numpy())

        recompensas[:,i] = torch.Tensor(recompensa_atual).to(processamento)
        observacao_atual = torch.Tensor(observacao_atual).to(processamento)
        acabou_atual = torch.Tensor(acabou_atual).to(processamento)
        if info: 
            count_infos+=1
            for item in info['final_info']:
                if item and "episode" in item.keys():
                    #print(item.keys())
                    recompensa_somada.append(item['episode']['r'])#float pq é a soma das recompensas de td os ambientes pra plotar dps o grafico
                    
        else:
            count_desinfos+=1

    return observacoes, acoes, recompensas, probabilidades, valor_dos_estados,observacao_atual, acabou_atual, recompensa_somada, acabou_vetor

def aproximacao_da_vantagem_geral(agente: Agente, gamma: float, lambda_param: float, num_ambientes: int, num_max_passos: int, ultima_observacao: torch.Tensor, recompensas: torch.Tensor, valores_dos_estados: torch.Tensor, acabou_vetor: torch.Tensor, processamento):
    #a observacao é a observacao logo apos a ultima recompensa, por isso ela é util pra pegar o valor do ultimo estado, assim calculando a vantagem da ultima acao ( q levou a aquela recompensa )
   
    with torch.no_grad():
        ultimo_valor = agente.rodar_valor(ultima_observacao).reshape(1, -1)

    vantagens = torch.zeros((config.num_ambientes,num_max_passos)).to(processamento)
    ultima_vantagem=0

    for i in reversed(range(num_max_passos)):
        nao_acabou = 1-acabou_vetor[:,i]
        ultimo_valor = ultimo_valor*nao_acabou
        ultima_vantagem = ultima_vantagem*nao_acabou
        delta = recompensas[:,i]+gamma*ultimo_valor - valores_dos_estados[:,i]
        vantagem = delta+ gamma*lambda_param*ultima_vantagem
        vantagens[:,i] = vantagem
        ultimo_valor = valores_dos_estados[:,i]

    recompensas_acumuladas = vantagens+valores_dos_estados
    return vantagens, recompensas_acumuladas # de alguma forma vantagens+valores_dos_estados = recompensas acumuladas

class Experiencia(torch.utils.data.Dataset):
    def __init__(self, vantagens:torch.Tensor, observacoes:torch.Tensor, acoes:torch.Tensor, probabilidades:torch.Tensor, recompensas_acumuladas:torch.Tensor):
        #tudo o que precisamos pras funcoes de perda, tirando o que vamos pegar na hora, como os novos valores pra cada estado
        self.vantagens = vantagens.reshape(-1)
        self.observacoes = observacoes.reshape((-1,) + ambientes.single_observation_space.shape)
        self.acoes = acoes.reshape((-1,) + ambientes.single_action_space.shape).long() #long
        self.probabilidades = probabilidades.reshape(-1)
        self.recompensas_acumuladas = recompensas_acumuladas.reshape(-1)

    def __getitem__(self, index:int):
        return [
            self.vantagens[index],
            self.observacoes[index],
            self.acoes[index],
            self.probabilidades[index],
            self.recompensas_acumuladas[index]
        ]
    
    def __len__(self):
        return len(self.observacoes)
    
def perda_limitada(vantagens: torch.Tensor, epsilon: float, prob_antiga: torch.Tensor, prob_nova: torch.Tensor):
    ratio = torch.exp(prob_nova - prob_antiga)  # Calcula o ratio entre as probabilidades antiga e nova
    policy_loss = -vantagens * ratio  # Perda original
    clipped_loss = -vantagens * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)  # Perda com clipping

    # Escolhe a maior perda (pior caso)
    perda = torch.max(policy_loss, clipped_loss).mean()
    
    return perda

def perda_rede_valor(valores:torch.Tensor,recompensas_acumuladas:torch.Tensor):
    return ((valores-recompensas_acumuladas)**2).mean()

ag = Agente(observacoes = ambientes.single_observation_space.shape[0], acoes=ambientes.single_action_space.n).to(processamento)

otimizador = torch.optim.Adam(ag.parameters(),  lr = config.vel_aprendizado)

observacao_atual = torch.Tensor(ambientes.reset()[0]).to(processamento)
acabou_atual = torch.zeros(config.num_ambientes).to(processamento)
recompensa_somada = []

for i in range(1,config.num_atualizacoes+1):

    observacoes, acoes, recompensas, probabilidades, valor_dos_estados,observacao_atual, acabou_atual, recompensa_somada, acabou_vetor = gerar_experiencia(ag, ambientes, config.max_passos_por_atualizacao, observacao_atual, acabou_atual, recompensa_somada)
    
    vantagens, recompensas_acumuladas = aproximacao_da_vantagem_geral(ag,config.gamma, config.evg_lambda,config.num_ambientes,config.max_passos_por_atualizacao,
                                                                      observacao_atual,recompensas,valor_dos_estados,acabou_vetor, processamento)

    print(acabou_vetor)
    '''print("vantagens.shape")
    print(vantagens.shape)
    print("observacoes.shape")
    print(observacoes.shape)
    print("acoes.shape")
    print(acoes.shape)
    print("probabilidades.shape")
    print(probabilidades.shape)
    print("recompensas_acumuladas.shape")
    print(recompensas_acumuladas.shape)
    print("\n\n")'''

    conjunto_de_dados = Experiencia(vantagens,observacoes,acoes,probabilidades,recompensas_acumuladas)
    carregador_de_dados = torch.utils.data.DataLoader(conjunto_de_dados,
                                             (config.num_ambientes*config.max_passos_por_atualizacao)//config.qtd_divisoes_da_experiencia,
                                             shuffle=True)

    porcentagem = 1-(i-1.0)/config.num_atualizacoes
    otimizador.param_groups[0]['lr'] = config.vel_aprendizado*config.num_ambientes#diminuir linearmente a velocidade de aprendizado

    for i in range(config.num_atualizacoes_com_mesmo_dataset):
        for dado in carregador_de_dados:
            exp_vantagens:torch.Tensor
            exp_observacoes:torch.Tensor
            exp_acoes:torch.Tensor
            exp_probabilidades:torch.Tensor
            exp_recompensas_acumuladas:torch.Tensor
            exp_vantagens, exp_observacoes, exp_acoes, exp_probabilidades, exp_recompensas_acumuladas = dado

            '''print("exp_vantagens.shape")
            print(exp_vantagens.shape)
            print("exp_observacoes.shape")
            print(exp_observacoes.shape)
            print("exp_acoes.shape")
            print(exp_acoes.shape)
            print("exp_probabilidades.shape")
            print(exp_probabilidades.shape)
            print("exp_recompensas_acumuladas.shape")
            print(exp_recompensas_acumuladas.shape)'''

            _,probabilidades_novas, entropias_novas, valor_dos_estados_novos =  ag.rodar_politica(exp_observacoes,exp_acoes)

            probabilidades_novas=probabilidades_novas.flatten()
            entropias_novas=entropias_novas.flatten()
            valor_dos_estados_novos=valor_dos_estados_novos.flatten()

            '''print("probabilidades_novas.shape")
            print(probabilidades_novas.shape)
            print("entropias_novas.shape")
            print(entropias_novas.shape)
            print("valor_dos_estados_novos.shape")
            print(valor_dos_estados_novos.shape)'''

            perda_limitada_atual = perda_limitada(exp_vantagens, config.limitador_epsilon, exp_probabilidades,probabilidades_novas)

            perda_entropia = -entropias_novas.mean()

            perda_rede_valor_atual =  perda_rede_valor(valor_dos_estados_novos,exp_recompensas_acumuladas )

            perda_total = perda_limitada_atual + (perda_entropia*config.peso_entropia) + (perda_rede_valor_atual*config.peso_valor)

            otimizador.zero_grad()
            perda_total.backward()
            torch.nn.utils.clip_grad_norm_(ag.parameters(), config.gradiente_max)
            break
        break
    break
    print(recompensa_somada[len(recompensa_somada)-1])

            
