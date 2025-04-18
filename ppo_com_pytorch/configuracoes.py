class configuracoes:
    def __init__(self):
        self.nome_id_ambiente = 'CartPole-v1'
        self.num_ambientes = 32
        self.semente = 200
        self.nome_video = 'cartpolev1'
        self.max_passos_por_atualizacao = 64 #isso para cada ambiente
        self.qtd_divisoes_da_experiencia = 2
        self.vel_aprendizado = 1e-3
        self.num_atualizacoes = 488
        self.gamma = 0.99
        self.evg_lambda = 0.95
        self.limitador_epsilon = 0.2
        self.num_atualizacoes_com_mesmo_dataset = 2
        self.peso_entropia = 0.01
        self.peso_valor = 0.5
        self.gradiente_max = 0.5
config = configuracoes()

class config_colab:
    def __init__(self):
        self.exp_name = "cartpole"
        self.gym_id = "CartPole-v1"
        self.learning_rate = 1e-3
        self.total_timesteps = 1000000
        self.max_grad_norm = 0.5
        self.num_trajcts = 32
        self.max_trajects_length = 64
        self.gamma = 0.99
        self.gae_lambda =0.95
        self.num_minibatches = 2
        self.updates_no_mesmo_rollout = 2
        self.clip_epsilon = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.num_returns_to_average = 3
        self.num_episodes_to_average = 23
        self.batch_size = self.num_trajcts * self.max_trajects_length
        self.minibatch_size = self.batch_size // self.num_minibatches

config_colabe = config_colab()