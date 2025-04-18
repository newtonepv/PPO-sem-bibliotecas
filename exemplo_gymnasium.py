import gymnasium as gym
import torch
from politicas_com_pytorch import Politica  # Assuming we saved the previous code as pytorch_policy.py
from copy import deepcopy
import random
from gymnasium.wrappers import RecordVideo

# Initialize Gymnasium environment with video recording
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, "videos", episode_trigger=lambda x: x % 50 == 0)

state_dim = env.observation_space.shape[0]  # State dimension
action_dim = int(env.action_space.n)  # Number of available actions

# PPO Parameters
camadas = [state_dim, 128, action_dim]  # Example network structure
diminuicao_nota = 0.99  # Discount factor for rewards
qtd_acoes_afetadas_pela_recompensa = 5
vel_aprendizagem = 0.01
limite_concordancia = 0.2

# Instantiate policy
politica = Politica(
    camadas,
    diminuicao_nota,
    qtd_acoes_afetadas_pela_recompensa,
    vel_aprendizagem,
    limite_concordancia,
)

# Training hyperparameters
n_episodios = 500
max_steps = 200  # Maximum steps per episode
epsilon = 0.2  # Used for clipping (not directly implemented here)

# Accumulated reward function
def calcular_recompensa_acumulada(recompensas, gamma):
    G = 0
    recompensas_acumuladas = []
    for r in reversed(recompensas):
        G = r + gamma * G
        recompensas_acumuladas.insert(0, G)
    return recompensas_acumuladas

# Training
for episodio in range(n_episodios):
    estado, _ = env.reset()
    recompensas = []
    acoes = []
    estados = []
    total_recompensa = 0

    for t in range(max_steps):
        # Convert state to tensor
        estado_tensor = [torch.tensor(s, dtype=torch.float32, requires_grad=True) for s in estado]

        # Select action using current policy
        acao = politica(estado_tensor)
        acoes.append(acao)

        # Execute action in environment
        novo_estado, recompensa, terminado, _, _ = env.step(acao)
        recompensas.append(recompensa)
        estados.append(estado_tensor)
        total_recompensa += recompensa

        # Update state
        estado = novo_estado

        if terminado:
            break

    # Calculate accumulated rewards
    recompensas_acumuladas = calcular_recompensa_acumulada(recompensas, diminuicao_nota)

    # Update policy based on rewards
    for idx, (recompensa, estado, acao) in enumerate(zip(recompensas_acumuladas, estados, acoes)):
        politica.premiar(recompensa, index_acao_a_premiar=idx, premio_esperado=9*(episodio+1)/100)

    politica.atualizar()

    # Feedback
    print(f"Episódio {episodio + 1}/{n_episodios}, Recompensa Total: {total_recompensa}")

print("Treinamento concluído!")
for param in politica.pol_atual.parameters():
    print(param)
# Close environment and ensure videos are saved properly
env.close()