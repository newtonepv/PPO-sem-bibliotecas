from politicas import Politica
import random
import math
import matplotlib.pyplot as plt

# Simple environment: Agent needs to choose larger of 2 numbers
def ambiente(acao: int, estado: list[float]) -> tuple[float, list[float]]:
    # Reward is positive if agent picks larger number, negative otherwise
    recompensa = 1.0 if acao == (1 if estado[1] > estado[0] else 0) else -1.0
    
    # Generate new state (2 random numbers)
    novo_estado = [random.uniform(-1, 1), random.uniform(-1, 1)]
    return recompensa, novo_estado

def plotar_recompensas(historico_recompensas):
    """Plot the rewards history"""
    plt.figure(figsize=(10, 5))
    
    # Raw data
    plt.plot(historico_recompensas, alpha=0.3, color='blue', label='Raw')
    
    # Smoothed data
    window = 100
    smoothed = [sum(historico_recompensas[max(0,i-window):i])/min(i,window) 
                for i in range(1,len(historico_recompensas)+1)]
    plt.plot(smoothed, color='red', label='Smoothed')
    
    plt.title('Recompensas por Episódio')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    plt.grid(True)
    plt.legend()
    
    # Save and show
    plt.savefig('training_rewards.png')
    plt.show()

def main():
    # Initialize policy (2 inputs, 3 hidden, 2 outputs)
    politica = Politica(
        camadas=[2, 3, 2],  # Input -> Hidden -> Output
        diminuicao_nota=0.99,
        qtd_acoes_afetadas_pela_recompensa=5,
        vel_aprendizagem=0.01,
        limite_concordancia=0.2
    )

    # Training loop
    estado = [random.uniform(-1, 1), random.uniform(-1, 1)]
    episodios = 1000
    
    recompensa_total_array=[]

    for ep in range(episodios):
        # Run one episode
        recompensa_total = 0
        for _ in range(10):  # 10 steps per episode
            # Get action from policy
            acao = politica(estado)
            
            # Environment step
            recompensa, novo_estado = ambiente(acao, estado)
            
            # Update
            politica.premiar(recompensa)
            recompensa_total += recompensa
            estado = novo_estado
            
        # Update policy at end of episode
        politica.atualizar()
        
        if ep % 10 == 0:
            print(f"Episódio {ep}, Recompensa: {recompensa_total}",flush=True)

        recompensa_total_array.append(recompensa_total)

    plotar_recompensas(recompensa_total_array)
if __name__ == "__main__":
    main()