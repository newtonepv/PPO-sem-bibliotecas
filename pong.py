import pygame
import sys
import random
import math
from copy import deepcopy
from politicas import Politica


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

#coisas de ia
SEGUNDOS_PREMIADOS_ANTES_DA_ACAO = 6
ACOES_POR_SEGUNDO = 10
POLITICA = Politica(
    camadas = [8,64,3],
    diminuicao_nota= 0.93,
    qtd_acoes_afetadas_pela_recompensa=SEGUNDOS_PREMIADOS_ANTES_DA_ACAO*ACOES_POR_SEGUNDO,
    vel_aprendizagem=0.01,
    limite_concordancia=0.2
)
PREMIO_TOQUE = 1
PREMIO_PONTO = 1.5

#initializing variables
frames_sem_acao = 0
ag_1 = deepcopy(POLITICA)
ag_2 = deepcopy(POLITICA)
index_acao_antes_ultima_batida_1:int = -1
index_acao_antes_ultima_batida_2:int = -1


# Initialize Pygame
pygame.init()


vel_sim = 1
# Constants
FPS = 60
WIDTH = 800
HEIGHT = 600
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 90
BALL_SIZE = 15
paddle_speed = 5*vel_sim
init_ball_speed = 2*vel_sim
max_ball_speed = 15*vel_sim
speed_increase = 0.5*vel_sim
MAX_ANGLE = 75
RANDOM_ANGLE_RANGE = 15  # Maximum degrees of random variation

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def atualizar_velocidades(vel_sim):
    return 5*vel_sim,2*vel_sim,15*vel_sim,0.5*vel_sim

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Two-Player Pong")

# Create game objects
player1 = pygame.Rect(50, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
player2 = pygame.Rect(WIDTH - 50 - PADDLE_WIDTH, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
ball = pygame.Rect(WIDTH//2 - BALL_SIZE//2, HEIGHT//2 - BALL_SIZE//2, BALL_SIZE, BALL_SIZE)

# Paddle movement states
player1_movement = 0  # -1 for up, 0 for stay, 1 for down
player2_movement = 0

# Ball velocity components
current_ball_speed = init_ball_speed
ball_dx = init_ball_speed * random.choice((1,0.5,-0.5 -1))
ball_dy = init_ball_speed * random.choice((1,0.5,-0.5 -1))

# Score
player1_score = 0
player2_score = 0
font = pygame.font.Font(None, 36)

def calculate_new_velocity(ball_rect, paddle_rect, current_dx, speed):
    """Calculate new ball velocity based on where it hits the paddle with random variation"""
    relative_intersect_y = (paddle_rect.centery - ball_rect.centery)
    normalized_intersect = relative_intersect_y / (PADDLE_HEIGHT/2)
    
    # Calculate base bounce angle and add random variation
    base_bounce_angle = normalized_intersect * MAX_ANGLE
    random_variation = random.uniform(-RANDOM_ANGLE_RANGE, RANDOM_ANGLE_RANGE)
    bounce_angle = base_bounce_angle + random_variation
    
    # Clamp the final angle to prevent too extreme angles
    bounce_angle = max(min(bounce_angle, MAX_ANGLE), -MAX_ANGLE)
    
    bounce_angle_radians = math.radians(bounce_angle)
    
    new_dx = -current_dx
    new_dy = -speed * math.sin(bounce_angle_radians)
    
    # Normalize the velocity to maintain consistent speed
    velocity_length = math.sqrt(new_dx**2 + new_dy**2)
    new_dx = (new_dx/velocity_length) * speed
    new_dy = (new_dy/velocity_length) * speed
    
    # Add slight random variation to the speed itself
    speed_variation = random.uniform(0.95, 1.05)
    new_dx *= speed_variation
    new_dy *= speed_variation
    
    return new_dx, new_dy

def reset_ball():
    """Reset ball to center with initial speed and random direction"""
    global current_ball_speed, ball_dx, ball_dy
    current_ball_speed = init_ball_speed
    ball.center = (WIDTH // 2, HEIGHT // 2)
    
    # Define random speed components ensuring vx ≠ 0
    ball_dx = random.uniform(0.5, 1.0) * current_ball_speed * random.choice((-1, 1))
    ball_dy = random.uniform(0.5, 1.0) * current_ball_speed * random.choice((-1, 1))
    
    # Normalize the velocity to ensure a consistent speed
    velocity_length = math.sqrt(ball_dx ** 2 + ball_dy ** 2)
    ball_dx = (ball_dx / velocity_length) * current_ball_speed
    ball_dy = (ball_dy / velocity_length) * current_ball_speed





# Game loop
while True:
    random.seed()  # Inicializa o gerador de números aleatórios com base no tempo atual

    frames_sem_acao+=1
    if(frames_sem_acao%int(FPS/ACOES_POR_SEGUNDO)==0):
        #tomar acao
        estado:list = [player1.x, player1.y, player2.x, player2.y, ball.x, ball.y, ball_dx, ball_dy]

        acao_1 = ag_1(estado)
        player1_movement = acao_1-1 #agora vai de -1,1 no lugar de ir de 0,2  

        acao_2 = ag_2(estado)
        player2_movement = acao_2-1 


    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_UP:
                vel_sim+=1
                paddle_speed,init_ball_speed,max_ball_speed,speed_increase=atualizar_velocidades(vel_sim)
                print("vel sim = "+str(vel_sim))
            elif event.key == pygame.K_DOWN:
                vel_sim-=1
                paddle_speed,init_ball_speed,max_ball_speed,speed_increase=atualizar_velocidades(vel_sim)
                print("vel sim = "+str(vel_sim))

    
    # Apply paddle movement based on states
    if player1_movement == -1 and player1.top > 0:
        player1.y -= paddle_speed
    elif player1_movement == 1 and player1.bottom < HEIGHT:
        player1.y += paddle_speed

    if player2_movement == -1 and player2.top > 0:
        player2.y -= paddle_speed
    elif player2_movement == 1 and player2.bottom < HEIGHT:
        player2.y += paddle_speed
    
    # Ball movement
    ball.x += ball_dx
    ball.y += ball_dy
    
    # Ball collisions with top and bottom
    if ball.top <= 0:
        ball.top = 0
        ball_dy = abs(ball_dy)
    elif ball.bottom >= HEIGHT:
        ball.bottom = HEIGHT
        ball_dy = -abs(ball_dy)
    

    # Scoring
    if ball.left <= 0: #PONTO2
        diminuicao = -PREMIO_PONTO*((math.sqrt((ball.y-player1.y)**2)/HEIGHT)+1)
        ag_1.premiar(diminuicao)#penalizar jogador 1
        
        if(index_acao_antes_ultima_batida_2!=-1):
            recompensa = PREMIO_PONTO*((math.sqrt((ball.y-player1.y)**2)/HEIGHT)+1)#enquanto mais longe estiver mais ponto ganha
            ag_2.premiar(recompensa,index_acao_antes_ultima_batida_2)

        player2_score += 1
        reset_ball()
        #carregar nova rede
        ag_1.atualizar()
        ag_2.atualizar()
        index_acao_antes_ultima_batida_1=-1
        index_acao_antes_ultima_batida_2=-1
    
    if ball.right >= WIDTH: #PONTO1
        diminuicao = -PREMIO_PONTO*((math.sqrt((ball.y-player2.y)**2)/HEIGHT)+1)
        ag_2.premiar(diminuicao)

        if(index_acao_antes_ultima_batida_1 != -1):#so ganha ponto se tiver batido
            recompensa = PREMIO_PONTO*((math.sqrt((ball.y-player2.y)**2)/HEIGHT)+1)
            ag_1.premiar(recompensa, index_acao_antes_ultima_batida_1)

        player1_score += 1
        reset_ball()
        #carregar nova rede
        ag_1.atualizar()
        ag_2.atualizar()
        index_acao_antes_ultima_batida_1=-1
        index_acao_antes_ultima_batida_2=-1


    # Paddle collisions with speed increase
    if ball.colliderect(player1): #TOQUE1
        index_acao_antes_ultima_batida_1 = ag_1.get_index_ultima_acao()
        ag_1.premiar(PREMIO_TOQUE)

        ball.left = player1.right
        current_ball_speed = min(current_ball_speed + speed_increase, max_ball_speed)
        ball_dx, ball_dy = calculate_new_velocity(ball, player1, ball_dx, current_ball_speed)
        
        
    if ball.colliderect(player2):#TOQUE2
        index_acao_antes_ultima_batida_2 = ag_2.get_index_ultima_acao()
        ag_2.premiar(PREMIO_TOQUE)

        ball.right = player2.left
        current_ball_speed = min(current_ball_speed + speed_increase, max_ball_speed)
        ball_dx, ball_dy = calculate_new_velocity(ball, player2, ball_dx, current_ball_speed)       


    # Drawing
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, player1)
    pygame.draw.rect(screen, WHITE, player2)
    pygame.draw.ellipse(screen, WHITE, ball)
    pygame.draw.aaline(screen, WHITE, (WIDTH//2, 0), (WIDTH//2, HEIGHT))
    
    # Score and speed display
    score_text = font.render(f"{player1_score} - {player2_score}", True, WHITE)
    speed_text = font.render(f"Speed: {current_ball_speed:.1f}", True, WHITE)
    screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, 20))
    screen.blit(speed_text, (WIDTH//2 - speed_text.get_width()//2, 50))
    
    #print(recompensas_1)
    pygame.display.flip()
