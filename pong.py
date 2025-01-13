import pygame
import sys
import random
import math
from backward_functioning import Valor, MLP
from copy import deepcopy

def index_da_acao_tomada(lista: list[Valor]) -> int:
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



#coisas de ia
APRENDIZADO = 0.01
LIMITE_RACIO = 1.2

def pegar_racio_e_acao(estado: list, rede_nova:MLP, rede_velha:MLP)-> tuple[Valor, int]:
        resposta_da_rede_nova = rede_nova(estado)
        resposta_da_rede_velha = rede_velha(estado)

        acao_tomada = index_da_acao_tomada(resposta_da_rede_nova)

        chance_da_nova_tomar_acao = resposta_da_rede_nova[acao_tomada]
        chance_da_velha_tomar_acao = resposta_da_rede_velha[acao_tomada]

        return (chance_da_nova_tomar_acao/chance_da_velha_tomar_acao,acao_tomada)

def carregar_nova_rede(rede_nova:MLP, rede_velha:MLP, concordancia:list[Valor], recompensas:list[float], recompensas_esperadas:list[float]):
    rede_velha = deepcopy(rede_nova)

    assert(len(concordancia)==len(recompensas)==len(recompensas_esperadas))

    for c,r,re in zip(concordancia,recompensas,recompensas_esperadas):

        if(c.valor_numerico>LIMITE_RACIO):
            c.valor_numerico = c.valor_numerico-(c.valor_numerico-LIMITE_RACIO)
        elif(c.valor_numerico<-LIMITE_RACIO):
            c.valor_numerico = c.valor_numerico+((-LIMITE_RACIO)-c.valor_numerico)

        nova_perda:Valor=c*(r-re)
        nova_perda.derivadas()

        for p in rede_nova.parametros():
            p.valor_numerico = p.valor_numerico+(APRENDIZADO*p.gradiente)
            p.gradiente = 0

    concordancia.clear()
    recompensas.clear()
    recompensas_esperadas.clear()
    

def adicionar_recompensa(recompensa:float, index_acao_premiada:int, vetor_premios:list):
    for i in range(SEGUNDOS_PREMIADOS_ANTES_DA_ACAO*ACOES_POR_SEGUNDO):

        index = (index_acao_premiada)-i
        if(index < 0):
            break

        vetor_premios[index] += recompensa*(DIMINUICAO_DO_PREMIO**i)

SEGUNDOS_PREMIADOS_ANTES_DA_ACAO = 6
ACOES_POR_SEGUNDO = 10
DIMINUICAO_DO_PREMIO = 0.93
PREMIO_TOQUE = 1
PREMIO_PONTO = 10
ARQUITETURA = [8,64,3]

#initializing variables
frames_sem_acao = 0

rede_nova_1 = MLP(ARQUITETURA)
rede_velha_1 = deepcopy(rede_nova_1)

index_acao_antes_ultima_batida_1:int = -1
recompensas_esperadas_1 = []
recompensas_1 = []
concordancia_1 = []

rede_nova_2 = MLP(ARQUITETURA)
rede_velha_2 = deepcopy(rede_nova_2)

index_acao_antes_ultima_batida_2:int = -1
recompensas_esperadas_2 = []
recompensas_2 = []
concordancia_2 = []


# Initialize Pygame
pygame.init()

# Constants
FPS = 60
WIDTH = 800
HEIGHT = 600
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 90
BALL_SIZE = 15
PADDLE_SPEED = 5
INITIAL_BALL_SPEED = 2
MAX_BALL_SPEED = 15
SPEED_INCREASE = 0.5
MAX_ANGLE = 75
RANDOM_ANGLE_RANGE = 15  # Maximum degrees of random variation

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

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
current_ball_speed = INITIAL_BALL_SPEED
ball_dx = INITIAL_BALL_SPEED * random.choice((1, -1))
ball_dy = INITIAL_BALL_SPEED * random.choice((1, -1))

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
    """Reset ball to center with initial speed"""
    global current_ball_speed, ball_dx, ball_dy
    current_ball_speed = INITIAL_BALL_SPEED
    ball.center = (WIDTH//2, HEIGHT//2)
    ball_dx = INITIAL_BALL_SPEED * random.choice((1, -1))
    ball_dy = INITIAL_BALL_SPEED * random.choice((1, -1))





# Game loop
clock = pygame.time.Clock()
while True:
    frames_sem_acao+=1
    if(frames_sem_acao%int(FPS/ACOES_POR_SEGUNDO)==0):

        recompensas_1.append(0)
        recompensas_esperadas_1.append(0)
        recompensas_2.append(0)
        recompensas_esperadas_2.append(0)

        #tomar acao
        estado:list = [player1.x, player1.y, player2.x, player2.y, ball.x, ball.y, ball_dx, ball_dy]

        racio_1, acao_1 = pegar_racio_e_acao(estado,rede_nova_1,rede_velha_1)
        concordancia_1.append(racio_1)
        player1_movement = acao_1-1 #agora vai de -1,1 no lugar de ir de 0,2  

        racio_2, acao_2 = pegar_racio_e_acao(estado,rede_nova_2,rede_velha_2)
        print(round(racio_2.valor_numerico,2), end=",", flush=True)
        concordancia_2.append(racio_2)
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

    
    # Apply paddle movement based on states
    if player1_movement == -1 and player1.top > 0:
        player1.y -= PADDLE_SPEED
    elif player1_movement == 1 and player1.bottom < HEIGHT:
        player1.y += PADDLE_SPEED

    if player2_movement == -1 and player2.top > 0:
        player2.y -= PADDLE_SPEED
    elif player2_movement == 1 and player2.bottom < HEIGHT:
        player2.y += PADDLE_SPEED
    
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
        adicionar_recompensa(diminuicao,len(recompensas_1)-1,recompensas_1)#penalizar jogador 1
        if(index_acao_antes_ultima_batida_2!=-1):
            recompensa = PREMIO_PONTO*((math.sqrt((ball.y-player1.y)**2)/HEIGHT)+1)#enquanto mais longe estiver mais ponto ganha
            adicionar_recompensa(recompensa,index_acao_antes_ultima_batida_2,recompensas_2)

        player2_score += 1
        reset_ball()
        #carregar nova rede
        print(len(recompensas_2))
        print(len(recompensas_1))
        carregar_nova_rede(rede_nova_1,rede_velha_1,concordancia_1,recompensas_1, recompensas_esperadas_1)
        carregar_nova_rede(rede_nova_2,rede_velha_2,concordancia_2,recompensas_2, recompensas_esperadas_2)
        index_acao_antes_ultima_batida_1=-1
        index_acao_antes_ultima_batida_2=-1
    
    if ball.right >= WIDTH: #PONTO1
        diminuicao = -PREMIO_PONTO*((math.sqrt((ball.y-player2.y)**2)/HEIGHT)+1)
        adicionar_recompensa(diminuicao,len(recompensas_2)-1,recompensas_2)#penalizar jogador 1
        if(index_acao_antes_ultima_batida_1 != -1):#so ganha ponto se tiver batido
            recompensa = PREMIO_PONTO*((math.sqrt((ball.y-player2.y)**2)/HEIGHT)+1)
            adicionar_recompensa(recompensa,index_acao_antes_ultima_batida_1,recompensas_1)#beneficiar jogador 1


        player1_score += 1
        reset_ball()
        #carregar nova rede
        print(len(recompensas_2))
        print(len(recompensas_1))
        carregar_nova_rede(rede_nova_1,rede_velha_1,concordancia_1,recompensas_1, recompensas_esperadas_1)
        carregar_nova_rede(rede_nova_2,rede_velha_2,concordancia_2,recompensas_2, recompensas_esperadas_2)
        index_acao_antes_ultima_batida_1=-1
        index_acao_antes_ultima_batida_2=-1


    # Paddle collisions with speed increase
    if ball.colliderect(player1): #TOQUE1
        index_acao_antes_ultima_batida_1 = len(recompensas_1)-1
        adicionar_recompensa(PREMIO_TOQUE,index_acao_antes_ultima_batida_1,recompensas_1)

        ball.left = player1.right
        current_ball_speed = min(current_ball_speed + SPEED_INCREASE, MAX_BALL_SPEED)
        ball_dx, ball_dy = calculate_new_velocity(ball, player1, ball_dx, current_ball_speed)
        
        
    if ball.colliderect(player2):#TOQUE2
        index_acao_antes_ultima_batida_2 = len(recompensas_2)-1
        adicionar_recompensa(PREMIO_TOQUE,index_acao_antes_ultima_batida_2, recompensas_2)

        ball.right = player2.left
        current_ball_speed = min(current_ball_speed + SPEED_INCREASE, MAX_BALL_SPEED)
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
    clock.tick(FPS)