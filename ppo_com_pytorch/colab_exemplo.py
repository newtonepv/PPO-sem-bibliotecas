import time
import random
import numpy as np
import matplotlib.pylab as plt
plt.style.use('dark_background')
from configuracoes import config_colabe
import gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical

seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

configs = config_colabe

run_name = f"{configs.gym_id}__{configs.exp_name}__{seed}__{int(time.time())}"

# create an env with random state 
def make_env_func(gym_id, seed, idx, run_name, capture_video=False):
    def env_fun():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            # initiate the video capture if not already initiated
            if idx == 0: 
                # wrapper to create the video of the performance
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return env_fun

# create N envs
envs = []
for i in range(configs.num_trajcts):
    envs.append( make_env_func(configs.gym_id, seed + i, i, run_name) )
envs = gym.vector.SyncVectorEnv(envs)
envs

class FCBlock(nn.Module):
    """A generic fully connected residual block with good setup"""
    def __init__(self, embd_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(embd_dim),
            nn.GELU(),
            nn.Linear(embd_dim, 4*embd_dim),
            nn.GELU(),
            nn.Linear(4*embd_dim, embd_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x)


class Agent(nn.Module):
    """an agent that creates actions and estimates values"""
    def __init__(self, env_observation_dim, action_space_dim, embd_dim=64, num_blocks=2):
        super().__init__()
        self.embedding_layer = nn.Linear(env_observation_dim, embd_dim)
        self.shared_layers = nn.Sequential(*[FCBlock(embd_dim=embd_dim) for _ in range(num_blocks)])
        self.value_head = nn.Linear(embd_dim, 1)
        self.policy_head = nn.Linear(embd_dim, action_space_dim)
        # orthogonal initialization with a hi entropy for more exploration at the start
        torch.nn.init.orthogonal_(self.policy_head.weight, 0.01)

    def value_func(self, state):
        hidden = self.shared_layers(self.embedding_layer(state))
        value = self.value_head(hidden)
        return value

    def policy(self, state, action=None)->tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.shared_layers(self.embedding_layer(state))
        logits = self.policy_head(hidden)
        # PyTorch categorical class helpful for sampling and probability calculations
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.value_head(hidden)
    
agent = Agent(
    env_observation_dim=envs.single_observation_space.shape[0],
    action_space_dim=envs.single_action_space.n
).to(device)

optimizer = optim.Adam(agent.parameters(), lr=configs.learning_rate)

def gae(
    cur_observation,  # the current state when advantages will be calculated
    rewards,          # rewards collected from trajectories of shape [num_trajcts, max_trajects_length]
    dones,            # binary marker of end of trajectories of shape [num_trajcts, max_trajects_length]
    values            # value estimates collected over trajectories of shape [num_trajcts, max_trajects_length]
):
    """
    Generalized Advantage Estimation (gae) estimating advantage of a particular trajecotry
    vs the expected return starting from a state
    """
    advantages = torch.zeros((configs.num_trajcts, configs.max_trajects_length))
    last_advantage = 0
    
    # the value after the last step
    with torch.no_grad(): 
        last_value = agent.value_func(cur_observation).reshape(1, -1)

    # reverse recursive to calculate advantages based on the delta formula
    for t in reversed(range(configs.max_trajects_length)):
        # mask if episode completed after step t
        mask = 1.0 - dones[:, t]
        last_value = last_value * mask
        last_advantage = last_advantage * mask
        delta = rewards[:, t] + configs.gamma * last_value - values[:, t]
        last_advantage = delta + configs.gamma * configs.gae_lambda * last_advantage
        advantages[:, t] = last_advantage
        last_value = values[:, t]
        
    advantages = advantages.to(device)
    returns = advantages + values
    
    return advantages, returns

def create_rollout(
    envs,            # parallel envs creating trajectories
    cur_observation, # starting observation of shape [num_trajcts, observation_dim]
    cur_done,        # current termination status of shape [num_trajcts,]
    all_returns      # a list to track returns
):
    """
    rollout phase: create parallel trajectories and store them in the rollout storage
    """
    
    # cache empty tensors to store the rollouts 
    observations = torch.zeros((configs.num_trajcts, configs.max_trajects_length) +
                               envs.single_observation_space.shape).to(device)
    actions = torch.zeros((configs.num_trajcts, configs.max_trajects_length) +
                          envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)
    rewards = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)
    dones = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)
    values = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)
    
    for t in range(configs.max_trajects_length):
        observations[:,t] = cur_observation
        dones[:,t] = cur_done

        # give observation to the model and collect action, logprobs of actions, entropy and value
        with torch.no_grad():
            action, logprob, _, value = agent.policy(cur_observation)
        values[:,t] = value.flatten()
        actions[:,t] = action
        logprobs[:,t] = logprob

        # apply the action to the env and collect observation and reward
        cur_observation, reward, cur_done, _, info = envs.step(action.cpu().numpy())
        rewards[:,t] = torch.tensor(reward).to(device).view(-1)
        cur_observation = torch.Tensor(cur_observation).to(device)
        cur_done = torch.Tensor(cur_done).to(device)
        
        # if an episode ended store its total reward for progress report
        if info: 
            for item in info['final_info']:
                if item and "episode" in item.keys():
                    all_returns.append(item['episode']['r'])
                    break
    
    # create the rollout storage
    rollout = {
        'cur_observation': cur_observation,
        'cur_done': cur_done,
        'observations': observations,
        'actions': actions,
        'logprobs': logprobs,
        'values': values,
        'dones': dones,
        'rewards': rewards
    }
    
    return rollout

class Storage(Dataset):
    def __init__(self, rollout, advantages, returns, envs):
        # fill in the storage and flatten the parallel trajectories
        self.observations = rollout['observations'].reshape((-1,) + envs.single_observation_space.shape)
        self.logprobs = rollout['logprobs'].reshape(-1)    
        self.actions = rollout['actions'].reshape((-1,) + envs.single_action_space.shape).long()
        self.advantages = advantages.reshape(-1)
        self.returns = returns.reshape(-1)

    def __getitem__(self, ix: int):
        item = [
            self.observations[ix],
            self.logprobs[ix],
            self.actions[ix],
            self.advantages[ix],
            self.returns[ix]
        ]
        return item

    def __len__(self) -> int:
        return len(self.observations)
    
def loss_clip(
    mb_oldlogporb,     # old logprob of mini batch actions collected during the rollout
    mb_newlogprob,     # new logprob of mini batch actions created by the new policy
    mb_advantages      # mini batch of advantages collected during the the rollout
)-> torch.Tensor:
    """
    policy loss with clipping to control gradients
    """
    ratio = torch.exp(mb_newlogprob - mb_oldlogporb)
    policy_loss = -mb_advantages * ratio
    # clipped policy gradient loss enforces closeness
    clipped_loss = -mb_advantages * torch.clamp(ratio, 1 - configs.clip_epsilon, 1 + configs.clip_epsilon)
    pessimistic_loss = torch.max(policy_loss, clipped_loss).mean()
    return pessimistic_loss


def loss_vf(
    mb_oldreturns,  # mini batch of old returns collected during the rollout
    mb_newvalues    # minibach of values calculated by the new value function    
)->torch.Tensor:
    """
    enforcing the value function to give more accurate estimates of returns
    """
    mb_newvalues = mb_newvalues.view(-1)
    loss = 0.5 * ((mb_newvalues - mb_oldreturns) ** 2).mean()
    return loss

# track returns
all_returns = []

# initialize the game
cur_observation = torch.Tensor(envs.reset()[0]).to(device)
cur_done = torch.zeros(configs.num_trajcts).to(device)

# progress bar
num_updates = configs.total_timesteps // configs.batch_size

for update in range(1, num_updates + 1):
    
    
    ##############################################
    # Phase 1: rollout creation 
    
    # parallel envs creating trajectories
    rollout = create_rollout(envs, cur_observation, cur_done, all_returns)
    
    cur_done = rollout['cur_done']
    cur_observation = rollout['cur_observation']
    rewards = rollout['rewards']
    dones = rollout['dones']
    values = rollout['values']
    
    # calculating advantages
    advantages, returns = gae(cur_observation, rewards, dones, values)

    # a dataset containing the rollouts
    dataset = Storage(rollout, advantages, returns, envs)
    
    # a standard dataloader made out of current storage
    trainloader = DataLoader(dataset, batch_size=configs.minibatch_size, shuffle=True)
    
    
    ##############################################
    # Phase 2: model update 
    
    # linearly shrink the lr from the initial lr to zero
    frac = 1.0 - (update - 1.0) / num_updates
    optimizer.param_groups[0]["lr"] = frac * configs.learning_rate
    # training loop
    for epoch in range(configs.updates_no_mesmo_rollout):
        for batch in trainloader:
            mb_observations, mb_logprobs, mb_actions, mb_advantages, mb_returns = batch
            
            # we calculate the distribution of actions through the updated model revisiting the old trajectories  
            _, mb_newlogprob, mb_entropy, mb_newvalues = agent.policy(mb_observations, mb_actions)
            
            policy_loss = loss_clip(mb_logprobs, mb_newlogprob, mb_advantages)
            
            value_loss = loss_vf(mb_returns, mb_newvalues)

            # average entory of the action space
            entropy_loss = mb_entropy.mean()
            
            # full weighted loss
            loss = policy_loss - configs.ent_coef * entropy_loss + configs.vf_coef * value_loss 

            optimizer.zero_grad()
            loss.backward()
            
            # extra clipping of the gradients to avoid overshoots
            nn.utils.clip_grad_norm_(agent.parameters(), configs.max_grad_norm)
            optimizer.step()
    


envs.close()

if not len(all_returns)%configs.num_episodes_to_average==0:
    all_returns_truncated = np.array(all_returns[:-(len(all_returns)%configs.num_episodes_to_average)])
else:
    all_returns_truncated = all_returns
all_returns_smoothed = np.average(all_returns_truncated.reshape(-1, configs.num_episodes_to_average), axis=1)
print('mean reward:', np.mean(all_returns_smoothed))
print('std reward:', np.std(all_returns_smoothed))
print('max reward:', np.max(all_returns_smoothed))
print('converge mean reward:', np.mean(all_returns_smoothed[-1]))
plt.plot(all_returns_smoothed)