import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from rl_utils import TransitionDataset
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from deep_sets import DeepSet
from attention import MLP, Attention

class DeepSetPolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, representation_dim = 256):
        super(DeepSetPolicyNet, self).__init__()
        self.deep_set = DeepSet(state_dim, representation_dim, representation_dim)
        self.mlp = MLP(state_dim + representation_dim, hidden_dim, action_dim)

    def forward(self, x):
        if len(x.shape) > 2: # if input is a batch
            set_representation = self.deep_set(x[:,1:,:]) # the first line is always the ego vehicle
            return F.softmax(self.mlp(torch.cat((x[:,0,:], set_representation), 1)), dim = 1)
            # Deep set does not learn to discern ego vehicle and the rest, unlike social attention
            # Hence we only embed the set of other vehicles, excluding the ego vehicle 
            # Then concatenate the state of ego vehicle and the set representation
        else:
            set_representation = self.deep_set(x[1:,:]) 
            return F.softmax(self.mlp(torch.cat((x[0,:], set_representation), 0)), dim = 0)
        

class SocialAttentionPolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(SocialAttentionPolicyNet, self).__init__()
        self.attn = Attention(state_dim, state_dim).to(device)
        self.MLP = MLP(self.attn.embed_dim, hidden_dim, action_dim).to(device)

    def forward(self, x):
        if len(x.shape) > 2:
            x = self.attn(x[:,0,:], x[:,:,:])
            d = 1
        else:
            x = self.attn(x[0], x[:])
            d = 0

        return F.softmax(self.MLP(x), dim=d)
    

class BehaviorClone(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device, focal_loss = True, gamma = 5, soft = False):
        super(BehaviorClone, self).__init__()
        # self.policy = DeepSetPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.policy = SocialAttentionPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.device = device
        self.focal_loss = focal_loss
        self.gamma = gamma
        self.soft = soft

    def learn(self, states, actions):
        states = states.to(self.device)
        actions = actions.view(-1, 1).to(self.device)
        log_probs = torch.log(self.policy(states).gather(1, actions)) 
        if self.focal_loss:
            bc_loss = torch.mean(-log_probs * torch.pow(1 - torch.exp(log_probs), self.gamma))
        else:
            bc_loss = torch.mean(-log_probs) # maximum log-likelihood <=> minimum cross-entropy loss
        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()
        return bc_loss.detach().item()

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.policy(state)
        print(probs)
        if self.soft:
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        else:
            action = torch.argmax(probs) # we are doing BC here, let's make hard decisions
        return action.item()


def test_agent(agent, env, n_episode):
    return_list = []
    for _ in range(n_episode):
        episode_return = 0
        state, _ = env.reset()
        done = truncated = False
        while not (done or truncated):
            action = agent.take_action(state)
            print(action)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_return += reward
            env.render()
        return_list.append(episode_return)
    return np.mean(return_list)

if __name__ == '__main__':

    env = gym.make('highway-v0', render_mode = 'rgb_array')
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 40,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False
        }
    }
    env.configure(config)
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = 5
    hidden_dim = [256,]
    action_dim = 5
    lr = 1e-3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr, device)
    # state_dict = torch.load("./model/bc_attn_fl5_r17.03.pt")
    # bc_agent.load_state_dict(state_dict)
    n_iterations = 500
    batch_size = 64
    test_returns = []
    
    dataset = TransitionDataset('transition_data_mc2.pkl')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    iterator = list(iter(dataloader))

    with tqdm(total=n_iterations, desc="Progress Bar") as pbar:
        for i in range(n_iterations):
            loss = []
            for batch in iterator:
                expert_s, expert_a = batch['state'], batch['action']
                loss.append(bc_agent.learn(expert_s, expert_a))
            pbar.set_postfix({'scaled average loss': '%.3f' % (10 * np.mean(loss))})
            if (10 * np.mean(loss)) < 0.05:
                break # Early stopping
            if (i + 1) % 10 == 0:
                bc_agent.scheduler.step()
            if (i + 1) % 100 == 0 and i + 1 < n_iterations:  
                iterator = list(iter(dataloader))      
            pbar.update(1)
    
            
    print("Average return: {}".format(test_agent(bc_agent, env, 50)))

    torch.save(bc_agent.state_dict(), "./model/bc_attn_fl5_new2.pt")
