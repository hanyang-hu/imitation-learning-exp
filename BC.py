import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from rl_utils import TransitionDataset
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt
from deep_sets import MLP, DeepSet
from attention import Attention

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
    def __init__(self, state_dim, hidden_dim, action_dim, ego_dim = 5, oppo_dim = 5):
        super(SocialAttentionPolicyNet, self).__init__()
        self.attn = Attention(ego_dim, oppo_dim)
        layer1 = self.attn.embed_dim
        layer_dim = [layer1,] + hidden_dim + [action_dim,]
        self.fc = torch.nn.ParameterList([torch.nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(hidden_dim))])

    def forward(self, x):
        if len(x.shape) > 2:
            x = self.attn(x[:,0,:], x[:,:,:])
            d = 1
        else:
            x = self.attn(x[0], x[:]) # the first line is always the ego vehicle
            d = 0

        for layer in self.fc:
            x = F.relu(layer(x))

        return F.softmax(x, dim = d)
    

class BehaviorClone(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device, focal_loss = True, gamma = 5, soft = False):
        super(BehaviorClone, self).__init__()
        # self.policy = DeepSetPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.policy = SocialAttentionPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
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
        return bc_loss

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.policy(state)
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
        },
        "manual_control": True
    }
    env.configure(config)
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = 5
    hidden_dim = [256, 256]
    action_dim = 5
    lr = 1e-3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr, device)
    n_iterations = 50
    batch_size = 64
    test_returns = []

    dataset = TransitionDataset('transition_data_mc.pkl')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    with tqdm(total=n_iterations, desc="Progress Bar") as pbar:
        for i in range(n_iterations):
            loss = []
            for batch in dataloader:
                expert_s, expert_a = batch['state'], batch['action']
                loss.append(bc_agent.learn(expert_s, expert_a))
            print(sum(loss) / len(loss))
            """
            if i > n_iterations * 0.7:
                current_return = test_agent(bc_agent, env, 3)
                test_returns.append(current_return)
                if (i + 1) % 2 == 0:
                    pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-2:])})
            """
            pbar.update(1)
            bc_agent.scheduler.step()

    print("Average return: {}".format(test_agent(bc_agent, env, 10)))

    torch.save(bc_agent.state_dict(), "./model/bc_attn_fl5.pt")
