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
    

class BehaviorClone:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device, focal_loss = False, gamma = 1.5):
        self.policy = DeepSetPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.device = device
        self.focal_loss = focal_loss
        self.gamma = gamma

    def learn(self, states, actions):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1).to(self.device)
        log_probs = torch.log(self.policy(states).gather(1, actions)) 
        if self.focal_loss:
            bc_loss = torch.mean(-log_probs * torch.pow(1 - torch.exp(log_probs), self.gamma))
        else:
            bc_loss = torch.mean(-log_probs) # maximum log-likelihood <=> minimum cross-entropy loss

        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


def test_agent(agent, env, n_episode):
    return_list = []
    for _ in range(n_episode):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
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
    env.config["duration"] = 60
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = 5
    hidden_dim = [256, 256]
    action_dim = 5
    lr = 1e-3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr, device)
    n_iterations = 100
    batch_size = 64
    test_returns = []

    dataset = TransitionDataset('transition_data.pkl')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    with tqdm(total=n_iterations, desc="Progress Bar") as pbar:
        for i in range(n_iterations):
            for batch in dataloader:
                expert_s, expert_a = batch['state'], batch['action']
                bc_agent.learn(expert_s, expert_a)
            current_return = test_agent(bc_agent, env, 5)
            test_returns.append(current_return)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
            pbar.update(1)

    iteration_list = list(range(len(test_returns)))
    plt.plot(iteration_list, test_returns)
    plt.xlabel('Iterations')
    plt.ylabel('Returns')
    plt.title('BC on {highway-v0}')
    plt.show()