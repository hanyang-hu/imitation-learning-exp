import torch
# import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributions import Categorical
import numpy as np
from attention import MLP, Attention

class AttentionFeatureExtractor(torch.nn.Module):
    def __init__(self, ego, oppo):
        super(AttentionFeatureExtractor, self).__init__()
        self.attn = Attention(ego, oppo)

    @property
    def embed_dim(self):
        return self.attn.embed_dim
    
    def forward(self, x):
        if len(x.shape) > 2:
            return self.attn(x[:,0,:], x[:,:,:])
        else:
            return self.attn(x[0], x[:])
    

class IQL(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, lr, alpha, gamma, tau, beta, device, optimizer = torch.optim.Adam):
        super(IQL, self).__init__()
        self.attn_feature = AttentionFeatureExtractor(state_dim[0], state_dim[1]).to(device)
        self.attn_feature.attn.load_state_dict(torch.load('./model/attn_feature.pt'))
        self.attn_optimizer = optimizer(self.attn_feature.parameters(), lr=lr)
        self.attn_scheduler = lr_scheduler.LinearLR(self.attn_optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        # self.attn_feature.requires_grad_(False)
        embed_dim = self.attn_feature.embed_dim

        self.v_net = MLP(embed_dim, hidden_dim, 1).to(device)
        self.v_optimizer = optimizer(self.v_net.parameters(), lr=lr)
        self.v_scheduler = lr_scheduler.LinearLR(self.v_optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)

        self.q1_net = MLP(embed_dim, hidden_dim, action_dim).to(device)
        self.q1_optimizer = optimizer(self.q1_net.parameters(), lr=lr)
        self.q1_scheduler = lr_scheduler.LinearLR(self.q1_optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.target_q1_net = MLP(embed_dim, hidden_dim, action_dim).to(device)
        self.target_q1_net.load_state_dict(self.q1_net.state_dict())
        self.target_q1_net.requires_grad_(False)

        self.q2_net = MLP(embed_dim, hidden_dim, action_dim).to(device)
        self.q2_optimizer = optimizer(self.q2_net.parameters(), lr=lr)
        self.q2_scheduler = lr_scheduler.LinearLR(self.q2_optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.target_q2_net = MLP(embed_dim, hidden_dim, action_dim).to(device)
        self.target_q2_net.load_state_dict(self.q2_net.state_dict())
        self.target_q2_net.requires_grad_(False)

        self.p_net = MLP(embed_dim, hidden_dim, action_dim, softmax = True).to(device)
        self.p_optimizer = optimizer(self.p_net.parameters(), lr=lr)

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.device = device

    def scheduler_step(self):
        self.attn_scheduler.step()
        self.q1_scheduler.step()
        self.q2_scheduler.step()
        self.v_scheduler.step()

    def take_action(self, state, soft_decision = True):
        state = torch.tensor(np.array([state]), dtype = torch.float).to(self.device)
        probs = self.p_net(state)
        if soft_decision:
            action_dist = Categorical(probs)
            action = action_dist.sample()
        else:
            action = torch.argmax(probs, dim = -1)
        return action.item()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.alpha) + param.data * self.alpha)

    def update(self, transition_dict):
        states = self.attn_feature(transition_dict['state'].float().to(self.device))
        actions = transition_dict['action'].view(-1, 1).to(self.device)
        rewards = transition_dict['reward'].float().view(-1, 1).to(self.device)
        next_states = self.attn_feature(transition_dict['next_state'].float().to(self.device))
        dones = transition_dict['done'].float().view(-1, 1).to(self.device)

        # Compute q-networks loss
        q1_pred = self.q1_net(states).gather(1, actions)
        q2_pred = self.q2_net(states).gather(1, actions)
        target_v_pred = self.v_net(next_states).detach()

        TD_target = (rewards + (1 - dones) * self.gamma * target_v_pred).detach()
        q1_loss = torch.nn.MSELoss()(q1_pred, TD_target)
        q2_loss = torch.nn.MSELoss()(q2_pred, TD_target)

        # Compute v-network loss (expectile)
        target_q_pred = torch.min(
            self.target_q1_net(states).gather(1, actions),
            self.target_q2_net(states).gather(1, actions)
        ).detach()
        v_pred = self.v_net(states)
        u = target_q_pred - v_pred
        factor = torch.abs(self.tau - (u < 0).float()) # |tau - 1[u < 0]|
        v_loss = (factor * (u**2)).mean()

        attn_loss = q1_loss + q2_loss + v_loss

        # Update
        self.q1_optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward(retain_graph=True)
        self.q2_optimizer.step()

        self.v_optimizer.zero_grad()
        v_loss.backward(retain_graph=True)
        self.v_optimizer.step()

        self.attn_optimizer.zero_grad()
        attn_loss.backward()
        self.attn_optimizer.step()

        self.soft_update(self.q1_net, self.target_q1_net)
        self.soft_update(self.q2_net, self.target_q2_net)

        return min(q1_loss.detach().item(), q2_loss.detach().item()), v_loss.detach().item()

    def policy_extraction(self, transition_dict):
        states = self.attn_feature(transition_dict['state'].float().to(self.device))
        actions = transition_dict['action'].view(-1, 1).to(self.device)

        q_pred = torch.min(
            self.q1_net(states).gather(1, actions),
            self.q2_net(states).gather(1, actions)
        ).detach()
        v_pred = self.v_net(states).detach()
        advantage = q_pred - v_pred
        factor = torch.exp(self.beta * advantage)
        log_probs = torch.log(self.p_net(states).gather(1, actions)) 
        # log_probs = log_probs * torch.pow(1 - torch.exp(log_probs), 5) # No focal loss, this is not plain classification
        policy_loss = -torch.mean(factor * log_probs)

        self.p_optimizer.zero_grad()
        policy_loss.backward()
        self.p_optimizer.step()

        return policy_loss.detach().item()
    

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from rl_utils import TransitionDataset

    state_dim = [5, 5]
    hidden_dim = [256, ]
    action_dim = 5
    lr = 1e-3
    alpha = 0.005
    gamma = 0.99
    tau = 0.8
    beta = 10.0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_size = 256
    n_iterations = 1000
    dataset = TransitionDataset('filtered_transition_data_iql2.pkl')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    iterator = list(iter(dataloader))

    iql_agent = IQL(state_dim, hidden_dim, action_dim, lr, alpha, gamma, tau, beta, device)

    with tqdm(total=n_iterations, desc="IQL Learning Progress") as pbar:
        for i in range(n_iterations):
            q_loss, v_loss = [], []
            for transition_dict in iterator:
                ql, vl = iql_agent.update(transition_dict)
                q_loss.append(ql)
                v_loss.append(vl)
            pbar.set_postfix({'Min TD Loss': '%.3f' % np.mean(q_loss), 'Expectile Loss': '%.3f' % np.mean(v_loss)})
            pbar.update(1)
            if (i + 1) % (n_iterations / 100):
                iql_agent.scheduler_step()

    torch.save(iql_agent.state_dict(), "./model/iql_attn_npe.pt")

    print("IQL learning completed.")

    batch_size = 128
    dataset = TransitionDataset('transition_data_bc.pkl')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    iterator = list(iter(dataloader))

    with tqdm(total=100, desc="Policy Extraction Progress") as pbar:
        for i in range(100):
            loss = []
            for transition_dict in iterator:
                loss.append(iql_agent.policy_extraction(transition_dict))
            pbar.set_postfix({'Policy Loss': '%.3f' % np.mean(loss)})
            pbar.update(1)

    torch.save(iql_agent.state_dict(), "./model/iql_attn_pe.pt")

    print("Policy extraction completed.")
