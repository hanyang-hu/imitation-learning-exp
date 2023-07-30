import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributions import Categorical
import numpy as np
from attention import MLP, Attention

class AttentionQValueNetwork(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(AttentionQValueNetwork, self).__init__()
        self.attn = Attention(state_dim[0], state_dim[1]).to(device)
        self.MLP = MLP(self.attn.embed_dim, hidden_dim, action_dim).to(device)

    def forward(self, x):
        if len(x.shape) > 2:
            x = self.attn(x[:,0,:], x[:,:,:])
        else:
            x = self.attn(x[0], x[:])

        return self.MLP(x)


class AttentionValueNetwork(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, device):
        super(AttentionValueNetwork, self).__init__()
        self.attn = Attention(state_dim[0], state_dim[1]).to(device)
        self.MLP = MLP(self.attn.embed_dim, hidden_dim, 1).to(device)

    def forward(self, x):
        if len(x.shape) > 2:
            x = self.attn(x[:,0,:], x[:,:,:])
        else:
            x = self.attn(x[0], x[:])

        return self.MLP(x)


class AttentionPolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(AttentionPolicyNetwork, self).__init__()
        self.attn = Attention(state_dim[0], state_dim[1]).to(device)
        self.MLP = MLP(self.attn.embed_dim, hidden_dim, action_dim).to(device)

    def forward(self, x):
        if len(x.shape) > 2:
            x = self.attn(x[:,0,:], x[:,:,:])
        else:
            x = self.attn(x[0], x[:])

        return F.softmax(self.MLP(x), dim=-2)
    

class IQL(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, lr, alpha, gamma, tau, beta, target_update, device):
        super(IQL, self).__init__()

        """
        self.q_net = AttentionQValueNetwork(state_dim, hidden_dim, action_dim, device)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.q_lr_scheduler = lr_scheduler.LinearLR(self.q_optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.target_q_net = AttentionQValueNetwork(state_dim, hidden_dim, action_dim, device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        """
        self.q1_net = AttentionQValueNetwork(state_dim, hidden_dim, action_dim, device)
        self.q1_optimizer = torch.optim.Adam(self.q1_net.parameters(), lr=lr)
        self.q1_lr_scheduler = lr_scheduler.LinearLR(self.q1_optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.target_q1_net = AttentionQValueNetwork(state_dim, hidden_dim, action_dim, device)
        self.target_q1_net.load_state_dict(self.q1_net.state_dict())
        self.q2_net = AttentionQValueNetwork(state_dim, hidden_dim, action_dim, device)
        self.q2_optimizer = torch.optim.Adam(self.q2_net.parameters(), lr=lr)
        self.q2_lr_scheduler = lr_scheduler.LinearLR(self.q2_optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.target_q2_net = AttentionQValueNetwork(state_dim, hidden_dim, action_dim, device)
        self.target_q2_net.load_state_dict(self.q2_net.state_dict())

        self.v_net = AttentionValueNetwork(state_dim, hidden_dim, device)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=lr)
        self.v_lr_scheduler = lr_scheduler.LinearLR(self.v_optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)

        self.policy = AttentionPolicyNetwork(state_dim, hidden_dim, action_dim, device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_lr_scheduler = lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.5, total_iters=10)
        
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.target_update = target_update
        self.count = 0

        self.device = device

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype = torch.float).to(self.device)
        probs = self.policy(state)
        action_dist = Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def asym_loss(self, u):
        return torch.abs(self.tau - (u < 0).float()) * (u**2)
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.alpha) + param.data * self.alpha)
    
    def update(self, transition_dict):
        states = transition_dict['state'].float().to(self.device)
        actions = transition_dict['action'].view(-1, 1).to(self.device)
        rewards = transition_dict['reward'].float().view(-1, 1).to(self.device)
        next_states = transition_dict['next_state'].float().to(self.device)

        """
        # Update value network that approximates an expectile
        target_q_pred = (self.target_q_net(states).gather(1, actions)).detach()
        v_pred = self.v_net(states)
        v_loss = torch.mean(self.asym_loss(target_q_pred - v_pred))
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q netowrk
        TD_targets = (rewards + self.gamma * self.v_net(next_states)).detach()
        q_pred = self.q_net(states).gather(1, actions)
        q_loss = torch.mean(F.mse_loss(TD_targets, q_pred))
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        if (self.count + 1) % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        """

        q1_pred = self.q1_net(states).gather(1, actions)
        q2_pred = self.q2_net(states).gather(1, actions)
        target_v_pred = self.v_net(next_states).detach()

        q_target = rewards + self.gamma * target_v_pred
        q_target = q_target.detach()
        q1_loss = torch.mean(F.mse_loss(q1_pred, q_target))
        q2_loss = torch.mean(F.mse_loss(q2_pred, q_target))

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.soft_update(self.q1_net, self.target_q1_net)

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        self.soft_update(self.q2_net, self.target_q2_net)

        q_loss = (q1_loss + q2_loss) / 2

        q_pred = torch.min(
            self.target_q1_net(states).gather(1, actions),
            self.target_q2_net(states).gather(1, actions)
        )
        v_pred = self.v_net(states)
        v_err = v_pred - q_pred
        v_sign = (v_err > 0).float()
        v_weight = (1 - v_sign) * self.tau + v_sign * (1 - self.tau)
        v_loss = (v_weight * (v_err** 2)).mean()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        self.count += 1

        return q_loss.detach().item(), v_loss.detach().item()

    def policy_extraction(self, transition_dict):
        states = transition_dict['state'].float().to(self.device)
        actions = transition_dict['action'].view(-1, 1).to(self.device)

        target_q_pred = self.target_q_net(states).gather(1, actions)
        v_pred = self.v_net(states)
        advantage = target_q_pred - v_pred
        factor = torch.exp(self.beta * advantage)
        probs = self.policy(states)
        log_probs = torch.log(probs + 1e-8)
        policy_loss = -torch.mean(factor * log_probs) 
        # negative because when beta = 0, it's BC, i.e. we should be minimizing the cross entropy

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.detach().item()


if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from rl_utils import TransitionDataset

    state_dim = [5, 5]
    hidden_dim = [256, 256]
    action_dim = 5
    lr = 5e-3
    alpha = 1e-2
    gamma = 0.8
    tau = 0.525
    beta = 1.0
    target_update = 50
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    iql_agent = IQL(state_dim, hidden_dim, action_dim, lr, alpha, gamma, tau, beta, target_update, device)
    batch_size = 64

    n_iterations = 1000
    dataset = TransitionDataset('transition_data_mc.pkl')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    iterator = list(iter(dataloader))

    cnt = 0

    with tqdm(total=n_iterations, desc="IQL Learning Progress") as pbar:
        for i in range(n_iterations):
            q_loss, v_loss = [], []
            for transition_dict in iterator:
                ql, vl = iql_agent.update(transition_dict)
                q_loss.append(ql)
                v_loss.append(vl)
            cnt += 1
            if (cnt + 1) % 100 == 0:
                    iterator = list(iter(dataloader))
            if (cnt + 1) % 10 == 0:
                iql_agent.q1_lr_scheduler.step()
                iql_agent.q2_lr_scheduler.step()
                iql_agent.v_lr_scheduler.step()
            pbar.set_postfix({'TD Loss': '%.3f' % np.mean(q_loss), 'Expectile Loss': '%.3f' % np.mean(v_loss)})
            pbar.update(1)

    torch.save(iql_agent.state_dict(), "./model/iql_attn_npe.pt")

    print("IQL learning completed.")

    with tqdm(total=10, desc="Policy Extraction Progress") as pbar:
        for i in range(10):
            loss = []
            for transition_dict in iterator:
                loss.append(iql_agent.policy_extraction(transition_dict))
            iql_agent.policy_lr_scheduler.step()
            pbar.set_postfix({'Policy Loss': '%.3f' % np.mean(loss)})
            pbar.update(1)

    torch.save(iql_agent.state_dict(), "./model/iql_attn.pt")

    print("Policy extraction completed.")
