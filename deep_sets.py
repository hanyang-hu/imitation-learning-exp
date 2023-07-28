import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        layer_dim = [input_dim,] + hidden_dim + [output_dim,]
        self.fc = [torch.nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(len(layer_dim) - 1)]
        for layer in self.fc:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.fc:
            x = F.relu(layer(x))
        return x
    

class DeepSet(torch.nn.Module):
    def __init__(self, i_dim, r_dim, e_dim, phi_hidden = [256, 256], rho_hidden = [256,]):
        super(DeepSet, self).__init__()
        
        self.phi = MLP(i_dim, phi_hidden, r_dim)
        self.rho = MLP(r_dim, rho_hidden, e_dim)
        
    def forward(self, x):
        # Accept both batched and unbatched input
        is_batched = x.dim() > 2
        if not is_batched:
            x = x.unsqueeze(0)

        o = self.rho(torch.sum(self.phi(x), axis = 1))

        return o if is_batched else o.squeeze(0)
