import torch
import os
class Network(torch.nn.Module):
    def __init__(self, n_actions, input_dims, hidden_dims=30):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_dims, hidden_dims)
        self.V = torch.nn.Linear(hidden_dims, 1)
        self.A = torch.nn.Linear(hidden_dims, n_actions)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = torch.nn.functional.relu(self.fc1(state))
        
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

class Agent:
    def __init__(self, turn):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weight_file_path = os.path.join(dir_path, turn, 'weight')
        
        self.network = Network(8, 340)
        self.network.load_state_dict(torch.load(weight_file_path))
    
    def choose_action(self, observation):
        state = torch.flatten(torch.tensor(observation[:, :17])).to(self.network.device)
        _, advantage = self.network.forward(state)
        action = torch.argmax(advantage).item()
        return action
