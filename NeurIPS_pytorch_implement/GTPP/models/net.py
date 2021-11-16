# Check the 2.4. The proposed model!

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Hazard_function_network(nn.Module):
    def __init__(self, cnf):
        super(Hazard_function_network, self).__init__()

        hid_dim = cnf.all_params["n_inputs"]
        mlp_dim = cnf.all_params["n_outputs"]
        mlp_layer = cnf.all_params["n_layers"]

        self.linear_tau = nn.Linear(in_features = 1, out_features = 1)                # for tau
        self.linear_tau_weight= nn.Linear(in_features = hid_dim + 1, out_features = mlp_dim)
        self.module_list = nn.ModuleList(
            [nn.Linear(mlp_dim, mlp_dim) for _ in range(mlp_layer-1)  ]
        )
        self.linear_hazard =  nn.Sequential(nn.Linear(in_features=mlp_dim, out_features=1), nn.Softplus())



    def forward(self, hidden_state, elapsed_time):
        """
        hidden_state : h_i from LSTM class.
        elapsed time : τ from the most recent event untill the next event.

        Check equation 10. \phi( τ | h_i) = ( \partial / \partial * τ ) /cumsum_phi( τ | h_i )
        """

        for weight in self.parameters():
            # The weights of the connections from the elapsed time τ to the first hidden layer and allthe connections from the hidden layers are constrained to be positive.
            #weight.data *= (weight.data >= 0)
            weight.data = torch.abs(weight.data)

        elapsed_time.requires_grad_(True)
        t =  self.linear_tau(elapsed_time.unsqueeze(dim = -1).float())                                # The first hidden layer in the network receives the elapsed time τ and the hidden state h_i of the RNN as the inputs
        out = torch.tanh(self.linear_tau_weight( torch.cat([hidden_state[:, -1, :], t], dim=-1) ))      # The activation functions of the hidden units is tanh function.

        for layer in self.module_list:
            out = torch.tanh(layer(out))


        cumsum_hazard_function = F.softplus(self.linear_hazard(out))   # output unit is softplus function
        cumsum_hazard_function = torch.mean(cumsum_hazard_function)
        cumsum_hazard_function.requires_grad_(True)

        # https://velog.io/@bismute/Pytorch%EC%9D%98-Autograd%EB%A5%BC-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0
        hazard_function = torch.autograd.grad(cumsum_hazard_function, elapsed_time, create_graph=True, retain_graph=True)[0]

        nll = torch.add(cumsum_hazard_function, -torch.mean(torch.log(hazard_function + 1e-7)))

        return nll, torch.mean(torch.log(hazard_function + 1e-7)), cumsum_hazard_function, hazard_function



class Net(nn.Module):
    # We combine RNN + Hazard function network
    def __init__(self, cnf):
        super(Net, self).__init__()

        logarithm_interval_time = cnf.log_bool # A simple form of the input is the inter-event interval as x_i = (t_i - t_{i-1}) or its logarithm as x_i = (log(t_i - t_{i-1}))
        event_class = cnf.all_params["n_event"]
        emb_dim = cnf.all_params["n_emb_dim"]
        dropout_rate = cnf.all_params["dropout_rate"]
        hid_dim = cnf.all_params["n_inputs"]


        self.emb = nn.Embedding(num_embeddings = event_class, embedding_dim= emb_dim)
        self.emb_dropout = nn.Dropout(p = dropout_rate)
        self.lstm = nn.LSTM(input_size =  emb_dim+1, # 1+emb_dim if exist the event type
                            hidden_size= hid_dim,
                            batch_first=True,
                            bidirectional=False)

        self.Hazard_function_network = Hazard_function_network(cnf)


    def forward(self, batch_data, batch_data2):

        time_seq, event_seq= batch_data, batch_data2

        event_seq = event_seq.long()
        emb = self.emb(event_seq)
        emb = self.emb_dropout(emb)

        lstm_input = torch.cat([emb, time_seq.unsqueeze(-1)], dim = -1).float()
       # lstm_input = torch.tensor(lstm_input, dtype= torch.float32)

        hidden_state, _ = self.lstm(lstm_input)

        nll, log_hazard_function, cumsum_hazard_function, hazard_function = self.Hazard_function_network(hidden_state, time_seq[:, -1])

        return nll, log_hazard_function, cumsum_hazard_function, hazard_function

    def set_optimizer(self, total_step):
        pass























