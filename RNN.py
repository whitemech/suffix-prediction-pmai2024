import torch.nn as nn
import torch.nn.functional as F
import torch
from FiniteStateMachine import DFA

if torch.cuda.is_available():
     device = 'cuda:0'
else:
    device = 'cpu'
class LSTM_model(nn.Module):

    def __init__(self, hidden_dim, vocab_size, tagset_size):
        super(LSTM_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.lstm = nn.LSTM(vocab_size, hidden_dim, self.num_layers, batch_first=True)
        self.input_size = vocab_size

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.positive_activation = torch.nn.ReLU()


    def forward(self, x):
        batch_size = x.size()[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        #x = lstm_out.reshape(-1, self.hidden_dim)
        x = lstm_out
        #print(x.size())
        tag_space = self.positive_activation(self.hidden2tag(x))+0.5
        #print(tag_space.size())

        return tag_space, (hn, cn)

    def forward_from_state(self, x, state):
        lstm_out, (hn, cn) = self.lstm(x, state)
        x = lstm_out
        tag_space = self.positive_activation(self.hidden2tag(x))+0.5
        return tag_space, (hn, cn)

    def next_sym_prob(self, x, state):
        tag_space, state = self.forward_from_state(x, state)
        tag_space = F.softmax(tag_space, dim=-1)
        return tag_space, state

    def predict(self, sentence):
        tag_space = self.forward(sentence)
        out = F.softmax(tag_space, dim=1)[-1]
        return out

class RNN_with_constraints_model(nn.Module):

    def __init__(self, rnn, ltl_formula):
        super(RNN_with_constraints_model, self).__init__()

        self.rnn = rnn
        #formula evaluator
        dfa = DFA(ltl_formula, 2, "random DNF declare", ['c0', 'c1', 'end'])
        self.deep_dfa_constraint = dfa.return_deep_dfa_constraint()

    def forward(self, x):
        pred_sym, hidden_states = self.rnn(x)
        #print(pred_sym.size())
        #print(x[0])
        #TODO transform one-hot into indices
        x_indices = torch.argmax(x, dim= -1).long()
        #print(x_indices.size())
        #print(x_indices[0])
        masks, dfa_state = self.deep_dfa_constraint(x_indices)

        #print(masks.size())
        #print(masks)

        #print(pred_sym.size())
        #print(pred_sym)


        pred_sym = (pred_sym) * masks

        #print(pred_sym.size())
        #print(pred_sym)

        return pred_sym, (hidden_states, dfa_state)

    def forward_from_state(self, x, tot_state):
        state_rnn, state_dfa = tot_state

        #print("state dfa")
        #print(state_dfa[0])

        #print("symbol one hot")
        #print(x[0])

        next_event, next_state_rnn = self.rnn.forward_from_state(x, state_rnn)

        next_event = next_event.squeeze()
        #print(x)
        #print(x.size())
        x = torch.argmax(x, -1).squeeze()
        #print("symbol index")
        #print(x[0])
        #print(x)
        #print(x.size())
        next_dfa_state, mask = self.deep_dfa_constraint.step(state_dfa, x)

        #print("next dfa state")
        #print(next_dfa_state[0])

        #print("mask to put on next event prediction")
        #print(mask[0])
        #print(mask.size())

        #print("next event according rnn")
        #print(next_event[0])
        #print(next_event.size())

        next_event =(next_event) * mask

        #print("next event according rnn + constraint")
        #print(next_event[0])
        #print(next_event.size())

        #print("_________________________________________________")

        return next_event.unsqueeze(1),  (next_state_rnn, next_dfa_state)

'''    
class LSTM_next_activity_predictor(nn.Module):
    
    def __init__(self, hidden_dim, vocab_size):
'''