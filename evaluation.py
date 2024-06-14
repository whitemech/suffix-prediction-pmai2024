import torch
from statistics import mean
from torch.nn.utils.rnn import pad_sequence
if torch.cuda.is_available():
     device = 'cuda:0'
else:
    device = 'cpu'

cross_entr_func = torch.nn.CrossEntropyLoss()

def evaluate_accuracy_next_activity(rnn, test_dataset, acc_func):
    rnn = rnn.to(device)
    accuracies = []
    for batch in [test_dataset]:
        # print(batch.size())
        X = batch[:, :-1, :].to(device)
        # print("X size:", X.size())
        Y = batch[:, 1:, :]
        # print(Y.size())
        target = torch.argmax(Y.reshape(-1, Y.size()[-1]), dim=-1).to(device)
        # print(target.size())
        with torch.no_grad():
            predictions, _ = rnn(X)
        predictions = predictions.reshape(-1, predictions.size()[-1])

        accuracies.append(acc_func(predictions, target).item())

    return mean(accuracies)

import torch.nn.functional as F


def round_to_one_hot(tensor):
    # Find the index of the maximum value along the last dimension
    max_indices = torch.argmax(tensor, dim=-1, keepdim=True)

    # Create a one-hot tensor with the same shape as the input tensor
    one_hot_tensor = torch.zeros_like(tensor)

    # Set the element corresponding to the maximum value to 1
    one_hot_tensor.scatter_(-1, max_indices, 1)

    return one_hot_tensor

def greedy_suffix_prediction(rnn, dataset, prefix_len):
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]

    predicted_traces = prefix

    len_traces = dataset.size()[1]

    next_event, rnn_state = rnn(prefix)

    for step in range(prefix_len, len_traces*2):
        next_event = F.softmax(next_event[:, -1:, :], dim=-1)
        next_event = round_to_one_hot(next_event)

        predicted_traces = torch.cat((predicted_traces, next_event), dim=1)

        #next_event = next_event.squeeze()
        next_event, rnn_state = rnn.forward_from_state(next_event, rnn_state)

    return predicted_traces


def sample_with_temperature(probabilities, temperature=1.0):
    if temperature == 0:
        return torch.argmax(probabilities, dim=-1)
    else:
        #logits = torch.log(probabilities) / temperature
        #probabilities = F.softmax(logits, dim=-1)

        # Mask out zero probabilities
        #TODO: se tutte le componenti sono 0 a zero aggiungi una costante
        #Nota: potrebbe averle tutte uguali a 0 perchè è andato nello stato di fallimento...
        #ma perchè c'è andato??
        #mask = probabilities <= 0
        #probabilities.masked_fill_(mask, -1e12)  # Replace zeros with a very small value

        batch_size = probabilities.size()[0]
        num_classes = probabilities.size()[-1]
        num_samples = 1

        probabilities = probabilities + 1e-10
        indices = torch.multinomial(probabilities.squeeze(), num_samples)

        # Create one-hot vectors based on the drawn indices
        one_hot_vectors = torch.zeros(batch_size, num_samples, num_classes).to(device)
        one_hot_vectors.scatter_(2, indices.unsqueeze(-1), 1)

        #print("ONE_HOT_SAMPLED", one_hot_vectors[0])
        return one_hot_vectors

#Da provare
def suffix_prediction_with_temperature(model, dataset, prefix_len, temperature=1.0):
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]

    predicted_traces = prefix

    len_traces = dataset.size()[1]

    next_event, rnn_state = model(prefix)

    for step in range(prefix_len, len_traces):
        next_event = next_event[:, -1:, :]
        next_event_one_hot = sample_with_temperature(next_event, temperature)

        predicted_traces = torch.cat((predicted_traces, next_event_one_hot), dim=1)

        next_event, rnn_state = model.forward_from_state(next_event_one_hot, rnn_state)
    return predicted_traces


def gumbel_softmax(logits, temperature=1.0, eps=1e-10):
    """
    Gumbel-Softmax sampling function.
    """
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def differentiable_suffix_prediction_with_temperature(model, dataset, prefix_len, temperature=1.0):
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]

    predicted_traces = prefix

    len_traces = dataset.size()[1]

    next_event, rnn_state = model(prefix)

    for step in range(prefix_len, len_traces*2):
        next_event = next_event[:, -1:, :]
        next_event_one_hot = gumbel_softmax(next_event, temperature)

        print("next event one-hot: (size)=", next_event_one_hot.size())
        print(next_event_one_hot[0])

        predicted_traces = torch.cat((predicted_traces, next_event_one_hot), dim=1)

        next_event, rnn_state = model.forward_from_state(next_event_one_hot, rnn_state)
    return predicted_traces

def logic_loss(rnn, deepdfa, data, prefix_len, temperature=1.0):
    dataset = data.to(device)
    prefix = dataset[:, :prefix_len, :]

    batch_size = dataset.size()[0]
    target = torch.ones(batch_size, dtype=torch.long, device=device)

    len_traces = dataset.size()[1]
    next_event, rnn_state = rnn(prefix)
    dfa_states, dfa_rew = deepdfa.forward_pi(prefix)

    dfa_state = dfa_states[:, -1, :]

    for step in range(prefix_len, int(len_traces*(1.5))):
        next_event = next_event[:, -1:, :]
        next_event_one_hot = gumbel_softmax(next_event, temperature)
        #print(next_event_one_hot)
        #predicted_traces = torch.cat((predicted_traces, next_event_one_hot), dim=1)
        #transit on the automaton
        dfa_state, dfa_rew = deepdfa.step_pi(dfa_state, next_event_one_hot.squeeze())

        next_event, rnn_state = rnn.forward_from_state(next_event_one_hot, rnn_state)
    loss = cross_entr_func(100*dfa_rew, target)

    return loss

def suffix_prediction_with_temperature_with_stop(model, dataset, prefix_len, stop_event=[0,0,0,1], temperature=1.0):
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]

    predicted_traces = prefix

    len_traces = dataset.size()[1]

    next_event, rnn_state = model(prefix)

    # Initialize a mask indicating which sequences have reached the stop event
    stop_mask = torch.zeros(prefix.size(0)).bool().to(device)

    for step in range(prefix_len, len_traces *2):
        #print("STEP: ", step)
        #next_event = F.softmax(next_event[:, -1:, :] / temperature, dim=-1)
        next_event = next_event[:, -1:, :]
        #print("next event after softmax:", next_event[0])
        next_event_one_hot = sample_with_temperature(next_event, temperature)

        predicted_traces = torch.cat((predicted_traces, next_event_one_hot), dim=1)

        # Check if any sequence has reached the stop event and update the stop mask
        stop_mask |= torch.all(next_event_one_hot.squeeze() == torch.tensor(stop_event).to(device), dim=-1)

        # Check if all sequences have reached the stop event
        if torch.all(stop_mask):
            break  # Stop predicting if all sequences have reached the stop event

        # Mask out the sequences that have already reached the stop event
        #next_event_one_hot = next_event_one_hot.masked_fill(stop_mask.unsqueeze(1).unsqueeze(2), 0)

        next_event, rnn_state = model.forward_from_state(next_event_one_hot, rnn_state)

    return predicted_traces

def greedy_suffix_prediction_with_stop(rnn, dataset, prefix_len, stop_event=[0,0,0,1]):
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]

    predicted_traces = prefix

    len_traces = dataset.size()[1]

    next_event, rnn_state = rnn(prefix)

    # Initialize a mask indicating which sequences have reached the stop event
    stop_mask = torch.zeros(prefix.size(0)).bool().to(device)

    for step in range(prefix_len, len_traces*2):
        next_event = F.softmax(next_event[:, -1:, :], dim=-1)
        next_event = round_to_one_hot(next_event)

        predicted_traces = torch.cat((predicted_traces, next_event), dim=1)

        # Check if any sequence has reached the stop event and update the stop mask
        stop_mask |= torch.all(next_event.squeeze() == torch.tensor(stop_event).to(device), dim=-1)

        # Check if all sequences have reached the stop event
        if torch.all(stop_mask):
            break  # Stop predicting if all sequences have reached the stop event

        #next_event = next_event.squeeze()
        next_event, rnn_state = rnn.forward_from_state(next_event, rnn_state)

    return predicted_traces

def suffix_prediction_beam_search(rnn, dataset, prefix_len, stop_event = torch.tensor([0, 0, 0, 1]).float()):
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]

    len_traces = dataset.size()[1]

    predicted_traces = beam_search(rnn, prefix, 3, len_traces*2, stop_event.to(device))

    #TODO: padd the predicted traces with end symbol
    print(prefix[:3,:,:])
    print(prefix.size())
    print(len(suffixes))
    for s in suffixes[:3]:
        print(s.size())
        print(s)
    assert False
    return predicted_traces

def beam_search(model, prefixes, beam_width, max_length, stop_event):
    suffixes = []
    with torch.no_grad():
        for prefix in prefixes:
            prefix_tensor = prefix
            beams = [(prefix_tensor, 1.0)]
            generated_new_beam = True

            while generated_new_beam:
                generated_new_beam = False
                new_beams = []
                for beam in beams:
                    prefix_tensor, prob = beam
                    #print(prefix_tensor)
                    #print(stop_event)
                    if len(prefix_tensor) >= max_length or torch.equal(prefix_tensor[-1], stop_event):
                        new_beams.append(beam)
                        continue
                    else:
                        generated_new_beam = True
                    output, _ = model(prefix_tensor.unsqueeze(0))
                    next_event_probs = F.softmax(output[:, -1, :], dim=-1)
                    #print("next event probs:", next_event_probs)
                    top_probs, top_indices = torch.topk(next_event_probs, beam_width, dim=-1)
                    #print("top probs:", top_probs)
                    #print("top indices:", top_indices)
                    for i in range(beam_width):
                        index = top_indices[0][i].item()
                        new_prefix_tensor = torch.cat([prefix_tensor, torch.zeros(1, model.input_size).to(device)])
                        new_prefix_tensor[-1][index] = 1.0
                        new_prob = prob * top_probs[0][i].item()
                        new_beams.append((new_prefix_tensor, new_prob))
                    #print("______new beams:")
                    #for b in new_beams:
                        #print("seq:", b[0])
                        #print("prob:", b[1])
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_width]

            # Choose the top-scoring suffix for the current prefix
            suffixes.append(beams[0][0])
    return suffixes

import numpy as np

def pad_sequences_with_stop_event(sequences, stop_event):
    # Get maximum sequence length
    max_length = max(len(seq) for seq in sequences)
    # Initialize padded sequences tensor
    padded_sequences = torch.zeros((len(sequences), max_length, sequences[0].shape[1])).to(device)
    # Pad sequences and replace last row with stop_event
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
        padded_sequences[i, len(seq)-1] = stop_event
    return padded_sequences
def suffix_prediction_beam_search_ltl(k, rnn, dataset, prefix_len, deep_dfa, stop_event = torch.tensor([0, 0, 0, 1]).float()):
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]

    len_traces = dataset.size()[1]

    suffixes = beam_search_with_ltl(rnn, prefix, k, len_traces*2, stop_event.to(device), deep_dfa)

    suffixes = pad_sequences_with_stop_event(suffixes, stop_event)

    return suffixes

def beam_search_with_ltl(model, prefixes, beam_width, max_length, stop_event, deepdfa):
    suffixes = []
    with torch.no_grad():
        for prefix in prefixes:
            prefix_tensor = prefix
            beams = [(prefix_tensor, 1.0)]
            generated_new_beam = True

            while generated_new_beam:
                generated_new_beam = False
                new_beams = []
                for beam in beams:
                    prefix_tensor, prob = beam
                    #print(prefix_tensor)
                    #print(stop_event)
                    if len(prefix_tensor) >= max_length or torch.equal(prefix_tensor[-1], stop_event):
                        new_beams.append(beam)
                        continue
                    else:
                        generated_new_beam = True
                    output, _ = model(prefix_tensor.unsqueeze(0))
                    next_event_probs = F.softmax(output[:, -1, :], dim=-1)
                    #print("next event probs:", next_event_probs)
                    top_probs, top_indices = torch.topk(next_event_probs, beam_width, dim=-1)
                    #print("top probs:", top_probs)
                    #print("top indices:", top_indices)
                    all_rejected = True
                    for i in range(beam_width):
                        index = top_indices[0][i].item()
                        new_prefix_tensor = torch.cat([prefix_tensor, torch.zeros(1, model.input_size).to(device)])
                        new_prefix_tensor[-1][index] = 1.0
                        new_prob = prob * top_probs[0][i].item()
                        new_beams.append((new_prefix_tensor, new_prob))

                    #print("______new beams:")
                    #for b in new_beams:
                        #print("seq:", b[0])
                        #print("prob:", b[1])

                new_beams.sort(key=lambda x: x[1], reverse=True)

                beams = new_beams[:beam_width]


            # Choose the top-scoring suffix compliant with the ltl
            found_compliant = False
            for beam in beams:
                r, _ = deepdfa.forward_pi(beam[0].unsqueeze(0))
                accepted = r[:, -1, -1]
                if accepted > 0:
                    suffixes.append(beam[0])
                    found_compliant = True
                    break
            #if none of the predicted suffixes is compliant return the most probable
            if not found_compliant:
                suffixes.append(beams[0][0])
    return suffixes
def evaluate_compliance_with_formula(deepdfa, traces):
    traces = torch.argmax(traces, dim= -1)

    r, _ = deepdfa(traces)
    accepted = r[:,-1,-1]

    return accepted.mean().item()

def evaluate_compliance_with_formulas(deepdfa_list, traces):
    traces = torch.argmax(traces, dim= -1)
    total_accepted = torch.ones(traces.size()[0])

    for deepdfa in deepdfa_list:
        r, _ = deepdfa(traces)
        accepted = r[:,-1,-1]
        total_accepted *= accepted

    return total_accepted.mean().item()

def damerau_levenshtein_distance(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)
    # Create a matrix to store the distances between substrings
    matrix = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]

    # Initialize the first row and column of the matrix
    for i in range(len_str1 + 1):
        matrix[i][0] = i
    for j in range(len_str2 + 1):
        matrix[0][j] = j

    # Populate the matrix
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i][j] = min(matrix[i-1][j] + 1,          # Deletion
                               matrix[i][j-1] + 1,          # Insertion
                               matrix[i-1][j-1] + cost)    # Substitution
            # Check for transposition
            if i > 1 and j > 1 and str1[i-1] == str2[j-2] and str1[i-2] == str2[j-1]:
                matrix[i][j] = min(matrix[i][j], matrix[i-2][j-2] + cost)

    return matrix[len_str1][len_str2]


def evaluate_DL_distance(predicted_traces, target_traces):
    DL_dists = []

    for i in range(predicted_traces.size()[0]):
        pred = tensor_to_string(predicted_traces[i])
        targ = tensor_to_string(target_traces[i])
        DL_dists.append(damerau_levenshtein_distance(pred, targ))

    return mean(DL_dists)

import torch

def tensor_to_string(one_hot_tensor):
    end_symbol = one_hot_tensor.size()[-1] -1
    # Convert the one-hot tensor to a numpy array
    numpy_array = one_hot_tensor.cpu().numpy()

    # Extract indices of maximum values along the second dimension
    indices = numpy_array.argmax(axis=1)

    # Convert indices into a string
    string = ''
    for idx in indices:
        #if idx == end_symbol:
        #    return string
        string += str(idx)
    #e se non ho terminato la stringa???
    #print(string)
    return string
