import torch
def expand_dataset_with_end_of_trace_symbol(original_dataset):
    """
    Expand the original dataset by appending an "end_of_sequence" symbol to each sequence.

    Args:
    original_dataset (torch.Tensor): Original dataset with shape (batch_size, len_sequences, num_symbols).

    Returns:
    torch.Tensor: Expanded dataset with shape (batch_size, len_sequences+1, num_symbols+1).
    """
    batch_size, len_sequences, num_symbols = original_dataset.size()

    # Expand dimensions of original dataset to accommodate end_of_sequence symbol
    expanded_dataset = torch.cat((original_dataset, torch.zeros(batch_size, len_sequences, 1)), dim=2)

    # Create end_of_sequence one-hot vector
    end_of_sequence = torch.zeros(batch_size, 1, num_symbols + 1)
    end_of_sequence[:, :, -1] = 1

    # Concatenate end_of_sequence to each sequence
    expanded_dataset = torch.cat((expanded_dataset, end_of_sequence), dim=1)

    return expanded_dataset