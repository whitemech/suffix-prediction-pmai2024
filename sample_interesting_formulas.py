import random
import torch

# Atomic propositions
atomic_propositions = ["p", "q", "r", "s", "t"]

# LTLf operators
operators = ["F", "G", "X", "&", "|", "->"]


def generate_complex_ltlf_formula(depth=3):
    # Base case: if depth is 0, return a random atomic proposition
    if depth == 0:
        return random.choice(atomic_propositions)

    # Choose a random operator
    op = random.choice(operators)

    # Generate subformulas recursively
    if op in ["F", "G", "X"]:
        subformula = generate_complex_ltlf_formula(depth - 1)
        return f"({op} {subformula})"
    elif op in ["&", "|", "->"]:
        left_subformula = generate_complex_ltlf_formula(depth - 1)
        right_subformula = generate_complex_ltlf_formula(depth - 1)
        return f"({left_subformula} {op} {right_subformula})"


def generate_random_string(length, characters):
    """
    Generate a random string of specified length using given characters.

    Parameters:
        length (int): Length of the random string to generate.
        characters (str): String containing the characters to choose from.

    Returns:
        str: Random string generated using the specified characters.
    """
    return "".join(random.choice(characters) for _ in range(length))


def generate_unique_random_strings(num_strings, length, characters):
    """
    Generate a set of unique random strings.

    Parameters:
        num_strings (int): Number of random strings to generate.
        length (int): Length of each random string.
        characters (str): String containing the characters to choose from.

    Returns:
        set: Set containing unique random strings.
    """
    unique_strings = set()
    while len(unique_strings) < num_strings:
        random_string = generate_random_string(length, characters)
        unique_strings.add(random_string)
    return unique_strings


def string_to_one_hot(string, characters):
    """
    Convert a string to one-hot encoding.

    Parameters:
        string (str): Input string to encode.
        characters (str): String containing the characters to encode.

    Returns:
        torch.Tensor: One-hot encoding of the input string.
    """
    char_to_idx = {char: idx for idx, char in enumerate(characters)}
    one_hot = torch.zeros(len(string), len(characters))
    for i, char in enumerate(string):
        one_hot[i, char_to_idx[char]] = 1
    return one_hot


# Example usage:
# num_strings = 200
# length = 20
# characters = ["0", "1", "2"]
def random_strings_dataset(num_strings, length, characters):
    random_strings = generate_unique_random_strings(num_strings, length, characters)

    # Convert strings to one-hot encoding
    one_hot_encodings = [string_to_one_hot(string, characters) for string in random_strings]

    # Convert list of one-hot encodings to a Torch tensor
    one_hot_tensor = torch.stack(one_hot_encodings)

    return one_hot_tensor


if __name__ == "__main__":
    # Generate and print a random complex LTLf formula
    for D in range(1, 8):
        random_formula = generate_complex_ltlf_formula(depth=D)
        print(random_formula)
