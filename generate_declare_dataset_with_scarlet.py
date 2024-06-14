import random


def generate_dnf_formula(all_variables, min_conjuncts, max_conjuncts, min_disjuncts, max_disjuncts, alphabet):
    formula = []
    num_disjuncts = random.randint(min_disjuncts, max_disjuncts)
    # print("#disjunct:", num_disjuncts)
    for _ in range(num_disjuncts):
        num_conjuncts = random.randint(min_conjuncts, max_conjuncts)
        # print("--- #num conjuncts", num_conjuncts)
        disjunct = []
        for _ in range(num_conjuncts):
            variable = random.choice(all_variables)

            variable = variable.replace("c0", random.choice(alphabet))
            variable = variable.replace("c1", random.choice(alphabet))

            variable = "("+variable+")"
            if random.random() < 0.5:
                variable = '!' + variable  # Negation
            disjunct.append(variable)
        formula.append('(' + ' & '.join(disjunct) + ')')

    return ' | '.join(formula)

import os
import numpy as np


def scarlet_traces_to_stlnet_format(scarlet_traces_folder, dataset_name = "FORMULANUMBER.dat"):
    # Iterate files in scarlet_traces_folder and get the traces in each file (each file contains traces from a single formula)
    traces_by_formula = []
    for filename in os.listdir(scarlet_traces_folder):
        if filename.endswith(".trace"):
            with open(os.path.join(scarlet_traces_folder, filename), "r", encoding="unicode_escape") as f:
                lines = f.read().splitlines()
                traces_by_formula.append(lines)

    # Get the positive traces (the items in the list traces_by_file that happen before the first item that is '---')
    positive_traces_by_formula = []
    for traces in traces_by_formula:
        positive_traces_by_formula.append(traces[: traces.index("---")])
    traces_by_formula = positive_traces_by_formula

    # Convert every trace into a list of timesteps by splitting the trace by ;
    traces_by_formula = [[trace.split(";") for trace in traces] for traces in traces_by_formula]
    # Convert every timestep into a list of variables by splitting the timestep by , and converting the variables to integers
    traces_by_formula = [[[[int(variable) for variable in timestep.split(",")] for timestep in trace] for trace in traces] for traces in traces_by_formula]

    # Convert traces list to a numpy array
    traces_by_formula = np.array(traces_by_formula, dtype=np.int8)
    # Reshape the traces such that every trace is a single row of data where [var1, var2, ..., varn] are concatenated
    traces_by_formula = [traces.reshape(traces.shape[0], traces.shape[1] * traces.shape[2]) for traces in traces_by_formula]

    # Save the each traces_by_formula to a file
    for i, traces in enumerate(traces_by_formula):
        np.savetxt(dataset_name.replace("FORMULANUMBER", str(i)), traces, fmt="%d")
