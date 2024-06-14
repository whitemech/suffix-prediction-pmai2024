import pathlib
import time
import json
from statistics import mean
from copy import deepcopy

import torch
import numpy as np
from Scarlet.genBenchmarks import SampleGenerator

from LTL2STL import infix_to_prefix, prefix_LTL_to_scarlet
from Declare_formulas import formulas
from generate_declare_dataset_with_scarlet import generate_dnf_formula, scarlet_traces_to_stlnet_format
from RNN import LSTM_model
from training import train
from evaluation import suffix_prediction_with_temperature_with_stop, evaluate_compliance_with_formula, evaluate_DL_distance, greedy_suffix_prediction_with_stop
from FiniteStateMachine import DFA
from utils import expand_dataset_with_end_of_trace_symbol
from sample_interesting_formulas import random_strings_dataset

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def main():
    # Create a folders and files to store the results with the current timestamp
    experiment_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = pathlib.Path("results", f"results_{experiment_datetime}")
    results_file = pathlib.Path(results_folder, "results.txt")
    results_folder.mkdir(parents=True, exist_ok=True)
    results_file.touch()

    # Number of experiments
    N_CONFIGURATIONS = 5
    N_FORMULAS = 5
    N_EXPERIMENTS_PER_FORMULA = 5

    # Parameters for RNN
    NVAR = 3
    HIDDEN_DIM = 100
    TRAIN_RATIO = 0.9
    TEMPERATURE = 0.7
    MAX_NUM_EPOCHS = 4000
    EPSILON = 0.01

    # Experiment parameters initial configuration
    D = 6
    C = 0

    # PARAMETERS TO VARY
    TRACE_LENGTH = 20
    SAMPLE_SIZE_START_VALUE = 250
    SAMPLE_SIZE_INCREMENT = 250
    SAMPLE_SIZE_INCREMENT_ITERATIONS = 4
    PREFIX_LEN_START_VALUE = 5
    PREFIX_LEN_INCREMENT = 5
    PREFIX_LEN_INCREMENT_ITERATIONS = 3

    for config in range(N_CONFIGURATIONS):
        # Dictionary to store the results for each configuration
        configuration_results = {}

        # Track execution time per config
        start_time = time.time()

        # Update the configuration
        C += 1
        D -= 1
        if D == 0:  # If formula is empty
            break

        min_conjuncts = C
        max_conjuncts = C
        min_disjuncts = D
        max_disjuncts = D

        # Variables to store the results of the current configuration
        configuration_results[str((D, C))] = {}

        # Create folders to store the results for the current configuration
        dataset_folder = pathlib.Path("datasets", f"datasets_{experiment_datetime}", f"dataset_declare_D={D}_C={C}")
        results_config_folder = pathlib.Path(results_folder, f"declare_D={D}_C={C}")
        dataset_folder.mkdir(parents=True, exist_ok=True)
        results_config_folder.mkdir(parents=True, exist_ok=True)

        # Generate random formulas and save to file
        formulas_dec_file = pathlib.Path(results_config_folder, "formulas_dec.txt")
        with open(formulas_dec_file, "w+") as f:
            
            # Generate N_FORMULAS
            DNF_formulas_infix = []
            for i_form in range(N_FORMULAS):

                # Store the results for i_form-th formula
                configuration_results[str((D, C))][i_form] = {}

                # Generate formula with satisfaction rate between 0.1 and 0.9
                sat_rate = 1
                while sat_rate < 0.1 or sat_rate > 0.9:
                    # Generate random formula
                    formula = generate_dnf_formula(formulas, min_conjuncts, max_conjuncts, min_disjuncts, max_disjuncts, ["c0", "c1", "c2"])

                    # Check if formula was already generated
                    if formula.replace("i", "->").replace("e", "<->") in DNF_formulas_infix:
                        print("Formula already generated, retrying...")
                        continue

                    # Check if formula is interesting for each sample size
                    for current_sample_size in range(SAMPLE_SIZE_START_VALUE, SAMPLE_SIZE_START_VALUE + SAMPLE_SIZE_INCREMENT * SAMPLE_SIZE_INCREMENT_ITERATIONS, SAMPLE_SIZE_INCREMENT):
                
                        # Store the results for the current formula and sample size
                        configuration_results[str((D, C))][i_form][current_sample_size] = {}

                        # Check if formula is interesting
                        rand_ds = random_strings_dataset(current_sample_size, TRACE_LENGTH, ["a", "b", "c"])
                        rand_ds = expand_dataset_with_end_of_trace_symbol(rand_ds).to(device)
                        dfa = DFA(formula.replace("i", "->").replace("e", "<->"), 3, "random DNF declare", ["c0", "c1", "c2", "end"])
                        deep_dfa = dfa.return_deep_dfa().to(device)
                        sat_rate = evaluate_compliance_with_formula(deep_dfa, rand_ds)

                        if sat_rate < 0.1 or sat_rate > 0.9:
                            print("Formula not interesting, retrying...")
                            break
                        else:
                            configuration_results[str((D, C))][i_form][current_sample_size]["sat_rate"] = sat_rate

                # Store the satisfaction rate for the current formula and sample size
                configuration_results[str((D, C))][i_form]["formula"] = formula
                print("Generated DNF formula:", formula)
                DNF_formulas_infix.append(formula.replace("i", "->").replace("e", "<->"))

                formula = "(" + formula + ") & (G((c0 i (! c1 & ! c2)) & (c1 i (! c0 & ! c2)) & (c2 i (! c0 & ! c1)) & (c0 | c1 | c2)))"  # Declare assumption
                print("+ declare assumption:", formula)
                formula_prefix = infix_to_prefix(formula)
                print("in prefix format:", formula_prefix)
                formula_scarlet, _ = prefix_LTL_to_scarlet(formula_prefix)
                print("in scarlet format:", formula_scarlet)
                formula_scarlet = formula_scarlet.replace("c0", "a")
                formula_scarlet = formula_scarlet.replace("c1", "b")
                formula_scarlet = formula_scarlet.replace("c2", "c")
                # formula_scarlet = formula_scarlet.replace("c3", "d")
                # formula_scarlet = formula_scarlet.replace("c4", "e")
                f.write(formula_scarlet + ";a,b,c\n")

        with open(pathlib.Path(results_config_folder, "formulas_dec_infix.txt"), "w+") as f:
            for form in DNF_formulas_infix:
                f.write(form + "\n")

        # Run experiments for different sample sizes
        for current_sample_size in range(SAMPLE_SIZE_START_VALUE, SAMPLE_SIZE_START_VALUE + SAMPLE_SIZE_INCREMENT * SAMPLE_SIZE_INCREMENT_ITERATIONS, SAMPLE_SIZE_INCREMENT):
            
            # Dataset folder for current_sample_size
            dataset_sample_size_folder = pathlib.Path(dataset_folder, f"sample_size_{current_sample_size}")

            # Generate positive traces from formulas
            generator = SampleGenerator(
                formula_file=str(formulas_dec_file),
                trace_lengths=[(TRACE_LENGTH, TRACE_LENGTH)],
                sample_sizes=[(current_sample_size, current_sample_size)],
                output_folder=str(dataset_sample_size_folder),
            )
            generator.generate()

            # Write traces on .dat files
            dataset_file_name = pathlib.Path(dataset_sample_size_folder, f"dataset_declare_D={D}_C={C}_{current_sample_size}_FORMULANUMBER.dat")
            scarlet_traces_to_stlnet_format(str(dataset_sample_size_folder) + "/TracesFiles", str(dataset_file_name))

            # Run experiments for each formula
            for i_form, formula in enumerate(DNF_formulas_infix):
                configuration_results[str((D, C))][i_form][current_sample_size]["results"] = {}

                # DFA formula evaluator
                dfa = DFA(formula, NVAR, "random DNF declare", ["c0", "c1", "c2", "end"])
                deep_dfa = dfa.return_deep_dfa()

                # Dataset
                dataset = torch.tensor(np.loadtxt(str(dataset_file_name).replace("FORMULANUMBER", str(i_form))))  # pylint: disable=no-member
                dataset = dataset.view(dataset.size(0), -1, NVAR)
                dataset = expand_dataset_with_end_of_trace_symbol(dataset)
                dataset = dataset.float()
                num_traces = dataset.size()[0]

                # Splitting in train and test
                train_dataset = dataset[: int(TRAIN_RATIO * num_traces)]
                test_dataset = dataset[int(TRAIN_RATIO * num_traces) :]

                # Variables to store the results of each experiment of the current formula, and for each prefix length value
                formula_experiment_results = {}
                for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                    formula_experiment_results[current_prefix_len] = {
                        # RNN results
                        "train_acc_rnn": [],
                        "test_acc_rnn": [],
                        "train_DL_rnn": [],
                        "test_DL_rnn": [],
                        "train_sat_rnn": [],
                        "test_sat_rnn": [],
                        # RNN+BK results
                        "train_acc_rnn_bk": [],
                        "test_acc_rnn_bk": [],
                        "train_DL_rnn_bk": [],
                        "test_DL_rnn_bk": [],
                        "train_sat_rnn_bk": [],
                        "test_sat_rnn_bk": [],
                        # RNN Greedy results
                        "train_acc_rnn_greedy": [],
                        "test_acc_rnn_greedy": [],
                        "train_DL_rnn_greedy": [],
                        "test_DL_rnn_greedy": [],
                        "train_sat_rnn_greedy": [],
                        "test_sat_rnn_greedy": [],
                        # RNN+BK Greedy results
                        "train_acc_rnn_bk_greedy": [],
                        "test_acc_rnn_bk_greedy": [],
                        "train_DL_rnn_bk_greedy": [],
                        "test_DL_rnn_bk_greedy": [],
                        "train_sat_rnn_bk_greedy": [],
                        "test_sat_rnn_bk_greedy": []
                    }

                # Run N_EXPERIMENTS_PER_FORMULA experiments for each formula
                for exp in range(N_EXPERIMENTS_PER_FORMULA):
                    # Models
                    rnn = LSTM_model(HIDDEN_DIM, NVAR + 1, NVAR + 1)
                    rnn_bk = deepcopy(rnn)

                    ########################################################
                    # Experiment with RNN and RNN Greedy
                    ########################################################

                    # Instantiate model
                    model = deepcopy(rnn).to(device)

                    # Training
                    train_acc, test_acc = train(model, train_dataset, test_dataset, MAX_NUM_EPOCHS, EPSILON)
                    
                    # Save the model
                    model_file = pathlib.Path(results_config_folder, f"model_rnn_formula_{i_form}_sample_size_{current_sample_size}_exp_{exp}.pt")
                    torch.save(model.state_dict(), model_file)

                    # We save the results for all prefix length values cause the training is the same for each value
                    for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                        formula_experiment_results[current_prefix_len]["train_acc_rnn"].append(train_acc)
                        formula_experiment_results[current_prefix_len]["test_acc_rnn"].append(test_acc)
                        formula_experiment_results[current_prefix_len]["train_acc_rnn_greedy"].append(train_acc)
                        formula_experiment_results[current_prefix_len]["test_acc_rnn_greedy"].append(test_acc)

                    # RNN Suffix prediction with temperature
                    for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                        train_predicted_traces = suffix_prediction_with_temperature_with_stop(model, train_dataset, current_prefix_len, temperature=TEMPERATURE)
                        test_predicted_traces = suffix_prediction_with_temperature_with_stop(model, test_dataset, current_prefix_len, temperature=TEMPERATURE)

                        # Evaluating compliance with the formula of stochastic sampling
                        train_sat = evaluate_compliance_with_formula(deep_dfa, train_predicted_traces)
                        test_sat = evaluate_compliance_with_formula(deep_dfa, test_predicted_traces)
                        formula_experiment_results[current_prefix_len]["train_sat_rnn"].append(train_sat)
                        formula_experiment_results[current_prefix_len]["test_sat_rnn"].append(test_sat)

                        # Evaluating DL distance
                        train_DL = evaluate_DL_distance(train_predicted_traces, train_dataset)
                        test_DL = evaluate_DL_distance(test_predicted_traces, test_dataset)
                        formula_experiment_results[current_prefix_len]["train_DL_rnn"].append(train_DL)
                        formula_experiment_results[current_prefix_len]["test_DL_rnn"].append(test_DL)

                        print(f"____________________RNN TEMPERATURE PREDICTION formula {i_form} / sample size {current_sample_size} / experiment {exp} / prefix length {current_prefix_len}____________________")
                        print(f"Satisfaction of formula {i_form}:")
                        print("- Train: ", train_sat)
                        print("- Test: ", test_sat)
                        print("DL distance:")
                        print("- Train: ", train_DL)
                        print("- Test: ", test_DL)

                    # RNN greedy suffix prediction
                    for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                        train_predicted_traces = greedy_suffix_prediction_with_stop(model, train_dataset, current_prefix_len)
                        test_predicted_traces = greedy_suffix_prediction_with_stop(model, test_dataset, current_prefix_len)

                        # Evaluating compliance with the formula of stochastic sampling
                        train_sat = evaluate_compliance_with_formula(deep_dfa, train_predicted_traces)
                        test_sat = evaluate_compliance_with_formula(deep_dfa, test_predicted_traces)
                        formula_experiment_results[current_prefix_len]["train_sat_rnn_greedy"].append(train_sat)
                        formula_experiment_results[current_prefix_len]["test_sat_rnn_greedy"].append(test_sat)

                        # Evaluating DL distance
                        train_DL = evaluate_DL_distance(train_predicted_traces, train_dataset)
                        test_DL = evaluate_DL_distance(test_predicted_traces, test_dataset)
                        formula_experiment_results[current_prefix_len]["train_DL_rnn_greedy"].append(train_DL)
                        formula_experiment_results[current_prefix_len]["test_DL_rnn_greedy"].append(test_DL)

                        print(f"____________________RNN GREEDY PREDICTION formula {i_form} / sample size {current_sample_size} / experiment {exp} / prefix length {current_prefix_len}____________________")
                        print(f"Satisfaction of formula {i_form}:")
                        print("- Train: ", train_sat)
                        print("- Test: ", test_sat)
                        print("DL distance:")
                        print("- Train: ", train_DL)
                        print("- Test: ", test_DL)

                    ########################################################
                    # Experiment RNN+BK
                    ########################################################
                    
                    for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                        # Instantiate model
                        model = deepcopy(rnn_bk).to(device)
                        
                        # Training
                        train_acc, test_acc = train(model, train_dataset, test_dataset, MAX_NUM_EPOCHS, EPSILON, deepdfa=deep_dfa, prefix_len=current_prefix_len)
                        
                        # Save the model
                        model_file = pathlib.Path(results_config_folder, f"model_rnn_bk_formula_{i_form}_sample_size_{current_sample_size}_exp_{exp}_prefix_len_{current_prefix_len}.pt")
                        torch.save(model.state_dict(), model_file)

                        # Save the results for all prefix length values cause the training is the same for each value
                        formula_experiment_results[current_prefix_len]["train_acc_rnn_bk"].append(train_acc)
                        formula_experiment_results[current_prefix_len]["test_acc_rnn_bk"].append(test_acc)
                        formula_experiment_results[current_prefix_len]["train_acc_rnn_bk_greedy"].append(train_acc)
                        formula_experiment_results[current_prefix_len]["test_acc_rnn_bk_greedy"].append(test_acc)

                        # Suffix prediction with temperature
                        train_predicted_traces = suffix_prediction_with_temperature_with_stop(model, train_dataset, current_prefix_len, temperature=TEMPERATURE)
                        test_predicted_traces = suffix_prediction_with_temperature_with_stop(model, test_dataset, current_prefix_len, temperature=TEMPERATURE)

                        # Evaluating compliance with the formula of stochastic sampling
                        train_sat = evaluate_compliance_with_formula(deep_dfa, train_predicted_traces)
                        test_sat = evaluate_compliance_with_formula(deep_dfa, test_predicted_traces)
                        formula_experiment_results[current_prefix_len]["train_sat_rnn_bk"].append(train_sat)
                        formula_experiment_results[current_prefix_len]["test_sat_rnn_bk"].append(test_sat)

                        # Evaluating DL distance
                        train_DL = evaluate_DL_distance(train_predicted_traces, train_dataset)
                        test_DL = evaluate_DL_distance(test_predicted_traces, test_dataset)
                        formula_experiment_results[current_prefix_len]["train_DL_rnn_bk"].append(train_DL)
                        formula_experiment_results[current_prefix_len]["test_DL_rnn_bk"].append(test_DL)

                        print(f"____________________RNN+BK TEMPERATURE PREDICTION formula {i_form} / sample size {current_sample_size} / experiment {exp} / prefix length {current_prefix_len}____________________")
                        print("Satisfaction:")
                        print("- Train: ", train_sat)
                        print("- Test: ", test_sat)
                        print("DL distance:")
                        print("- Train: ", train_DL)
                        print("- Test: ", test_DL)

                        # Greedy suffix prediction
                        train_predicted_traces = greedy_suffix_prediction_with_stop(model, train_dataset, current_prefix_len)
                        test_predicted_traces = greedy_suffix_prediction_with_stop(model, test_dataset, current_prefix_len)

                        # Evaluating compliance with the formula of stochastic sampling
                        train_sat = evaluate_compliance_with_formula(deep_dfa, train_predicted_traces)
                        test_sat = evaluate_compliance_with_formula(deep_dfa, test_predicted_traces)
                        formula_experiment_results[current_prefix_len]["train_sat_rnn_bk_greedy"].append(train_sat)
                        formula_experiment_results[current_prefix_len]["test_sat_rnn_bk_greedy"].append(test_sat)

                        # Evaluating DL distance
                        train_DL = evaluate_DL_distance(train_predicted_traces, train_dataset)
                        test_DL = evaluate_DL_distance(test_predicted_traces, test_dataset)
                        formula_experiment_results[current_prefix_len]["train_DL_rnn_bk_greedy"].append(train_DL)
                        formula_experiment_results[current_prefix_len]["test_DL_rnn_bk_greedy"].append(test_DL)

                        print(f"____________________RNN+BK GREEDY PREDICTION formula {i_form} / sample size {current_sample_size} / experiment {exp} / prefix length {current_prefix_len}____________________")
                        print("Satisfaction:")
                        print("- Train: ", train_sat)
                        print("- Test: ", test_sat)
                        print("DL distance:")
                        print("- Train: ", train_DL)
                        print("- Test: ", test_DL)

                    # Track time of execution
                    end_time = time.time()
                    print(f"Execution time for experiment {exp}: ", end_time - start_time)

                    # Save the results of the experiment number {exp} for the current formula
                    configuration_results[str((D, C))][i_form][current_sample_size]["results"] = formula_experiment_results
                    # Save in text file
                    results_config_file = pathlib.Path(results_config_folder, "results.txt")
                    with open(results_config_file, "a") as f:
                        sat_rate = configuration_results[str((D, C))][i_form][current_sample_size]["sat_rate"]
                        f.write(f"____________{i_form=}___{current_sample_size=}___{sat_rate=}___{exp=}____________\n")
                        for current_prefix_len in range(PREFIX_LEN_START_VALUE, PREFIX_LEN_START_VALUE + PREFIX_LEN_INCREMENT * PREFIX_LEN_INCREMENT_ITERATIONS, PREFIX_LEN_INCREMENT):
                            f.write(f"- Prefix length: {current_prefix_len}\n")
                            f.write("train acc next activity:\nRNN:{}\tRNN+BK:{}\tRNN Greedy:{}\tRNN+BK Greedy:{}\n".format(mean(formula_experiment_results[current_prefix_len]["train_acc_rnn"]), mean(formula_experiment_results[current_prefix_len]["train_acc_rnn_bk"]), mean(formula_experiment_results[current_prefix_len]["train_acc_rnn_greedy"]), mean(formula_experiment_results[current_prefix_len]["train_acc_rnn_bk_greedy"])))
                            f.write("test acc next activity:\nRNN:{}\tRNN+BK:{}\tRNN Greedy:{}\tRNN+BK Greedy:{}\n\n".format(mean(formula_experiment_results[current_prefix_len]["test_acc_rnn"]), mean(formula_experiment_results[current_prefix_len]["test_acc_rnn_bk"]), mean(formula_experiment_results[current_prefix_len]["test_acc_rnn_greedy"]), mean(formula_experiment_results[current_prefix_len]["test_acc_rnn_bk_greedy"])))
                            f.write("train DL distance:\nRNN:{}\tRNN+BK:{}\tRNN Greedy:{}\tRNN+BK Greedy:{}\n".format(mean(formula_experiment_results[current_prefix_len]["train_DL_rnn"]), mean(formula_experiment_results[current_prefix_len]["train_DL_rnn_bk"]), mean(formula_experiment_results[current_prefix_len]["train_DL_rnn_greedy"]), mean(formula_experiment_results[current_prefix_len]["train_DL_rnn_bk_greedy"])))
                            f.write("test DL distance:\nRNN:{}\tRNN+BK:{}\tRNN Greedy:{}\tRNN+BK Greedy:{}\n\n".format(mean(formula_experiment_results[current_prefix_len]["test_DL_rnn"]), mean(formula_experiment_results[current_prefix_len]["test_DL_rnn_bk"]), mean(formula_experiment_results[current_prefix_len]["test_DL_rnn_greedy"]), mean(formula_experiment_results[current_prefix_len]["test_DL_rnn_bk_greedy"])))
                            f.write("train sat suffix:\nRNN:{}\tRNN+BK:{}\tRNN Greedy:{}\tRNN+BK Greedy:{}\n".format(mean(formula_experiment_results[current_prefix_len]["train_sat_rnn"]), mean(formula_experiment_results[current_prefix_len]["train_sat_rnn_bk"]), mean(formula_experiment_results[current_prefix_len]["train_sat_rnn_greedy"]), mean(formula_experiment_results[current_prefix_len]["train_sat_rnn_bk_greedy"])))
                            f.write("test sat suffix:\nRNN:{}\tRNN+BK:{}\tRNN Greedy:{}\tRNN+BK Greedy:{}\n".format(mean(formula_experiment_results[current_prefix_len]["test_sat_rnn"]), mean(formula_experiment_results[current_prefix_len]["test_sat_rnn_bk"]), mean(formula_experiment_results[current_prefix_len]["test_sat_rnn_greedy"]), mean(formula_experiment_results[current_prefix_len]["test_sat_rnn_bk_greedy"])))
                            f.write("\n")
                        f.write("Execution time: {}\n\n".format(end_time - start_time))
                    # Save in JSON file
                    results_config_json_file = pathlib.Path(results_config_folder, "results.json")
                    with open(results_config_json_file, "w+") as f:
                        json.dump(configuration_results, f, indent=4)



if __name__ == "__main__":
    # NOTA: se faccio il suffix con la greedy search tende a non mettere mai il simbolo end
    # NOTA 2: se faccio il suffix con il random sampling con temperatura invece tende a non soddisfare i vincoli quando la probabilità è tutta bassa

    try:
        main()
    except Exception as e:
        # Save the exception to the results file
        with open("results/exceptions.txt", "a") as f:
            f.write(f"{str(e)} \n\n")
        raise e
