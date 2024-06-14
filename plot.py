import pathlib
import json

from matplotlib import pyplot as plt
import numpy as np


def load_all_results():
    all_results = {}

    # Iterate all configs
    for config in CONFIGS:
        # Try to find the results.json file corresponding to the current config
        # by iterating through the subfolders in the RESULTS_FOLDER
        results = None
        for subfolder_results_date in RESULTS_FOLDER.iterdir():
            # Ignore file exceptions.txt
            if subfolder_results_date.is_file():
                continue
            # Iterate the subfolder_results_date and check if it contains a folder
            # with the name results_config_folder_name for the current config
            results_config_folder_name = f"declare_D={config[1]}_C={config[4]}"
            results_folder_path = pathlib.Path(subfolder_results_date, results_config_folder_name)
            if results_folder_path in subfolder_results_date.iterdir():
                results_file_path = pathlib.Path(results_folder_path, "results.json")
                # NOTE hardcoding this so not to get results from this specific folder
                if "results/results_2024-06-04_17-44-39/" in str(results_file_path) and "declare_D=5_C=1" not in str(results_file_path):
                    continue
                try:
                    with open(results_file_path, "r") as f:
                        results = json.load(f)
                    print(f"Results for configuration {config} found in {results_file_path}")
                except FileNotFoundError:
                    print(f"File {results_file_path} not found")
                break

        # Check if results for the config were found
        if results is None:
            print(f"Configuration {config} not found in any subfolder")
            continue
        else:
            all_results[config] = results[config]

    return all_results


def load_results(results_file_path):
    with open(results_file_path, "r") as f:
        results = json.load(f)
    return results


def summarise_results(results):
    for configuration in results:
        for i_form, formula_dict in results[configuration].items():
            print(f"- Configuration {configuration} formula {i_form}:")
            for sample_size in SAMPLE_SIZES:
                sample_size_dict = formula_dict[str(sample_size)]
                print(f"  - Sample size {sample_size} sat rate: {sample_size_dict['sat_rate']}")
                try:
                    prefix_lengths_dict = sample_size_dict["results"]
                    for prefix_length, prefix_length_dict in prefix_lengths_dict.items():
                        print(f"    - Prefix length {prefix_length}:")
                        print("      - RNN:")
                        print(f"        - Avg. train accuracy: {np.mean(prefix_length_dict['train_acc_rnn'])}")
                        print(f"        - Avg. test accuracy: {np.mean(prefix_length_dict['test_acc_rnn'])}")
                        print(f"        - Avg. train DL distance: {np.mean(prefix_length_dict['train_DL_rnn'])}")
                        print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn'])}")
                        print(f"        - Avg. train satisfaction rate: {np.mean(prefix_length_dict['train_sat_rnn'])}")
                        print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn'])}")
                        print("      - RNN+BK:")
                        print(f"        - Avg. train accuracy: {np.mean(prefix_length_dict['train_acc_rnn_bk'])}")
                        print(f"        - Avg. test accuracy: {np.mean(prefix_length_dict['test_acc_rnn_bk'])}")
                        print(f"        - Avg. train DL distance: {np.mean(prefix_length_dict['train_DL_rnn_bk'])}")
                        print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_bk'])}")
                        print(f"        - Avg. train satisfaction rate: {np.mean(prefix_length_dict['train_sat_rnn_bk'])}")
                        print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_bk'])}")
                        print("      - RNN Greedy:")
                        print(f"        - Avg. train accuracy: {np.mean(prefix_length_dict['train_acc_rnn_greedy'])}")
                        print(f"        - Avg. test accuracy: {np.mean(prefix_length_dict['test_acc_rnn_greedy'])}")
                        print(f"        - Avg. train DL distance: {np.mean(prefix_length_dict['train_DL_rnn_greedy'])}")
                        print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_greedy'])}")
                        print(f"        - Avg. train satisfaction rate: {np.mean(prefix_length_dict['train_sat_rnn_greedy'])}")
                        print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_greedy'])}")
                        print("      - RNN+BK Greedy:")
                        print(f"        - Avg. train accuracy: {np.mean(prefix_length_dict['train_acc_rnn_bk_greedy'])}")
                        print(f"        - Avg. test accuracy: {np.mean(prefix_length_dict['test_acc_rnn_bk_greedy'])}")
                        print(f"        - Avg. train DL distance: {np.mean(prefix_length_dict['train_DL_rnn_bk_greedy'])}")
                        print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_bk_greedy'])}")
                        print(f"        - Avg. train satisfaction rate: {np.mean(prefix_length_dict['train_sat_rnn_bk_greedy'])}")
                        print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_bk_greedy'])}")
                except KeyError:
                    print("    - NOT ENOUGH RESULTS YET")
            print("\n")
        print("\n")


def plot(configuration_name, configuration_dict, sample_size, metric):
    # Check metric
    assert metric in ["acc", "sat", "DL"]
    if metric == "acc":
        metric_name = "Accuracy"
    elif metric == "sat":
        metric_name = "Satisfiability"
    elif metric == "DL":
        metric_name = "DL distance"

    # Add D= and C= to configuration name
    # Cast configuration name to tuple
    configuration_name = configuration_name.replace(" ", "").replace("(", "").replace(")", "").split(",")
    configuration_name = f"(D={configuration_name[0]}, C={configuration_name[1]})"

    # Join the sample_size metric values of all formulas for each prefix length
    metric_values_for_sample_size_rnn_by_prefix_length = {}
    metric_values_for_sample_size_rnn_bk_by_prefix_length = {}
    metric_values_for_sample_size_rnn_greedy_by_prefix_length = {}
    metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length = {}
    for i_form, formula_dict in configuration_dict.items():
        sample_size_dict = formula_dict[str(sample_size)]
        try:
            prefix_lengths_dict = sample_size_dict["results"]
            for prefix_length, prefix_length_dict in prefix_lengths_dict.items():
                if prefix_length not in metric_values_for_sample_size_rnn_by_prefix_length:
                    metric_values_for_sample_size_rnn_by_prefix_length[prefix_length] = []
                    metric_values_for_sample_size_rnn_bk_by_prefix_length[prefix_length] = []
                    metric_values_for_sample_size_rnn_greedy_by_prefix_length[prefix_length] = []
                    metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length[prefix_length] = []
                metric_values_for_sample_size_rnn_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn"])
                metric_values_for_sample_size_rnn_bk_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn_bk"])
                metric_values_for_sample_size_rnn_greedy_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn_greedy"])
                metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn_bk_greedy"])
        except KeyError as e:
            print(f"Skipping formula {i_form} for sample size {sample_size} as not enough results yet")
            continue

    # Initialise plot
    fig, ax = plt.subplots()
    bar_width = 0.18
    bar_distance = 0.04
    index = np.arange(len(metric_values_for_sample_size_rnn_by_prefix_length))

    # Plot bars
    rnn_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_by_prefix_length.values()]
    rnn_bk_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_by_prefix_length.values()]
    rnn_greedy_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_greedy_by_prefix_length.values()]
    rnn_bk_greedy_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length.values()]
    ax.bar(index, rnn_x_values, bar_width, label="RNN", color="skyblue")
    ax.bar(index + bar_width, rnn_bk_x_values, bar_width, label="RNN+BK", color="lightgreen")
    ax.bar((index + bar_distance) + 2 * bar_width, rnn_greedy_x_values, bar_width, label="RNN Greedy", color="dodgerblue")
    ax.bar((index + bar_distance) + 3 * bar_width, rnn_bk_greedy_x_values, bar_width, label="RNN+BK Greedy", color="seagreen")

    # Plot error bars
    rnn_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_by_prefix_length.values()]
    rnn_bk_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_by_prefix_length.values()]
    rnn_greedy_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_greedy_by_prefix_length.values()]
    rnn_bk_greedy_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length.values()]
    ax.errorbar(index, rnn_x_values, rnn_error, fmt="none", ecolor="black", capsize=5)
    ax.errorbar(index + bar_width, rnn_bk_x_values, rnn_bk_error, fmt="none", ecolor="black", capsize=5)
    ax.errorbar((index + bar_distance) + 2 * bar_width, rnn_greedy_x_values, rnn_greedy_error, fmt="none", ecolor="black", capsize=5)
    ax.errorbar((index + bar_distance) + 3 * bar_width, rnn_bk_greedy_x_values, rnn_bk_greedy_error, fmt="none", ecolor="black", capsize=5)

    # If metric is DL, set y-axis values from 0 to 20
    # if metric == "DL":
    #     ax.set_ylim(0, 20)
    # If metric is acc, set y-axis values from 0 to 0.50
    if metric == "acc":
        ax.set_ylim(0, 0.50)
    # If metric is sat, set y-axis values from 0 to 1
    if metric == "sat":
        ax.set_ylim(0, 1)

    # Set plot labels
    ax.set_xlabel("Prefix length")
    ax.set_ylabel(f"{metric_name}")
    ax.set_title(f"{metric_name} by prefix length for sample size {sample_size}")
    ax.set_xticks((index + bar_distance) + 2 * bar_width - bar_width / 2 - bar_distance / 1.5)
    ax.set_xticklabels(metric_values_for_sample_size_rnn_by_prefix_length.keys())

    # Set legent
    ax.legend()

    # Save plot
    fig.tight_layout()
    fig.savefig(PLOTS_FOLDER / f"{metric}_plots" / f"config_{configuration_name}_sample_size_{sample_size}.png")
    plt.close(fig)


CONFIGS = ["(5, 1)", "(4, 2)", "(3, 3)", "(2, 4)", "(1, 5)"]
# CONFIGS = ["(5, 1)"]
# SAMPLE_SIZES = [250, 500, 750, 1000]
SAMPLE_SIZES = [1000]
PREFIX_LENGTHS = [5, 10, 15]

RESULTS_FOLDER = pathlib.Path("results")

PLOTS_FOLDER = pathlib.Path("plots")
ACC_PLOTS_FOLDER = PLOTS_FOLDER / "acc_plots"
SAT_PLOTS_FOLDER = PLOTS_FOLDER / "sat_plots"
DL_PLOTS_FOLDER = PLOTS_FOLDER / "DL_plots"
# Create plots folders if they don't exist
for folder in [PLOTS_FOLDER, ACC_PLOTS_FOLDER, SAT_PLOTS_FOLDER, DL_PLOTS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Load results
    results = load_all_results()

    # Summarise results
    # summarise_results(results)

    # Plot
    # for metric in ["acc", "sat", "DL"]:
    #     print(f"Plotting {metric} plots")
    #     for configuration_name, configuration_dict in results.items():
    #         print(f"- Configuration {configuration_name}")
    #         for sample_size in SAMPLE_SIZES:
    #             print(f"  - Sample size {sample_size}")
    #             plot(configuration_name, configuration_dict, sample_size, metric)

    # Plot for all configurations as a whole
    for metric in ["acc", "sat", "DL"]:
        print(f"Plotting {metric} plots for all configurations")
        for sample_size in SAMPLE_SIZES:
            print(f"  - Sample size {sample_size}")

            # Check metric
            assert metric in ["acc", "sat", "DL"]
            if metric == "acc":
                metric_name = "Accuracy %"
            elif metric == "sat":
                metric_name = "Satisfiability %"
            elif metric == "DL":
                metric_name = "DL distance"

            # Create configuration name
            configuration_name = "ALL CONFIGURATIONS"

            # Join the metric values of all configurations for each prefix length
            metric_values_for_sample_size_rnn_by_prefix_length = {}
            metric_values_for_sample_size_rnn_bk_by_prefix_length = {}
            metric_values_for_sample_size_rnn_greedy_by_prefix_length = {}
            metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length = {}
            for configuration_dict in results.values():
                for i_form, formula_dict in configuration_dict.items():
                    sample_size_dict = formula_dict[str(sample_size)]
                    try:
                        prefix_lengths_dict = sample_size_dict["results"]
                        for prefix_length, prefix_length_dict in prefix_lengths_dict.items():
                            if prefix_length not in metric_values_for_sample_size_rnn_by_prefix_length:
                                metric_values_for_sample_size_rnn_by_prefix_length[prefix_length] = []
                                metric_values_for_sample_size_rnn_bk_by_prefix_length[prefix_length] = []
                                metric_values_for_sample_size_rnn_greedy_by_prefix_length[prefix_length] = []
                                metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length[prefix_length] = []
                            metric_values_for_sample_size_rnn_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn"])
                            metric_values_for_sample_size_rnn_bk_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn_bk"])
                            metric_values_for_sample_size_rnn_greedy_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn_greedy"])
                            metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length[prefix_length].extend(prefix_length_dict[f"test_{metric}_rnn_bk_greedy"])
                    except KeyError:
                        print(f"Skipping formula {i_form} for sample size {sample_size} as not enough results yet")
                        continue
            
            # If metric is accuracy or satisfiability, convert to percentage
            if metric in ["acc", "sat"]:
                for prefix_length in metric_values_for_sample_size_rnn_by_prefix_length:
                    metric_values_for_sample_size_rnn_by_prefix_length[prefix_length] = [value * 100 for value in metric_values_for_sample_size_rnn_by_prefix_length[prefix_length]]
                    metric_values_for_sample_size_rnn_bk_by_prefix_length[prefix_length] = [value * 100 for value in metric_values_for_sample_size_rnn_bk_by_prefix_length[prefix_length]]
                    metric_values_for_sample_size_rnn_greedy_by_prefix_length[prefix_length] = [value * 100 for value in metric_values_for_sample_size_rnn_greedy_by_prefix_length[prefix_length]]
                    metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length[prefix_length] = [value * 100 for value in metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length[prefix_length]]

            # Initialise plot
            fig, ax = plt.subplots()
            bar_width = 0.18
            bar_distance = 0.04
            index = np.arange(len(metric_values_for_sample_size_rnn_by_prefix_length))

            # Plot bars
            rnn_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_by_prefix_length.values()]
            rnn_bk_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_by_prefix_length.values()]
            rnn_greedy_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_greedy_by_prefix_length.values()]
            rnn_bk_greedy_x_values = [np.mean(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length.values()]
            ax.bar(index, rnn_x_values, bar_width, label="RNN (random)", color="skyblue")
            ax.bar(index + bar_width, rnn_bk_x_values, bar_width, label="RNN+LTL (random)", color="lightgreen")
            ax.bar((index + bar_distance) + 2 * bar_width, rnn_greedy_x_values, bar_width, label="RNN (greedy)", color="dodgerblue")
            ax.bar((index + bar_distance) + 3 * bar_width, rnn_bk_greedy_x_values, bar_width, label="RNN+LTL (greedy)", color="seagreen")

            # Plot error bars
            rnn_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_by_prefix_length.values()]
            rnn_bk_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_by_prefix_length.values()]
            rnn_greedy_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_greedy_by_prefix_length.values()]
            rnn_bk_greedy_error = [np.std(metric_values) for metric_values in metric_values_for_sample_size_rnn_bk_greedy_by_prefix_length.values()]
            ax.errorbar(index, rnn_x_values, rnn_error, fmt="none", ecolor="black", capsize=5)
            ax.errorbar(index + bar_width, rnn_bk_x_values, rnn_bk_error, fmt="none", ecolor="black", capsize=5)
            ax.errorbar((index + bar_distance) + 2 * bar_width, rnn_greedy_x_values, rnn_greedy_error, fmt="none", ecolor="black", capsize=5)
            ax.errorbar((index + bar_distance) + 3 * bar_width, rnn_bk_greedy_x_values, rnn_bk_greedy_error, fmt="none", ecolor="black", capsize=5)

            # If metric is DL, set y-axis values from 0 to 20
            # if metric == "DL":
            #     ax.set_ylim(0, 20)
            # If metric is acc, set y-axis values from 0 to 0.50
            if metric == "acc":
                ax.set_ylim(0, 50)
            # If metric is sat, set y-axis values from 0 to 1
            if metric == "sat":
                ax.set_ylim(0, 100)

            # Set plot labels
            ax.set_xlabel("Prefix length")
            ax.set_ylabel(f"{metric_name}")
            # ax.set_title(f"{metric_name} by prefix length for sample size {sample_size}")
            ax.set_title(f"{metric_name} by prefix length")
            ax.set_xticks((index + bar_distance) + 2 * bar_width - bar_width / 2 - bar_distance / 1.5)
            ax.set_xticklabels(metric_values_for_sample_size_rnn_by_prefix_length.keys())

            # Set legend
            ax.legend(loc="upper center", ncol=4, fontsize="small", columnspacing=1.0, handletextpad=0.5, handlelength=1.5)

            # Save plot
            fig.tight_layout()
            fig.savefig(PLOTS_FOLDER / f"{metric}_plots" / f"config_{configuration_name}_sample_size_{sample_size}.pdf")
            plt.close(fig)
