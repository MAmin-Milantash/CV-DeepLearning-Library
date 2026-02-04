import matplotlib.pyplot as plt
import json

def log_results(params, metrics, filename="results.json"):
    with open(filename, "w") as f:
        json.dump({"params": params, "metrics": metrics}, f, indent=4)

def plot_training_curve(metrics_dict, ylabel="Loss"):
    for label, values in metrics_dict.items():
        plt.plot(values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()