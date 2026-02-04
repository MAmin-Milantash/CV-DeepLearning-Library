import json
import matplotlib.pyplot as plt

def log_results(params, score, filename="results.json"):
    with open(filename, "a") as f:
        f.write(json.dumps({"params": params, "score": score}) + "\n")

def plot_hyperparam_performance(results):
    scores = [r["score"] for r in results]
    plt.plot(scores)
    plt.xlabel("Experiment")
    plt.ylabel("Validation Loss")
    plt.show()
