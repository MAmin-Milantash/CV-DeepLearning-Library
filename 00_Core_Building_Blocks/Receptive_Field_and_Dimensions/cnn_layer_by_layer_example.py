layers = [
    {"name": "Conv1", "k": 7, "s": 2, "p": 3},
    {"name": "MaxPool", "k": 3, "s": 2, "p": 1},
    {"name": "Conv2", "k": 3, "s": 1, "p": 1},
]

input_size = 224

for layer in layers:
    input_size = (input_size + 2*layer["p"] - layer["k"]) // layer["s"] + 1
    print(f"{layer['name']} → {input_size}")
