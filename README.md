# Book Repository

https://github.com/rasbt/LLMs-from-scratch

## Directory organization


Sebastian Raschka's *Build a Large Language Model (From Scratch)* is specifically built around Python and PyTorch, progressing through embeddings, attention, GPT architecture, pretraining, and fine-tuning. ([Sebastian Raschka, PhD][1])

### The setup I'd use

```text
D:\workspace\AI\LLM\llm-from-scratch\
│
├── .venv\
├── notebooks\
│   ├── 01_tokens.ipynb
│   ├── 02_embeddings.ipynb
│   ├── 03_attention.ipynb
│   └── ...
│
├── src\
│   ├── model.py
│   ├── attention.py
│   ├── tokenizer.py
│   └── train.py
│
├── scripts\
├── checkpoints\
├── data\
├── pyproject.toml
└── README.md
```

Or


Build the book as a real software project.

For example:

```text
llm-from-scratch/
│
├── notebooks/
│    ├── ch02_text.ipynb
│    ├── ch03_attention.ipynb
│    └── ch04_gpt.ipynb
│
├── src/
│    └── llm/
│         ├── tokenizer.py
│         ├── embeddings.py
│         ├── attention.py
│         ├── transformer.py
│         ├── model.py
│         └── trainer.py
│
├── tests/
│    ├── test_attention.py
│    ├── test_tokenizer.py
│    └── test_model.py
│
├── scripts/
│    ├── check_gpu.py
│    └── train.py
│
└── pyproject.toml
```

That will make the concepts line up beautifully with your existing Java/software-architecture background.




### Installing PyTorch

- run the `nvidia-smi` to check the cude version
- use the command `uv add torch --index-url https://download.pytorch.org/whl/cu132`
- Install with other PyTorch packages: `uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`


### Usage of `nn_visualization_old.py`

Yes. I went ahead and redesigned the module rather than just patching the existing functions.

The original file had several things that would prevent it from being genuinely reusable — most importantly, it assumed `model.layers`, hardcoded the ReLU/Linear structure, hardcoded the tensor dimensions, and the data-flow labels were literally `"50 → 30"`, `"30 → 20"`, etc. 

### What I changed

The new `nn_visualization_old.py` is designed around a **single facade**:

```python
from nn_visualization_old import visualize_nn

visualize_nn(model, X)
```

It automatically introspects the model and captures its execution using PyTorch forward hooks.

You can also do:

```python
visualize_nn(model, X, show="all")
```

or selectively:

```python
visualize_nn(
    model,
    X,
    show=[
        "architecture",
        "data_flow",
        "activations",
        "weights",
        "relu",
        "neuron",
        "animation",
    ],
)
```

### Available visualizations

| Visualization  | What it shows                                                                   |
| -------------- | ------------------------------------------------------------------------------- |
| `architecture` | Network topology, representative neurons, connections, shapes and actual values |
| `data_flow`    | Values flowing through each module                                              |
| `activations`  | Min/mean/max and positive activation statistics                                 |
| `weights`      | Weight matrices for applicable layers                                           |
| `relu`         | Before/after ReLU, including how many values were zeroed                        |
| `neuron`       | **Actual calculation inside one neuron**: inputs × weights + bias               |
| `animation`    | Animated sample moving through the network                                      |

And:

```python
visualize_nn(model, X, show="default")
```

gives the educational default:

* architecture
* data flow
* activation statistics

### Animation

This is one of the bigger additions.

```python
visualize_nn(
    model,
    X,
    show="animation",
)
```

will display the animation **directly inside Jupyter**.

And:

```python
visualize_nn(
    model,
    X,
    show="animation",
    save=True,
)
```

also saves an HTML animation under:

```text
assets/
└── nn_visualizations/
    └── neuralnetwork_data_flow_animation.html
```

So you get both the notebook experience and persistent assets.

### Assets

For example:

```python
visualize_nn(
    model,
    X,
    show="all",
    save=True,
)
```

will create an asset directory such as:

```text
assets/
└── nn_visualizations/
    ├── neuralnetwork_architecture.png
    ├── neuralnetwork_data_flow.png
    ├── neuralnetwork_activations.png
    ├── neuralnetwork_weights_0_layers.0.png
    ├── neuralnetwork_weights_1_layers.2.png
    ├── neuralnetwork_weights_2_layers.4.png
    ├── neuralnetwork_relu_0_layers.1.png
    ├── neuralnetwork_relu_1_layers.3.png
    ├── neuralnetwork_neuron_layer_0_neuron_0.png
    └── neuralnetwork_data_flow_animation.html
```

The filenames are generated from the actual model/module names.

### Most importantly: no `model.layers`

I changed the implementation to inspect the actual PyTorch module tree:

```python
model.named_modules()
```

and identify leaf modules dynamically.

So your current:

```text
Sequential
 ├── Linear
 ├── ReLU
 ├── Linear
 ├── ReLU
 └── Linear
```

works, but so would a model containing nested `Sequential`s or other module structures.

The module also correctly calculates your current network's:

```text
50 → 30 → ReLU → 30 → 20 → ReLU → 20 → 3
```

and detects **2,213 parameters** rather than having that information embedded anywhere.

I also tested the generic introspection against your current `NeuralNetwork` and verified that it discovers:

```text
layers.0   Linear
layers.1   ReLU
layers.2   Linear
layers.3   ReLU
layers.4   Linear
```

with output shapes `(1,30)`, `(1,30)`, `(1,20)`, `(1,20)`, `(1,3)`.

### A particularly useful addition for your book

You can now do:

```python
visualize_nn(
    model,
    X,
    show="neuron",
    layer=0,
    neuron_index=3,
)
```

This is intended specifically for understanding the math behind the network.

It takes the **real values from your model** and shows the calculation conceptually as:

```text
                 Linear layer

x[0]  × w[0] ─┐
x[1]  × w[1] ─┤
x[2]  × w[2] ─┤
...           ├── Σ ── + bias ──> z
x[n]  × w[n] ─┘

                         ↓

                    pre-activation

                         ↓ ReLU

                     activation
```

So when you're reading the book's discussion of:

```text
y = xWᵀ + b
```

you'll be able to see an actual neuron performing that calculation.

---

## Download the new module

[Download `nn_visualization_old.py`](sandbox:/mnt/data/nn_visualization.py)

Put it directly in your project root:

```text
llm-from-scratch/
│
├── nn_visualization.py
│
├── notebooks/
│   └── ...
│
├── assets/
│   └── nn_visualizations/
│
├── pyproject.toml
└── uv.lock
```

Then from your notebook:

```python
from nn_visualization_old import visualize_nn
```

and start with simply:

```python
visualize_nn(model, X)
```

**One caveat:** this version intentionally uses **PyTorch + Matplotlib**, but does **not** require `torchviz` or the Graphviz `dot.exe` installation that was causing your original problem. It also doesn't add another visualization framework just for the sake of it.

I would use this as the foundation and then, as you encounter CNNs, embeddings, attention, transformers, etc. in the book, we can extend the same facade with specialized visualizations rather than replacing it.


---------

Done. I modified the module into a **teaching/debug version**, rather than just a network-inspection utility.

### Download the updated module

[Download `nn_visualization.py`](sandbox:/mnt/data/nn_visualization.py)

I also validated that the new module imports correctly, captures the forward pass, and successfully generates/saves the animation HTML.

### The new API

Your existing code still works:

```python
from nn_visualization import visualize_nn

visualize_nn(model, X)
```

For the learning mode you asked for:

```python
visualize_nn(model, X, debug=True)
```

This now gives you a guided walkthrough like:

```text
Step 0 — Input
       ↓
Step 1 — Linear 50 → 30
       ↓
Step 2 — ReLU 30
       ↓
Step 3 — Linear 30 → 20
       ↓
Step 4 — ReLU 20
       ↓
Step 5 — Linear 20 → 3
       ↓
Final output
```

For every step, it explains **what the layer is doing, why it is doing it, the tensor shapes, and actual values from your `X`**.

For a `Linear` layer, it even drills into **one actual neuron**:

```text
x[0] × w[0]
x[1] × w[1]
x[2] × w[2]
...
x[49] × w[49]
       ↓
sum
       +
bias
       ↓
pre-activation z
```

and connects that to:

$$
z = xW^T + b
$$

For ReLU, it explains which actual values were negative and therefore became zero.

### Animation

You can now explicitly request it:

```python
visualize_nn(
    model,
    X,
    debug=True,
    animation=True
)
```

Or save everything:

```python
result = visualize_nn(
    model,
    X,
    debug=True,
    animation=True,
    save=True
)
```

The generated assets go under:

```text
assets/
└── nn_visualizations/
```

relative to the notebook's current working directory.

The returned object also tells you exactly where they went:

```python
result["assets"]
```

and:

```python
result["asset_dir"]
```

The animation is saved as an `.html` file and is also displayed directly in Jupyter.

### One important design choice

I **didn't make `debug=True` automatically generate the animation**. Animation can be relatively large and slow, so:

```python
visualize_nn(model, X, debug=True)
```

means **"teach me the forward pass."**

Whereas:

```python
visualize_nn(model, X, debug=True, animation=True, save=True)
```

means **"teach me everything and create the reusable assets."**

I think this is a much better fit for your current stage in *Build a Large Language Model from Scratch*: you're not just trying to inspect a neural network—you want to build the intuition for **what PyTorch is actually doing mathematically**.
