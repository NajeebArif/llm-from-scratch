
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch


def visualize_network(model, X):
    """
    Visualize a PyTorch feed-forward neural network.

    Designed for learning:
      - Shows network architecture
      - Shows neurons and connections
      - Shows actual activations
      - Shows tensor shapes
      - Shows weights and biases
      - Shows ReLU effect
    """

    model.eval()

    # ---------------------------------------------------------
    # 1. Capture values flowing through the network
    # ---------------------------------------------------------

    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            activations[name] = {
                "input": inputs[0].detach().clone(),
                "output": output.detach().clone()
            }
        return hook

    for i, layer in enumerate(model.layers):
        hooks.append(
            layer.register_forward_hook(
                make_hook(f"{i}_{layer.__class__.__name__}")
            )
        )

    with torch.no_grad():
        final_output = model(X)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # ---------------------------------------------------------
    # 2. Extract layers
    # ---------------------------------------------------------

    linear_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, torch.nn.Linear)
    ]

    # ---------------------------------------------------------
    # 3. Create architecture figure
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(16, 9))

    ax.set_xlim(-1, len(linear_layers))
    ax.set_ylim(-2, 13)
    ax.axis("off")

    # ---------------------------------------------------------
    # 4. Determine displayed neuron counts
    #
    # We don't draw all 50 input neurons.
    # Instead we show representative neurons.
    # ---------------------------------------------------------

    actual_sizes = [X.shape[-1]]

    for layer in linear_layers:
        actual_sizes.append(layer.out_features)

    max_display = 10

    display_sizes = [
        min(size, max_display)
        for size in actual_sizes
    ]

    # ---------------------------------------------------------
    # 5. Calculate neuron positions
    # ---------------------------------------------------------

    positions = []

    for layer_index, size in enumerate(display_sizes):

        y_positions = torch.linspace(
            1,
            11,
            size
        ).tolist()

        positions.append([
            (layer_index, y)
            for y in y_positions
        ])

    # ---------------------------------------------------------
    # 6. Draw connections
    # ---------------------------------------------------------

    for layer_index in range(len(positions) - 1):

        left = positions[layer_index]
        right = positions[layer_index + 1]

        for x1, y1 in left:
            for x2, y2 in right:

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    linewidth=0.6,
                    alpha=0.15
                )

    # ---------------------------------------------------------
    # 7. Draw neurons
    # ---------------------------------------------------------

    for layer_index, layer_positions in enumerate(positions):

        actual_size = actual_sizes[layer_index]

        for neuron_index, (x, y) in enumerate(layer_positions):

            circle = Circle(
                (x, y),
                radius=0.18,
                linewidth=1.5
            )

            ax.add_patch(circle)

            # Show neuron number
            if actual_size <= max_display:

                ax.text(
                    x,
                    y,
                    str(neuron_index + 1),
                    ha="center",
                    va="center",
                    fontsize=8
                )

        # Layer title
        if layer_index == 0:
            title = "INPUT"
        elif layer_index == len(positions) - 1:
            title = "OUTPUT"
        else:
            title = f"HIDDEN {layer_index}"

        ax.text(
            layer_index,
            12,
            title,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold"
        )

        ax.text(
            layer_index,
            11.5,
            f"{actual_size} neurons",
            ha="center",
            va="center",
            fontsize=10
        )

        # Indicate omitted neurons
        if actual_size > max_display:

            ax.text(
                layer_index,
                6,
                "⋮",
                ha="center",
                va="center",
                fontsize=25
            )

    # ---------------------------------------------------------
    # 8. Add layer descriptions
    # ---------------------------------------------------------

    for i, layer in enumerate(linear_layers):

        ax.text(
            i + 0.5,
            0.3,
            f"Linear\n{layer.in_features} → {layer.out_features}",
            ha="center",
            va="center",
            fontsize=10
        )

        # ReLU follows every hidden Linear
        if i < len(linear_layers) - 1:

            ax.text(
                i + 0.5,
                1.0,
                "↓ ReLU",
                ha="center",
                va="center",
                fontsize=10
            )

    # ---------------------------------------------------------
    # 9. Show tensor shapes
    # ---------------------------------------------------------

    shape_text = "Tensor flow:\n\n"

    shape_text += f"Input:       {tuple(X.shape)}\n"

    for layer in linear_layers:

        shape_text += (
            f"Linear {layer.in_features} → "
            f"{layer.out_features}: "
            f"(1, {layer.out_features})\n"
        )

    shape_text += (
        f"\nOutput:      {tuple(final_output.shape)}"
    )

    ax.text(
        len(linear_layers) + 0.5,
        6,
        shape_text,
        fontsize=10,
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            alpha=0.1
        )
    )

    plt.title(
        "Neural Network — Data Flow",
        fontsize=20,
        fontweight="bold"
    )

    plt.show()

    # ---------------------------------------------------------
    # 10. Print detailed information
    # ---------------------------------------------------------

    print("=" * 70)
    print("NETWORK INFORMATION")
    print("=" * 70)

    print(f"\nInput:")
    print(f"  Shape: {tuple(X.shape)}")
    print(f"  Values: {X.flatten()[:10].tolist()} ...")

    print("\n" + "-" * 70)

    total_params = 0

    for name, layer in model.named_modules():

        if isinstance(layer, torch.nn.Linear):

            weight_params = layer.weight.numel()
            bias_params = layer.bias.numel()

            layer_params = weight_params + bias_params
            total_params += layer_params

            print(f"\n{name}")
            print(f"  Linear: {layer.in_features} → {layer.out_features}")

            print(
                f"  Weight shape: "
                f"{tuple(layer.weight.shape)}"
            )

            print(
                f"  Bias shape: "
                f"{tuple(layer.bias.shape)}"
            )

            print(
                f"  Parameters: "
                f"{layer_params:,}"
            )

            print(
                f"  Weight range: "
                f"{layer.weight.min().item():.4f} "
                f"to "
                f"{layer.weight.max().item():.4f}"
            )

            print(
                f"  Bias range: "
                f"{layer.bias.min().item():.4f} "
                f"to "
                f"{layer.bias.max().item():.4f}"
            )

    print("\n" + "=" * 70)
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print("=" * 70)

    # ---------------------------------------------------------
    # 11. Show activations
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ACTIVATIONS")
    print("=" * 70)

    for name, values in activations.items():

        output = values["output"]

        print(f"\n{name}")
        print(f"  Shape: {tuple(output.shape)}")

        flat = output.flatten()

        print(
            f"  Min: {flat.min().item():.4f}"
        )

        print(
            f"  Max: {flat.max().item():.4f}"
        )

        print(
            f"  Mean: {flat.mean().item():.4f}"
        )

        print(
            f"  Positive values: "
            f"{(flat > 0).sum().item()} / {flat.numel()}"
        )

        print(
            f"  First values: "
            f"{flat[:10].tolist()}"
        )


def visualize_relu(model, X):

    model.eval()

    with torch.no_grad():

        x = X

        # First Linear
        linear1 = model.layers[0]
        relu1 = model.layers[1]

        before_relu = linear1(x)
        after_relu = relu1(before_relu)

    values_before = before_relu.flatten().numpy()
    values_after = after_relu.flatten().numpy()

    fig, ax = plt.subplots(figsize=(14, 6))

    indices = range(len(values_before))

    ax.plot(
        indices,
        values_before,
        marker="o",
        label="Before ReLU"
    )

    ax.plot(
        indices,
        values_after,
        marker="o",
        label="After ReLU"
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_title(
        "What ReLU Does to the 30 Hidden Neurons",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel("Neuron")
    ax.set_ylabel("Activation")

    ax.legend()

    ax.grid(alpha=0.2)

    plt.show()


def visualize_weights(model):

    linear_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, torch.nn.Linear)
    ]

    for index, layer in enumerate(linear_layers):

        weights = layer.weight.detach().numpy()

        fig, ax = plt.subplots(figsize=(10, 5))

        image = ax.imshow(
            weights,
            aspect="auto"
        )

        ax.set_title(
            f"Layer {index + 1}: "
            f"Weight Matrix "
            f"({layer.out_features} × {layer.in_features})",
            fontsize=16,
            fontweight="bold"
        )

        ax.set_xlabel(
            f"Input features ({layer.in_features})"
        )

        ax.set_ylabel(
            f"Neurons ({layer.out_features})"
        )

        fig.colorbar(
            image,
            ax=ax,
            label="Weight value"
        )

        plt.show()

def visualize_data_flow(model, X):

    model.eval()

    values = [X.detach().flatten()]

    x = X

    for layer in model.layers:

        with torch.no_grad():
            x = layer(x)

        if isinstance(layer, (torch.nn.Linear, torch.nn.ReLU)):
            values.append(x.detach().flatten())

    names = [
        "Input\n50",
        "Linear\n50 → 30",
        "ReLU\n30",
        "Linear\n30 → 20",
        "ReLU\n20",
        "Output\n20 → 3"
    ]

    fig, axes = plt.subplots(
        len(values),
        1,
        figsize=(14, 14)
    )

    for ax, data, name in zip(axes, values, names):

        data = data.numpy()

        ax.bar(range(len(data)), data)

        ax.axhline(0, linewidth=1)

        ax.set_title(
            f"{name}     shape = ({len(data)},)",
            loc="left",
            fontweight="bold"
        )

        ax.set_xlim(-1, len(data))

    fig.suptitle(
        "One Input Sample Moving Through the Neural Network",
        fontsize=20,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()

