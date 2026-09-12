"""
nn_visualization.py

Reusable, educational visualizations for PyTorch neural networks.

Design goals
------------
* No assumptions about a model having `model.layers`.
* Introspects any torch.nn.Module using PyTorch's module tree.
* Works with nested Sequential/custom modules for module-level visualizations.
* Displays figures directly in Jupyter.
* Can persist PNG/SVG/HTML assets under an asset directory.
* Includes architecture, data-flow, activations, weights, ReLU, neuron-detail,
  and animated data-flow views.
* Keeps the public API small: normally call `visualize_nn()`.

Dependencies
------------
PyTorch + Matplotlib. No Graphviz/torchviz required.

Typical notebook usage
----------------------
    from nn_visualization import visualize_nn

    visualize_nn(model, X)

    visualize_nn(model, X, show="architecture")
    visualize_nn(model, X, show="all")
    visualize_nn(model, X, show=["architecture", "data_flow", "weights"])

    visualize_nn(model, X, show="animation", save=True)

The returned value is a dictionary containing generated matplotlib figures and
captured runtime information, which is useful if you want to customize things.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import math
import html
import json

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ASSET_DIR = Path("assets") / "nn_visualizations"
DEFAULT_MAX_NEURONS = 10


@dataclass
class ModuleInfo:
    """Information about one executable/leaf module in the model."""

    name: str
    module: nn.Module
    module_type: str


@dataclass
class TraceEntry:
    """Input/output tensors captured for one module during a forward pass."""

    name: str
    module: nn.Module
    input: torch.Tensor | None
    output: torch.Tensor | None


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _ensure_input_tensor(X: Any) -> torch.Tensor:
    if not isinstance(X, torch.Tensor):
        raise TypeError(
            f"X must be a torch.Tensor, got {type(X).__name__}."
        )
    if X.numel() == 0:
        raise ValueError("X must contain at least one value.")
    return X


def _leaf_modules(model: nn.Module) -> list[ModuleInfo]:
    """
    Return leaf modules in forward-tree order.

    A leaf module is a module with no child modules. This avoids hardcoding
    model.layers and works with nested Sequential/custom containers.
    """
    result: list[ModuleInfo] = []

    for name, module in model.named_modules():
        if name == "":
            continue
        if not any(module.children()):
            result.append(
                ModuleInfo(
                    name=name,
                    module=module,
                    module_type=module.__class__.__name__,
                )
            )

    return result


def _safe_shape(value: Any) -> tuple | None:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, (tuple, list)):
        for item in value:
            shape = _safe_shape(item)
            if shape is not None:
                return shape
    return None


def _flatten_sample(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flatten while preserving the first batch/sample when possible.

    For visualization we use the first sample from a batch. If the tensor is
    already one-dimensional, the complete tensor is used.
    """
    t = tensor.detach().cpu()

    if t.ndim == 0:
        return t.reshape(1)

    if t.ndim == 1:
        return t

    return t[0].reshape(-1)


def _format_number(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}f}"


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().float().cpu()
    flat = t.reshape(-1)

    return {
        "shape": tuple(t.shape),
        "numel": int(t.numel()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)) if flat.numel() > 1 else 0.0,
        "positive": int((flat > 0).sum()),
        "negative": int((flat < 0).sum()),
        "zero": int((flat == 0).sum()),
    }


def _parameter_count(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _display_indices(size: int, max_neurons: int) -> list[int]:
    """
    Choose representative indices without pretending omitted neurons do not
    exist. For small layers, show everything. For large layers, show the first,
    last and evenly spaced middle neurons.
    """
    if size <= max_neurons:
        return list(range(size))

    if max_neurons < 4:
        return list(range(max_neurons))

    indices = {0, size - 1}
    remaining = max_neurons - len(indices)

    for i in range(1, remaining + 1):
        idx = round(i * (size - 1) / (remaining + 1))
        indices.add(idx)

    return sorted(indices)


def _short_module_label(info: ModuleInfo) -> str:
    module = info.module

    if isinstance(module, nn.Linear):
        return f"Linear\\n{module.in_features} → {module.out_features}"

    if isinstance(module, nn.Conv1d):
        return f"Conv1d\\n{module.in_channels} → {module.out_channels}"

    if isinstance(module, nn.Conv2d):
        return f"Conv2d\\n{module.in_channels} → {module.out_channels}"

    if isinstance(module, nn.MultiheadAttention):
        return f"MultiheadAttention\\nheads={module.num_heads}"

    if isinstance(module, nn.LayerNorm):
        return f"LayerNorm\\n{module.normalized_shape}"

    if isinstance(module, nn.BatchNorm1d):
        return f"BatchNorm1d\\n{module.num_features}"

    if isinstance(module, nn.BatchNorm2d):
        return f"BatchNorm2d\\n{module.num_features}"

    return info.module_type


def _save_figure(fig, path: Path, dpi: int = 160) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _asset_path(
    asset_dir: str | Path,
    name: str,
    extension: str,
) -> Path:
    directory = Path(asset_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        c if c.isalnum() or c in "-_." else "_"
        for c in name
    )
    return directory / f"{safe_name}.{extension.lstrip('.')}"


# ---------------------------------------------------------------------------
# Runtime tracing
# ---------------------------------------------------------------------------

def _capture_forward(model: nn.Module, X: torch.Tensor) -> tuple[
    torch.Tensor, list[TraceEntry]
]:
    """
    Run one forward pass and capture leaf-module inputs/outputs.

    Hooks are always removed, even if the model raises an exception.
    """
    traces: dict[str, TraceEntry] = {}
    hooks = []

    for info in _leaf_modules(model):
        name = info.name

        def make_hook(module_name: str, module: nn.Module):
            def hook(_module, inputs, output):
                input_tensor = None
                if inputs:
                    input_tensor = (
                        inputs[0].detach().clone()
                        if isinstance(inputs[0], torch.Tensor)
                        else None
                    )

                output_tensor = (
                    output.detach().clone()
                    if isinstance(output, torch.Tensor)
                    else None
                )

                traces[module_name] = TraceEntry(
                    name=module_name,
                    module=module,
                    input=input_tensor,
                    output=output_tensor,
                )

            return hook

        hooks.append(info.module.register_forward_hook(
            make_hook(name, info.module)
        ))

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            final_output = model(X)
    finally:
        for hook in hooks:
            hook.remove()

        if was_training:
            model.train()

    ordered = [
        traces[info.name]
        for info in _leaf_modules(model)
        if info.name in traces
    ]

    return final_output.detach(), ordered


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def _architecture_figure(
    model: nn.Module,
    X: torch.Tensor,
    traces: Sequence[TraceEntry],
    *,
    max_neurons: int = DEFAULT_MAX_NEURONS,
):
    """
    Draw a generic module-level architecture.

    Connections are drawn between representative features only. The figure
    explicitly reports the actual tensor sizes, so a 1000-neuron layer is not
    misleadingly represented as a 10-neuron layer.
    """
    stages: list[dict[str, Any]] = []

    input_sample = _flatten_sample(X)
    stages.append({
        "name": "Input",
        "type": "Input",
        "size": int(input_sample.numel()),
        "shape": tuple(X.shape),
        "values": input_sample,
    })

    for entry in traces:
        output = entry.output
        if output is None:
            continue

        values = _flatten_sample(output)

        stages.append({
            "name": entry.name or entry.module.__class__.__name__,
            "type": entry.module.__class__.__name__,
            "size": int(values.numel()),
            "shape": tuple(output.shape),
            "values": values,
        })

    n = len(stages)
    fig_width = max(14, n * 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 9))

    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(-2.0, 12.5)
    ax.axis("off")

    positions: list[list[tuple[float, float, int]]] = []

    for stage_index, stage in enumerate(stages):
        indices = _display_indices(stage["size"], max_neurons)
        count = len(indices)

        if count == 1:
            ys = [6.5]
        else:
            ys = torch.linspace(2.0, 11.0, count).tolist()

        layer_positions = [
            (float(stage_index), y, neuron_index)
            for y, neuron_index in zip(ys, indices)
        ]
        positions.append(layer_positions)

    # Connections.
    for left_index in range(len(positions) - 1):
        left = positions[left_index]
        right = positions[left_index + 1]

        for x1, y1, _ in left:
            for x2, y2, _ in right:
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    linewidth=0.65,
                    alpha=0.13,
                    zorder=1,
                )

    # Neurons.
    for stage_index, (stage, layer_positions) in enumerate(
        zip(stages, positions)
    ):
        values = stage["values"]

        for x, y, neuron_index in layer_positions:
            value = float(values[neuron_index])

            circle = plt.Circle(
                (x, y),
                radius=0.22,
                linewidth=1.4,
                fill=True,
                alpha=0.9,
                zorder=3,
            )
            ax.add_patch(circle)

            # Tooltip-like text embedded in the static figure.
            ax.text(
                x,
                y,
                str(neuron_index + 1),
                ha="center",
                va="center",
                fontsize=7,
                zorder=4,
            )

            ax.text(
                x + 0.27,
                y,
                _format_number(value, 2),
                fontsize=6.5,
                va="center",
                alpha=0.75,
            )

        if stage["size"] > len(layer_positions):
            ax.text(
                stage_index,
                6.5,
                "⋮",
                ha="center",
                va="center",
                fontsize=25,
                zorder=2,
            )

        title = stage["name"]
        if stage_index == 0:
            title = "INPUT"

        ax.text(
            stage_index,
            12.0,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        ax.text(
            stage_index,
            11.5,
            f"{stage['type']}   |   {stage['size']} values",
            ha="center",
            va="center",
            fontsize=8.5,
            alpha=0.75,
        )

        ax.text(
            stage_index,
            0.9,
            f"shape = {stage['shape']}",
            ha="center",
            va="center",
            fontsize=8,
        )

    trainable, total = _parameter_count(model)

    summary = (
        "NETWORK SUMMARY\n\n"
        f"Modules shown: {len(stages) - 1}\n"
        f"Input shape: {tuple(X.shape)}\n"
        f"Trainable parameters: {trainable:,}\n"
        f"Total parameters: {total:,}\n\n"
        "Numbers beside neurons are actual\n"
        "values from the supplied sample.\n\n"
        f"Large layers show representative neurons\n"
        f"(max {max_neurons} per stage)."
    )

    ax.text(
        n - 0.05,
        -0.4,
        summary,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.6", alpha=0.08),
    )

    fig.suptitle(
        "Neural Network — Architecture & Data Flow",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Data flow
# ---------------------------------------------------------------------------

def _data_flow_figure(
    model: nn.Module,
    X: torch.Tensor,
    traces: Sequence[TraceEntry],
    *,
    max_values: int = 80,
):
    """
    Show one sample as it moves through every captured module.
    """
    stages = [{
        "name": "Input",
        "tensor": X,
    }]

    for entry in traces:
        if entry.output is not None:
            stages.append({
                "name": entry.name or entry.module.__class__.__name__,
                "tensor": entry.output,
            })

    rows = len(stages)
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(14, max(4, rows * 2.2)),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, stage in zip(axes, stages):
        values = _flatten_sample(stage["tensor"]).numpy()

        if len(values) > max_values:
            # Keep beginning/end while making the truncation explicit.
            half = max_values // 2
            display_values = list(values[:half]) + list(values[-half:])
            x_positions = list(range(half)) + list(
                range(len(values) - half, len(values))
            )
            truncated = True
        else:
            display_values = values
            x_positions = range(len(values))
            truncated = False

        ax.bar(x_positions, display_values)
        ax.axhline(0, linewidth=0.8)
        ax.set_ylabel("value")
        ax.set_title(
            f"{stage['name']}    shape = {tuple(stage['tensor'].shape)}",
            loc="left",
            fontweight="bold",
        )

        if truncated:
            ax.set_xlabel(
                f"Showing {max_values} of {len(values)} values "
                "(first and last portions)"
            )

    fig.suptitle(
        "One Sample Moving Through the Neural Network",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Activation statistics
# ---------------------------------------------------------------------------

def _activations_figure(
    traces: Sequence[TraceEntry],
    *,
    max_values: int = 100,
):
    usable = [t for t in traces if t.output is not None]

    if not usable:
        return None

    names = []
    mins = []
    means = []
    maxs = []
    positives = []

    for entry in usable:
        stats = _tensor_stats(entry.output)
        names.append(entry.name or entry.module.__class__.__name__)
        mins.append(stats["min"])
        means.append(stats["mean"])
        maxs.append(stats["max"])
        positives.append(
            100.0 * stats["positive"] / max(1, stats["numel"])
        )

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.8), 6))

    x = list(range(len(names)))

    ax.plot(x, mins, marker="o", label="Min")
    ax.plot(x, means, marker="o", label="Mean")
    ax.plot(x, maxs, marker="o", label="Max")

    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Activation value")
    ax.set_title(
        "Activation Statistics Across the Network",
        fontsize=17,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(alpha=0.2)

    # Add positive percentage above each stage.
    for i, percentage in enumerate(positives):
        ax.text(
            i,
            maxs[i],
            f"  positive: {percentage:.1f}%",
            fontsize=7.5,
            va="bottom",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Weight matrices
# ---------------------------------------------------------------------------

def _weight_figures(
    model: nn.Module,
) -> list[tuple[str, Any]]:
    figures = []

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue

        weight = getattr(module, "weight", None)

        if not isinstance(weight, torch.Tensor):
            continue

        if weight.ndim != 2:
            # Heatmaps for arbitrary high-dimensional weights quickly become
            # misleading. Report them through the text summary instead.
            continue

        matrix = weight.detach().float().cpu().numpy()

        fig, ax = plt.subplots(
            figsize=(max(8, min(16, matrix.shape[1] / 3)),
                     max(4, min(10, matrix.shape[0] / 3)))
        )

        image = ax.imshow(matrix, aspect="auto")

        ax.set_title(
            f"{name or module.__class__.__name__} — Weight Matrix "
            f"{tuple(weight.shape)}",
            fontsize=15,
            fontweight="bold",
        )
        ax.set_xlabel("Input dimension")
        ax.set_ylabel("Output neuron")

        fig.colorbar(image, ax=ax, label="Weight value")
        fig.tight_layout()

        figures.append((name or module.__class__.__name__, fig))

    return figures


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------

def _relu_figures(
    traces: Sequence[TraceEntry],
) -> list[tuple[str, Any]]:
    figures = []

    for entry in traces:
        if not isinstance(entry.module, nn.ReLU):
            continue

        if entry.input is None or entry.output is None:
            continue

        before = _flatten_sample(entry.input).numpy()
        after = _flatten_sample(entry.output).numpy()

        fig, ax = plt.subplots(figsize=(14, 5.5))

        indices = range(len(before))

        ax.plot(
            indices,
            before,
            marker="o",
            label="Before ReLU",
        )
        ax.plot(
            indices,
            after,
            marker="o",
            label="After ReLU",
        )
        ax.axhline(0, linewidth=0.9)

        zeroed = int((before < 0).sum())

        ax.set_title(
            f"{entry.name or 'ReLU'} — Before vs After ReLU",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Neuron / feature")
        ax.set_ylabel("Activation")
        ax.legend()
        ax.grid(alpha=0.2)

        ax.text(
            0.99,
            0.97,
            f"Negative values zeroed: {zeroed} / {len(before)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", alpha=0.08),
        )

        fig.tight_layout()
        figures.append((entry.name or "ReLU", fig))

    return figures


# ---------------------------------------------------------------------------
# Individual neuron calculation
# ---------------------------------------------------------------------------

def _neuron_detail_figure(
    model: nn.Module,
    traces: Sequence[TraceEntry],
    *,
    layer: int | str = 0,
    neuron: int = 0,
    max_terms: int = 12,
):
    """
    Explain one Linear neuron using its actual inputs, weights and bias.

    `layer` can be a zero-based index among Linear layers or a module name.
    """
    linear_entries = [
        entry for entry in traces
        if isinstance(entry.module, nn.Linear)
        and entry.input is not None
        and entry.output is not None
    ]

    if not linear_entries:
        raise ValueError("The model contains no traced nn.Linear layer.")

    if isinstance(layer, str):
        candidates = [
            entry for entry in linear_entries
            if entry.name == layer
        ]
        if not candidates:
            available = [entry.name for entry in linear_entries]
            raise ValueError(
                f"Linear layer '{layer}' was not found. "
                f"Available: {available}"
            )
        entry = candidates[0]
    else:
        if layer < 0 or layer >= len(linear_entries):
            raise IndexError(
                f"Linear layer index {layer} is out of range "
                f"(0..{len(linear_entries)-1})."
            )
        entry = linear_entries[layer]

    linear = entry.module

    if neuron < 0 or neuron >= linear.out_features:
        raise IndexError(
            f"Neuron {neuron} is out of range "
            f"(0..{linear.out_features-1})."
        )

    x = _flatten_sample(entry.input).float()
    w = linear.weight.detach().cpu()[neuron].float()
    bias = (
        float(linear.bias.detach().cpu()[neuron])
        if linear.bias is not None
        else 0.0
    )

    contributions = x * w
    pre_activation = float(contributions.sum() + bias)
    actual_output = float(
        _flatten_sample(entry.output)[neuron]
    )

    n = len(x)
    shown = min(max_terms, n)

    if n <= max_terms:
        indices = list(range(n))
    else:
        half = max_terms // 2
        indices = list(range(half)) + list(range(n - half, n))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [1, 1.5]},
    )

    ax = axes[0]
    ax.bar(range(len(indices)), [float(contributions[i]) for i in indices])
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([str(i) for i in indices])
    ax.set_xlabel("Input feature index")
    ax.set_ylabel("xᵢ × wᵢ")
    ax.set_title(
        f"Contributions to {entry.name or 'Linear'} neuron {neuron}",
        fontweight="bold",
    )

    formula = (
        f"z[{neuron}] = Σ(xᵢ × wᵢ) + b\n"
        f"bias = {bias:.6f}\n"
        f"pre-activation z = {pre_activation:.6f}\n"
        f"module output = {actual_output:.6f}"
    )

    axes[1].axis("off")
    axes[1].text(
        0.02,
        0.95,
        formula,
        transform=axes[1].transAxes,
        va="top",
        fontsize=13,
        family="monospace",
    )

    terms = []
    for i in indices:
        terms.append(
            f"x[{i}]={float(x[i]):+.4f} × "
            f"w[{i}]={float(w[i]):+.4f}"
            f" = {float(contributions[i]):+.4f}"
        )

    if n > max_terms:
        terms.insert(shown // 2, "                 ...")

    axes[1].text(
        0.02,
        0.65,
        "\n".join(terms),
        transform=axes[1].transAxes,
        va="top",
        fontsize=9.5,
        family="monospace",
    )

    fig.suptitle(
        "Inside One Neuron",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def _animation(
    model: nn.Module,
    X: torch.Tensor,
    traces: Sequence[TraceEntry],
    *,
    interval: int = 900,
    max_values: int = 40,
):
    """
    Animate one sample moving through the traced modules.

    The returned FuncAnimation can be embedded directly in Jupyter using
    HTML(anim.to_jshtml()).

    No ffmpeg/Graphviz is required for notebook playback.
    """
    stages = [("Input", X)]

    for entry in traces:
        if entry.output is not None:
            stages.append((
                entry.name or entry.module.__class__.__name__,
                entry.output,
            ))

    prepared = []

    for name, tensor in stages:
        values = _flatten_sample(tensor).numpy()

        if len(values) > max_values:
            # Representative beginning/end values.
            half = max_values // 2
            values = list(values[:half]) + list(values[-half:])

        prepared.append((name, values))

    max_len = max(len(values) for _, values in prepared)

    fig, ax = plt.subplots(figsize=(14, 7))

    bars = ax.bar(range(max_len), [0.0] * max_len)
    ax.axhline(0, linewidth=0.9)

    ax.set_xlim(-1, max_len)
    ax.set_ylim(
        min(float(min(values)) for _, values in prepared) * 1.15 - 0.01,
        max(float(max(values)) for _, values in prepared) * 1.15 + 0.01,
    )
    ax.set_xlabel("Feature / neuron")
    ax.set_ylabel("Value")

    title = ax.set_title("", fontsize=18, fontweight="bold")

    def update(frame: int):
        name, values = prepared[frame]

        for index, bar in enumerate(bars):
            if index < len(values):
                bar.set_height(float(values[index]))
                bar.set_visible(True)
            else:
                bar.set_height(0)
                bar.set_visible(False)

        title.set_text(
            f"Data Flow — Step {frame + 1}/{len(prepared)}: {name}"
        )

        return (*bars, title)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(prepared),
        interval=interval,
        blit=False,
        repeat=True,
    )

    return fig, animation


def _save_animation_html(animation, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        animation.to_jshtml(),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Text / machine-readable summary
# ---------------------------------------------------------------------------

def network_summary(
    model: nn.Module,
    X: torch.Tensor | None = None,
) -> dict[str, Any]:
    """
    Return a JSON-serializable model summary.

    This is useful independently of visualization and makes the module useful
    in notebooks, tests, documentation and experiments.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")

    result: dict[str, Any] = {
        "model_type": model.__class__.__name__,
        "training": bool(model.training),
        "parameters": {},
        "modules": [],
    }

    trainable, total = _parameter_count(model)

    result["parameters"] = {
        "trainable": trainable,
        "total": total,
    }

    for info in _leaf_modules(model):
        module = info.module

        item: dict[str, Any] = {
            "name": info.name,
            "type": info.module_type,
            "parameters": sum(
                p.numel() for p in module.parameters(recurse=False)
            ),
        }

        if isinstance(module, nn.Linear):
            item.update({
                "in_features": module.in_features,
                "out_features": module.out_features,
                "weight_shape": tuple(module.weight.shape),
                "bias_shape": (
                    tuple(module.bias.shape)
                    if module.bias is not None
                    else None
                ),
            })

        result["modules"].append(item)

    if X is not None:
        result["input"] = {
            "shape": tuple(X.shape),
            "dtype": str(X.dtype),
            "device": str(X.device),
        }

    return result


def print_network_summary(
    model: nn.Module,
    X: torch.Tensor | None = None,
) -> None:
    """Print a compact human-readable summary."""
    summary = network_summary(model, X)

    print("=" * 78)
    print("NEURAL NETWORK SUMMARY")
    print("=" * 78)
    print(f"Model:              {summary['model_type']}")
    print(f"Trainable params:   {summary['parameters']['trainable']:,}")
    print(f"Total params:       {summary['parameters']['total']:,}")

    if "input" in summary:
        print(f"Input shape:        {summary['input']['shape']}")
        print(f"Input dtype:        {summary['input']['dtype']}")
        print(f"Input device:       {summary['input']['device']}")

    print("\nMODULES")
    print("-" * 78)

    for index, module in enumerate(summary["modules"]):
        print(
            f"{index:>2}. "
            f"{module['name'] or '<unnamed>'}: "
            f"{module['type']} "
            f"({module['parameters']:,} params)"
        )

        if "in_features" in module:
            print(
                f"     {module['in_features']} → "
                f"{module['out_features']} | "
                f"weight={module['weight_shape']} | "
                f"bias={module['bias_shape']}"
            )

    print("=" * 78)



# ---------------------------------------------------------------------------
# Teaching / debug explanations
# ---------------------------------------------------------------------------

def _display_markdown(text: str) -> None:
    """Display Markdown in Jupyter, or fall back to plain text elsewhere."""
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except ImportError:
        # Strip the most common Markdown markers for terminal users.
        plain = text.replace("**", "").replace("`", "")
        print(plain)


def _shape_text(value: torch.Tensor | None) -> str:
    if value is None:
        return "unknown"
    return str(tuple(value.shape))


def _module_role(module: nn.Module) -> str:
    if isinstance(module, nn.Linear):
        return "Linear layer"
    if isinstance(module, nn.ReLU):
        return "ReLU activation"
    if isinstance(module, nn.Sigmoid):
        return "Sigmoid activation"
    if isinstance(module, nn.Tanh):
        return "Tanh activation"
    return module.__class__.__name__


def _debug_layer_explanation(
    index: int,
    entry: TraceEntry,
    before: torch.Tensor,
    after: torch.Tensor | None,
    *,
    max_values: int = 12,
) -> None:
    """Explain one actual forward-pass step using the captured tensors."""
    module = entry.module
    role = _module_role(module)
    name = entry.name or role
    before_shape = _shape_text(before)
    after_shape = _shape_text(after)

    _display_markdown(
        f"""
### Step {index}: `{name}` — {role}

The **output of the previous step becomes the input to this step**.

- Input tensor shape: `{before_shape}`
- Output tensor shape: `{after_shape}`
"""
    )

    if isinstance(module, nn.Linear):
        _display_markdown(
            f"""
**What this layer is doing**

This is a fully connected layer with **{module.in_features} input features**
and **{module.out_features} neurons**.

Each neuron receives **all {module.in_features} input values** and computes:

\\[
z_j = \\sum_i x_i w_{{j,i}} + b_j
\\]

In matrix form, PyTorch performs:

\\[
z = xW^T + b
\\]

So the transformation is:

`{module.in_features} values → {module.out_features} values`

The values below are the **actual values produced by this forward pass**.
"""
        )

        sample_in = before.detach().reshape(-1)
        sample_out = after.detach().reshape(-1) if after is not None else None

        if sample_in.numel() > 0 and module.out_features > 0:
            neuron_idx = min(0, module.out_features - 1)
            weights = module.weight.detach()[neuron_idx]
            bias = (
                float(module.bias.detach()[neuron_idx])
                if module.bias is not None
                else 0.0
            )
            products = sample_in * weights

            limit = min(max_values, sample_in.numel())
            terms = []
            for i in range(limit):
                terms.append(
                    f"`x[{i}] × w[{i}]` = "
                    f"`{_format_number(float(sample_in[i]))} × "
                    f"{_format_number(float(weights[i]))}` = "
                    f"`{_format_number(float(products[i]))}`"
                )
            if sample_in.numel() > limit:
                terms.append(f"*… {sample_in.numel() - limit} more terms …*")

            weighted_sum = float(products.sum())
            preactivation = weighted_sum + bias
            actual_output = (
                float(sample_out[neuron_idx])
                if sample_out is not None and sample_out.numel() > neuron_idx
                else None
            )

            _display_markdown(
                f"""
#### Inside neuron 0

Let's zoom into **one neuron**. This is the most important idea to understand
what a `Linear` layer actually does.

{"  \n".join(terms)}

Then:

- Sum of weighted inputs = `{_format_number(weighted_sum)}`
- Bias = `{_format_number(bias)}`
- Pre-activation `z` = `{_format_number(preactivation)}`
- Neuron output = `{_format_number(actual_output) if actual_output is not None else "unknown"}`

So one neuron is simply a **weighted sum of the inputs plus a bias**.
The layer repeats this calculation for every neuron.
"""
            )

    elif isinstance(module, nn.ReLU):
        _display_markdown(
            """
**What this layer is doing**

ReLU applies:

\\[
\\operatorname{ReLU}(z) = \\max(0,z)
\\]

It does something very simple:

- negative values become `0`
- positive values stay unchanged

This gives the network a **non-linear transformation**. Without non-linear
activations such as ReLU, stacking Linear layers would still be equivalent to
one larger Linear transformation.
"""
        )

        before_values = before.detach().reshape(-1)
        after_values = after.detach().reshape(-1) if after is not None else None
        if after_values is not None:
            negative = int((before_values < 0).sum().item())
            zero = int((after_values == 0).sum().item())
            positive = int((before_values > 0).sum().item())
            _display_markdown(
                f"""
**What happened to this actual sample?**

- Values entering ReLU: `{before_values.numel()}`
- Negative values removed: **{negative}**
- Positive values passed through: **{positive}**
- Zeros after ReLU: **{zero}**

So ReLU changed the representation from:

`{before_shape} → {after_shape}`

while replacing negative activations with zero.
"""
            )

    else:
        _display_markdown(
            f"""
This module is `{module.__class__.__name__}`. The visualizer captured its
actual input and output tensors so you can see how the representation changes
during the forward pass.
"""
        )


def _debug_figure_explanation(kind: str, model: nn.Module, X: torch.Tensor,
                               traces: Sequence[TraceEntry]) -> None:
    """Explain the meaning of each visualization before it is displayed."""
    linear_count = sum(isinstance(t.module, nn.Linear) for t in traces)
    relu_count = sum(isinstance(t.module, nn.ReLU) for t in traces)

    explanations = {
        "architecture": f"""
### Visualization: Network architecture

This image is the **map of the neural network**. It answers:

> *What components does the model contain, and how are they connected?*

It is not showing the calculation for one particular neuron. Instead, it
shows the structure discovered directly from the PyTorch model.

For this model, the visualizer found **{len(traces)} executable layers/modules**,
including **{linear_count} Linear layer(s)** and **{relu_count} ReLU activation(s)**.

The important thing to notice is that the size changes as information moves
through the network. A layer's output size becomes the next layer's input size.
""",
        "data_flow": """
### Visualization: How one input sample travels through the network

This image follows **one sample from `X`** through the forward pass.

Each stage represents the actual tensor produced by a module. Therefore, this
is not a hypothetical diagram: it is based on the values captured while
PyTorch executed `model(X)`.

Think of it as a pipeline:

`input → transformation → activation → transformation → activation → output`

The purpose is to make the tensor-shape changes and representation changes
visible.
""",
        "activations": """
### Visualization: Activation statistics across the network

This image summarizes the values produced at each module.

An **activation** is simply the value produced by a neuron/module during a
forward pass. Statistics such as minimum, maximum, mean and standard
deviation help us understand the numerical range of those values.

For ReLU layers, the percentage of zero values is especially useful: it tells
us how many neurons were inactive for this particular input.

This graph is therefore answering:

> *What kind of numbers are flowing through the network?*
""",
        "weights": """
### Visualization: Learned weights

A Linear layer contains a weight matrix. Each row corresponds to one output
neuron, and each column corresponds to one input feature.

The image shows those learned parameters rather than the values flowing
through the network.

Remember the distinction:

- **weights** = parameters learned by the model
- **activations** = values produced when a particular input is processed
""",
        "relu": """
### Visualization: ReLU behavior

This image focuses specifically on the ReLU transformation.

It lets you compare values **before** and **after** ReLU so you can see which
values were passed through and which negative values became zero.
""",
        "neuron": """
### Visualization: Inside one neuron

This image zooms into a single neuron of a Linear layer.

The neuron calculates:

`weighted inputs → sum → + bias → pre-activation`

This is the concrete numerical version of:

`z = xWᵀ + b`

It is the best visualization to use when connecting the mathematics in the
book with what PyTorch is actually executing.
""",
        "animation": """
### Animation: Watch the representation move

The animation shows the same forward pass one stage at a time.

It is useful for developing an intuition that the neural network is not
magically producing an answer in one operation. The representation is
repeatedly transformed:

`input → Linear → ReLU → Linear → ReLU → output`

The animation is saved as an HTML file when `save=True`, so it can be opened
again later without rerunning the model.
""",
    }

    _display_markdown(explanations.get(kind, ""))


def _debug_walkthrough(
    model: nn.Module,
    X: torch.Tensor,
    traces: Sequence[TraceEntry],
    *,
    max_values: int,
) -> None:
    """Display a guided, step-by-step forward-pass lesson."""
    _display_markdown(
        f"""
# Neural network walkthrough

We are going to follow **one sample** through `{model.__class__.__name__}`.

Your supplied input has shape **`{tuple(X.shape)}`**.

The key idea is:

> A neural network repeatedly takes a tensor, transforms it, and passes the
> resulting tensor to the next layer.

We will inspect each actual transformation performed during this forward pass.
"""
    )

    sample = X[0] if X.ndim > 1 else X
    flat = sample.detach().reshape(-1)
    preview = ", ".join(_format_number(float(v)) for v in flat[:max_values])
    if flat.numel() > max_values:
        preview += f", … ({flat.numel() - max_values} more values)"

    _display_markdown(
        f"""
## Step 0: Input

We start with **one sample** containing `{flat.numel()}` value(s).

Example values from this sample:

`[{preview}]`

This tensor is the starting representation. Nothing has been learned or
changed yet — it is simply the data being given to the model.
"""
    )

    previous = X
    for index, entry in enumerate(traces, start=1):
        _debug_layer_explanation(
            index,
            entry,
            previous,
            entry.output,
            max_values=max_values,
        )
        previous = entry.output if entry.output is not None else previous

    final = traces[-1].output if traces and traces[-1].output is not None else None
    if final is not None:
        values = final.detach().reshape(-1)
        preview = ", ".join(_format_number(float(v)) for v in values[:max_values])
        if values.numel() > max_values:
            preview += f", … ({values.numel() - max_values} more values)"

        _display_markdown(
            f"""
## Final output

The final layer produced:

`[{preview}]`

Shape: **`{tuple(final.shape)}`**

This is the end of the forward pass. What these numbers *mean* depends on
the task and the training setup. For a classification model, for example,
they might later be interpreted as scores or logits.

### The complete journey

`{tuple(X.shape)}` → {" → ".join(_shape_text(t.output) for t in traces if t.output is not None)}

The most important mental model is:

**output of one module = input to the next module.**
"""
        )


def _animation_explanation_only() -> None:
    _display_markdown(
        """
### Animation controls

The animation is an interactive visual version of the forward-pass journey.
It is deliberately separate from the static figures because animation can be
larger and slower to generate.

Use:

```python
visualize_nn(model, X, animation=True)
```

or:

```python
visualize_nn(model, X, show="animation", save=True)
```

When `save=True`, the HTML animation is written below the configured asset
directory.
"""
    )


# ---------------------------------------------------------------------------
# Main facade
# ---------------------------------------------------------------------------

def visualize_nn(
    model: nn.Module,
    X: torch.Tensor,
    *,
    show: str | Sequence[str] = "default",
    debug: bool = False,
    architecture: bool | None = None,
    data_flow: bool | None = None,
    activations: bool | None = None,
    weights: bool | None = None,
    relu: bool | None = None,
    neuron: bool | None = None,
    animation: bool | None = None,
    layer: int | str = 0,
    neuron_index: int = 0,
    max_neurons: int = DEFAULT_MAX_NEURONS,
    max_values: int = 80,
    animation_interval: int = 900,
    save: bool = False,
    asset_dir: str | Path = DEFAULT_ASSET_DIR,
    prefix: str | None = None,
    print_summary: bool = True,
) -> dict[str, Any]:
    """
    Main teaching/debug facade for PyTorch neural-network visualization.

    Normal mode:
        visualize_nn(model, X)

    Teaching mode:
        visualize_nn(model, X, debug=True)

    Teaching mode + animation + saved assets:
        visualize_nn(model, X, debug=True, animation=True, save=True)

    Parameters
    ----------
    debug:
        If True, display a guided explanation before every visualization and
        walk through the actual forward pass module by module. This is intended
        as a learning mode, not merely a debugging flag.

    show:
        "default" -> architecture + data_flow + activations.
        "all" -> all applicable static visualizations + animation.
        A single visualization name or a sequence of names.

    save:
        If True, save figures and animation HTML under asset_dir.

    asset_dir:
        Asset directory. Relative paths are resolved from the notebook's
        current working directory.

    Returns
    -------
    dict
        Contains summary, final_output, traces, figures, animations, and
        generated asset paths.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")

    X = _ensure_input_tensor(X)

    if max_neurons < 1:
        raise ValueError("max_neurons must be >= 1.")
    if max_values < 1:
        raise ValueError("max_values must be >= 1.")
    if animation_interval < 50:
        raise ValueError("animation_interval must be >= 50 ms.")

    if isinstance(show, str):
        if show == "default":
            requested = {"architecture", "data_flow", "activations"}
        elif show == "all":
            requested = {
                "architecture", "data_flow", "activations",
                "weights", "relu", "neuron", "animation",
            }
        else:
            requested = {show}
    else:
        requested = set(show)

    aliases = {
        "architecture": architecture,
        "data_flow": data_flow,
        "activations": activations,
        "weights": weights,
        "relu": relu,
        "neuron": neuron,
        "animation": animation,
    }

    for name, flag in aliases.items():
        if flag is True:
            requested.add(name)
        elif flag is False:
            requested.discard(name)

    valid = set(aliases)
    unknown = requested - valid
    if unknown:
        raise ValueError(
            f"Unknown visualization(s): {sorted(unknown)}. "
            f"Valid options: {sorted(valid)}"
        )

    # Capture the actual forward pass once. Hooks are removed even if the
    # model raises an exception.
    final_output, traces = _capture_forward(model, X)

    result: dict[str, Any] = {
        "summary": network_summary(model, X),
        "final_output": final_output,
        "traces": traces,
        "figures": {},
        "animations": {},
        "assets": [],
        "asset_dir": str(Path(asset_dir).resolve()),
        "debug": debug,
    }

    if print_summary:
        print_network_summary(model, X)

    prefix = prefix or model.__class__.__name__.lower()

    # In teaching mode, explain the complete forward pass first.
    if debug:
        _debug_walkthrough(
            model,
            X,
            traces,
            max_values=min(max_values, 20),
        )

    def save_if_requested(fig: Any, filename: str) -> None:
        if save:
            path = _asset_path(asset_dir, f"{prefix}_{filename}", "png")
            result["assets"].append(str(_save_figure(fig, path)))

    # Architecture
    if "architecture" in requested:
        if debug:
            _debug_figure_explanation("architecture", model, X, traces)
        fig = _architecture_figure(
            model, X, traces, max_neurons=max_neurons
        )
        result["figures"]["architecture"] = fig
        save_if_requested(fig, "architecture")
        plt.show()

    # Data flow
    if "data_flow" in requested:
        if debug:
            _debug_figure_explanation("data_flow", model, X, traces)
        fig = _data_flow_figure(
            model, X, traces, max_values=max_values
        )
        result["figures"]["data_flow"] = fig
        save_if_requested(fig, "data_flow")
        plt.show()

    # Activation statistics
    if "activations" in requested:
        if debug:
            _debug_figure_explanation("activations", model, X, traces)
        fig = _activations_figure(traces)
        if fig is not None:
            result["figures"]["activations"] = fig
            save_if_requested(fig, "activations")
            plt.show()

    # Weight matrices
    if "weights" in requested:
        if debug:
            _debug_figure_explanation("weights", model, X, traces)
        figures = _weight_figures(model)
        result["figures"]["weights"] = dict(figures)
        for index, (name, fig) in enumerate(figures):
            save_if_requested(fig, f"weights_{index}_{name}")
            plt.show()

    # ReLU visualizations
    if "relu" in requested:
        if debug:
            _debug_figure_explanation("relu", model, X, traces)
        figures = _relu_figures(traces)
        result["figures"]["relu"] = dict(figures)
        for index, (name, fig) in enumerate(figures):
            save_if_requested(fig, f"relu_{index}_{name}")
            plt.show()

    # One-neuron calculation
    if "neuron" in requested:
        if debug:
            _debug_figure_explanation("neuron", model, X, traces)
        fig = _neuron_detail_figure(
            model, traces, layer=layer, neuron=neuron_index
        )
        result["figures"]["neuron"] = fig
        save_if_requested(
            fig, f"neuron_layer_{layer}_neuron_{neuron_index}"
        )
        plt.show()

    # Animation
    if "animation" in requested:
        if debug:
            _debug_figure_explanation("animation", model, X, traces)
        fig, anim = _animation(
            model,
            X,
            traces,
            interval=animation_interval,
            max_values=min(max_values, 60),
        )
        result["animations"]["data_flow"] = anim

        if save:
            path = _asset_path(
                asset_dir,
                f"{prefix}_data_flow_animation",
                "html",
            )
            result["assets"].append(str(_save_animation_html(anim, path)))

        try:
            from IPython.display import HTML, display
            display(HTML(anim.to_jshtml()))
        except ImportError:
            pass

        plt.close(fig)

    if debug and not save:
        _display_markdown(
            """
### Assets

No files were written because `save=False`.

To save every generated figure and the animation (when requested), use:

```python
visualize_nn(model, X, debug=True, animation=True, save=True)
```
"""
        )

    if save:
        _display_markdown(
            f"""
### Assets

Generated assets are saved under:

`{Path(asset_dir).resolve()}`

The returned dictionary also contains the exact generated file paths in:

`result["assets"]`
"""
        )

    return result


__all__ = [
    "visualize_nn",
    "network_summary",
    "print_network_summary",
    "DEFAULT_ASSET_DIR",
]
