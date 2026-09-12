"""
nn_visualization_old.py

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
# Main facade
# ---------------------------------------------------------------------------

def visualize_nn(
    model: nn.Module,
    X: torch.Tensor,
    *,
    show: str | Sequence[str] = "default",
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
    Facade for all visualizations.

    Parameters
    ----------
    model:
        Any torch.nn.Module.

    X:
        Example input tensor. For a batch, the first sample is visualized.

    show:
        "default" -> architecture + data flow + activation statistics.
        "all" -> every visualization that applies to the model.
        A string such as "architecture".
        A sequence such as ["architecture", "weights"].

        Supported names:
            architecture
            data_flow
            activations
            weights
            relu
            neuron
            animation

    The boolean flags are optional convenience aliases. If supplied they are
    merged with `show`. For example:

        visualize_nn(model, X, show="default", weights=True)

    save:
        Save generated figures/assets under `asset_dir`.

    asset_dir:
        Directory relative to the notebook's current working directory unless
        an absolute path is supplied.

    Returns
    -------
    dict
        {
            "summary": ...,
            "final_output": tensor,
            "traces": [...],
            "figures": {...},
            "animations": {...},
            "assets": [...]
        }
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

    # Resolve requested visualizations.
    if isinstance(show, str):
        if show == "default":
            requested = {
                "architecture",
                "data_flow",
                "activations",
            }
        elif show == "all":
            requested = {
                "architecture",
                "data_flow",
                "activations",
                "weights",
                "relu",
                "neuron",
                "animation",
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

    final_output, traces = _capture_forward(model, X)

    result: dict[str, Any] = {
        "summary": network_summary(model, X),
        "final_output": final_output,
        "traces": traces,
        "figures": {},
        "animations": {},
        "assets": [],
    }

    if print_summary:
        print_network_summary(model, X)

    prefix = prefix or model.__class__.__name__.lower()

    # Architecture
    if "architecture" in requested:
        fig = _architecture_figure(
            model,
            X,
            traces,
            max_neurons=max_neurons,
        )
        result["figures"]["architecture"] = fig

        if save:
            path = _asset_path(
                asset_dir,
                f"{prefix}_architecture",
                "png",
            )
            result["assets"].append(str(_save_figure(fig, path)))

        plt.show()

    # Data flow
    if "data_flow" in requested:
        fig = _data_flow_figure(
            model,
            X,
            traces,
            max_values=max_values,
        )
        result["figures"]["data_flow"] = fig

        if save:
            path = _asset_path(
                asset_dir,
                f"{prefix}_data_flow",
                "png",
            )
            result["assets"].append(str(_save_figure(fig, path)))

        plt.show()

    # Activation statistics
    if "activations" in requested:
        fig = _activations_figure(traces)

        if fig is not None:
            result["figures"]["activations"] = fig

            if save:
                path = _asset_path(
                    asset_dir,
                    f"{prefix}_activations",
                    "png",
                )
                result["assets"].append(str(_save_figure(fig, path)))

            plt.show()

    # Weight matrices
    if "weights" in requested:
        figures = _weight_figures(model)
        result["figures"]["weights"] = {
            name: fig for name, fig in figures
        }

        for index, (name, fig) in enumerate(figures):
            if save:
                path = _asset_path(
                    asset_dir,
                    f"{prefix}_weights_{index}_{name}",
                    "png",
                )
                result["assets"].append(str(_save_figure(fig, path)))

            plt.show()

    # ReLU
    if "relu" in requested:
        figures = _relu_figures(traces)
        result["figures"]["relu"] = {
            name: fig for name, fig in figures
        }

        for index, (name, fig) in enumerate(figures):
            if save:
                path = _asset_path(
                    asset_dir,
                    f"{prefix}_relu_{index}_{name}",
                    "png",
                )
                result["assets"].append(str(_save_figure(fig, path)))

            plt.show()

    # One-neuron calculation
    if "neuron" in requested:
        fig = _neuron_detail_figure(
            model,
            traces,
            layer=layer,
            neuron=neuron_index,
        )
        result["figures"]["neuron"] = fig

        if save:
            path = _asset_path(
                asset_dir,
                f"{prefix}_neuron_layer_{layer}_neuron_{neuron_index}",
                "png",
            )
            result["assets"].append(str(_save_figure(fig, path)))

        plt.show()

    # Animation
    if "animation" in requested:
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

        # In Jupyter, IPython's display system can render the HTML animation.
        try:
            from IPython.display import HTML, display
            display(HTML(anim.to_jshtml()))
        except ImportError:
            # Outside Jupyter, keep the animation object available to caller.
            pass

        plt.close(fig)

    return result


__all__ = [
    "visualize_nn",
    "network_summary",
    "print_network_summary",
    "DEFAULT_ASSET_DIR",
]
