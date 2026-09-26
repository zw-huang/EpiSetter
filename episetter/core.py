"""Small, testable linear-algebra primitives used by the GPU experiments."""

from __future__ import annotations

from typing import Any


def orthogonalize(vector: Any, protected_basis: Any, *, eps: float = 1e-8) -> Any:
    """Project ``vector`` out of an orthonormal row basis and return unit norm."""
    if protected_basis is not None and protected_basis.numel():
        vector = vector - protected_basis.T @ (protected_basis @ vector)
    norm = vector.norm()
    if not bool(norm.detach().isfinite()) or float(norm.detach()) <= eps:
        raise ValueError("Direction vanished in the protected nullspace")
    return vector / norm.to(vector.dtype)


def replace_coordinate(hidden: Any, direction: Any, source_hidden: Any) -> Any:
    """Keep hidden's orthogonal complement and copy source's scalar coordinate."""
    return set_coordinate(hidden, direction, source_hidden.float() @ direction.float())


def set_coordinate(hidden: Any, direction: Any, value: Any) -> Any:
    """Replace a unit direction coordinate with an explicitly supplied value."""
    current = hidden.float() @ direction.float()
    delta = value.to(current.device, current.dtype) - current
    return (hidden.float() + delta.unsqueeze(-1) * direction.float()).to(hidden.dtype)


def protected_basis_from_gradients(gradients: Any, rank: int | None, torch: Any) -> Any:
    """Top right-singular vectors of normalized functional gradients."""
    if gradients.ndim != 2:
        raise ValueError("gradients must have shape [examples, hidden_size]")
    gradients = gradients.float()
    gradients = gradients / gradients.norm(dim=1, keepdim=True).clamp_min(1e-8)
    _, singular_values, vh = torch.linalg.svd(gradients, full_matrices=False)
    effective = int((singular_values > singular_values.max().clamp_min(1e-8) * 1e-5).sum().item())
    if rank is not None:
        effective = min(rank, effective)
    return vh[:effective]


def first_token_id(tokenizer: Any, answer: str, torch: Any) -> int:
    ids = tokenizer(" " + answer.strip(), add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ][0]
    if not len(ids):
        raise ValueError("empty answer tokenization")
    return int(ids[0])
