"""The two index spaces for Parcae, in one place.

    KV plane space   (strategy- and T_max-dependent)
        full         : prelude i -> i
                       core (r,l) -> L_p + r*L_c + l          [r = recurrence depth]
                       coda i     -> L_p + T_max*L_c + i      [pinned at T_max, not the request's T]
                       count      = L_p + T_max*L_c + L_d
        compress-s1  : prelude i -> i
                       core (r,l) -> L_p + l                  [all depths collapse, last-write-wins]
                       coda i     -> L_p + L_c + i            [compacted]
                       count      = L_p + L_c + L_d

    ve / weight space  (neither strategy- nor depth-dependent)
                       prelude i -> i
                       core l    -> L_p + l                   [same table at every depth]
                       coda i    -> L_p + L_c + i
                       count     = L_p + L_c + L_d
"""

from __future__ import annotations

STRATEGIES = ("full", "compress-s1")


def n_core_planes(strategy: str, t_max: int) -> int:
    if strategy == "full":
        return t_max
    if strategy == "compress-s1":
        return 1
    raise ValueError(f"unknown lookup strategy: {strategy!r}")


def plane_count(strategy: str, t_max: int, n_prelude: int, n_core: int, n_coda: int) -> int:
    """Number of KV planes to declare to SGLang as `num_hidden_layers`.

        If using full (per-depth KV), then the plane count is L_p + T_max*L_c + L_d.
    """
    return n_prelude + (n_core_planes(strategy, t_max) * n_core) + n_coda


def plane(
    kind: str,
    layer: int,
    *,
    depth: int = 0,
    strategy: str,
    t_max: int,
    n_prelude: int,
    n_core: int,
) -> int:
    """KV cache plane (== SGLang `RadixAttention.layer_id`) for one block invocation."""
    if kind == "prelude":
        if not 0 <= layer < n_prelude:
            raise ValueError(f"prelude layer {layer} out of range")
        return layer
    elif kind == "core":
        if not 0 <= layer < n_core:
            raise ValueError(f"core layer {layer} out of range")
        if not 0 <= depth < t_max:
            raise ValueError(f"recurrence depth {depth} out of range for t_max={t_max}")
        effective_depth = depth if strategy == "full" else 0
        return n_prelude + (effective_depth * n_core) + layer
    elif kind == "coda":
        return n_prelude + n_core_planes(strategy, t_max) * n_core + layer
    raise ValueError(f"unknown block kind: {kind!r}")


def ve_key(kind: str, layer: int, *, n_prelude: int, n_core: int) -> str:
    """Key into `value_embeds` (a ModuleDict keyed by str).

    Built over `n_effective_layers = L_p + L_c + L_d` -- not the plane count.
    """
    if kind == "prelude":
        index = layer
    elif kind == "core":
        index = n_prelude + layer
    elif kind == "coda":
        index = n_prelude + n_core + layer
    else:
        raise ValueError(f"unknown block kind: {kind!r}")
    return str(index)


def has_ve(index: int, n_effective_layers: int) -> bool:
    """Mirror of parcae_lm.modules.mixer.has_ve: alternating layers, last always included."""
    return index % 2 == (n_effective_layers - 1) % 2


def _plane_assignments(
    strategy: str, t_max: int, n_prelude: int, n_core: int, n_coda: int
) -> list[tuple[str, int]]:
    """Every distinct plane the model will write, as (label, plane) pairs."""
    addr = dict(strategy=strategy, t_max=t_max, n_prelude=n_prelude, n_core=n_core)
    out: list[tuple[str, int]] = []
    for i in range(n_prelude):
        out.append((f"prelude[{i}]", plane("prelude", i, **addr)))
    for depth in range(n_core_planes(strategy, t_max)):
        for layer in range(n_core):
            out.append((f"core[d={depth},l={layer}]", plane("core", layer, depth=depth, **addr)))
    for i in range(n_coda):
        out.append((f"coda[{i}]", plane("coda", i, **addr)))
    return out


def _check_unique(assignments: list[tuple[str, int]]) -> dict[int, str]:
    """No two different blocks may share a plane -- that would silently overwrite KV."""
    owner: dict[int, str] = {}
    for label, p in assignments:
        if p in owner:
            raise AssertionError(f"plane {p} is claimed by both {owner[p]} and {label}")
        owner[p] = label
    return owner


def _check_covers_range(owner: dict[int, str], total: int) -> None:
    """Planes must be exactly [0, total)."""
    if sorted(owner) == list(range(total)):
        return
    missing = sorted(set(range(total)) - set(owner))
    unexpected = sorted(set(owner) - set(range(total)))
    raise AssertionError(
        f"planes must cover [0,{total}) exactly; missing={missing[:8]} unexpected={unexpected[:8]}"
    )


def _check_depth_behaviour(strategy: str, t_max: int, n_prelude: int, n_core: int) -> None:
    """The property that actually separates the two strategies."""
    addr = dict(strategy=strategy, t_max=t_max, n_prelude=n_prelude, n_core=n_core)
    for layer in range(n_core):
        by_depth = [plane("core", layer, depth=d, **addr) for d in range(t_max)]
        distinct = len(set(by_depth))
        if strategy == "compress-s1" and distinct != 1:
            raise AssertionError(
                f"compress-s1 must collapse every depth onto one plane, but core layer {layer} "
                f"maps to {distinct} distinct planes: {sorted(set(by_depth))}"
            )
        if strategy == "full" and distinct != t_max:
            raise AssertionError(
                f"full must give every depth its own plane, but core layer {layer} maps to "
                f"{distinct} distinct planes across {t_max} depths: {sorted(set(by_depth))}"
            )


def validate_planes(
    strategy: str, t_max: int, n_prelude: int, n_core: int, n_coda: int
) -> None:
    """Startup guard on the KV plane map. Checks three independent properties:

      1. unique     -- no two blocks share a plane
      2. covering   -- the planes are exactly [0, plane_count), no holes or overshoot
      3. depth      -- compress-s1 collapses all depths onto one plane per core layer; full does not
    """
    assignments = _plane_assignments(strategy, t_max, n_prelude, n_core, n_coda)
    owner = _check_unique(assignments)
    _check_covers_range(owner, plane_count(strategy, t_max, n_prelude, n_core, n_coda))
    _check_depth_behaviour(strategy, t_max, n_prelude, n_core)
