"""Opt-in DINOv3 ablation without changing the existing MMD contract."""

ALL_BRANCHES = ("dinov3", "vjepa2", "videomae_v2")
NO_DINO_BRANCHES = ("vjepa2", "videomae_v2")


def active_mmd_branches(value=None):
    """Keep legacy order and reject accidental branch/weight experiments."""
    branches = ALL_BRANCHES if value is None else tuple(value)
    if branches not in (ALL_BRANCHES, NO_DINO_BRANCHES):
        raise ValueError(
            "triple_active_branches must be ordered as "
            f"{ALL_BRANCHES!r} or {NO_DINO_BRANCHES!r}; got {branches!r}")
    return branches
