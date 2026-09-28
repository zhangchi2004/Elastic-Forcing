"""Use real wandb when installed; allow explicitly disabled training without it."""

try:
    import wandb as wandb
except ModuleNotFoundError as exc:
    if exc.name != "wandb":
        raise

    class _DisabledWandb:
        @staticmethod
        def finish(*args, **kwargs):
            return None

        def __getattr__(self, name):
            raise RuntimeError(
                "wandb is not installed; launch with --disable-wandb"
            )

    wandb = _DisabledWandb()
