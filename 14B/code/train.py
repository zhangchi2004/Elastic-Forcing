import argparse
import os
import torch.distributed as dist
from omegaconf import OmegaConf
from utils.config import load_config
from utils.optional_wandb import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument(
        "--max_steps", "--training-updates",
        type=int,
        default=None,
        help="Optional runtime override for training_updates (useful for smoke tests)",
    )
    parser.add_argument(
        "--fresh-video-local-samples", "--distribution-samples-per-rank",
        type=int,
        default=None,
        help=(
            "Optional smoke-test override for the number of fresh videos "
            "collected per rank and optimizer update."
        ),
    )
    parser.add_argument(
        "--total-batch-size", "--distribution-batch-size",
        type=int,
        default=None,
        help="Optional matching logical global-batch override.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Optional runtime eval interval override; 0 disables eval.",
    )
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Build and validate the complete distributed trainer, then exit before updates.",
    )

    args = parser.parse_args()

    config = load_config(args.config_path, "configs/default_config.yaml")
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    if args.max_steps is not None:
        if args.max_steps <= 0:
            parser.error("--max_steps must be positive")
        config.max_steps = args.max_steps
    if args.fresh_video_local_samples is not None:
        if args.fresh_video_local_samples <= 0:
            parser.error("--fresh-video-local-samples must be positive")
        config.fresh_video_local_samples = args.fresh_video_local_samples
    if args.total_batch_size is not None:
        if args.total_batch_size <= 0:
            parser.error("--total-batch-size must be positive")
        config.total_batch_size = args.total_batch_size
    if args.eval_interval is not None:
        if args.eval_interval < 0:
            parser.error("--eval-interval must be non-negative")
        config.eval_interval = args.eval_interval

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    # Preserve the exact YAML identity for strict offline-reference checks.
    # Runtime-only smoke overrides remain separate and do not change the
    # generator trajectory contract encoded by this file.
    config.config_path = os.path.realpath(args.config_path)
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    if config.trainer != "score_distillation":
        raise ValueError("This release uses the Elastic Forcing trainer.")
    from trainer.elastic_forcing import ElasticForcingTrainer
    trainer = ElasticForcingTrainer(config)
    try:
        if args.preflight_only:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            if not dist.is_initialized() or dist.get_rank() == 0:
                print("Distributed trainer preflight complete; no update or checkpoint written.")
        else:
            trainer.train()
    finally:
        wandb.finish()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
