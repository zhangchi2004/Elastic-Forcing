#!/usr/bin/env python3
"""Audit the readability refactor against its public Git baseline.

Compare resolved settings and executable syntax, allowing only the explicit
symbol/module renames below, documentation edits and compatibility aliases.
This does not substitute for a GPU training run.
"""
import argparse
import ast
import difflib
import importlib.util
from pathlib import Path
import re
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "7985874ff49de326b8350a66516c50645a604f45"
# New names are mapped back to their baseline spelling for the syntax comparison.
SYMBOLS = {
    "ElasticForcingModel": "WanKVMMD",
    "ElasticForcingTrainer": "Trainer",
    "MultiEncoderMMD": "MultiRepresentationVideoNystromMMD",
    "normalize_mmd_with_detached_scale": "fd_normalize_raw_mmd",
    "FeatureGradientReplay": "MultiRepresentationFreshVideoGradCache",
    "FeatureGradientReplayResult": "MultiRepresentationFreshVideoGradCacheResult",
    "_collect_rollout_microbatch": "_prepare_fresh_video_microbatch",
    "_run_distribution_matching_step": "_fresh_video_generator_update",
    "collect_rollout_features": "fresh_video_gradcache_forward",
    "prepare_latent_gradients": "fresh_video_gradcache_prepare_replay",
    "replay_generator_gradients": "fresh_video_gradcache_replay",
    "_initialize_multi_encoder_features": "_initialize_triple_dinov3_vjepa2_videomae_v2_features",
    "_build_multi_encoder_objective": "_replace_with_triple_nystrom_objective",
    "_extract_multi_encoder_features": "_extract_dual_video_features",
    "_collect_multi_encoder_rollout_features": "_fresh_video_dual_gradcache_forward",
    "_prepare_multi_encoder_latent_gradients": "_dual_gradcache_prepare_replay",
}
REPLAY_LOCALS = {
    "global_distribution_features": "global_reference",
    "global_differentiation_features": "global_gradient",
    "global_reference_features": "global_teacher",
    "local_distribution_features": "local_reference",
    "local_differentiation_features": "local_gradient",
    "local_reference_features": "local_teacher",
    "distribution_parts": "reference_parts",
    "differentiation_parts": "gradient_parts",
    "reference_parts": "teacher_parts",
}
MODULES = {
    "trainer.feature_gradient_replay": "trainer.multi_representation_fresh_video_gradcache",
    "losses.multi_encoder_mmd": "losses.multi_representation_video_nystrom_mmd",
}
FILES = {
    "model/elastic_forcing.py": "model/wan_kv_mmd.py",
    "trainer/elastic_forcing.py": "trainer/distillation.py",
    "trainer/feature_gradient_replay.py": "trainer/multi_representation_fresh_video_gradcache.py",
    "losses/multi_encoder_mmd.py": "losses/multi_representation_video_nystrom_mmd.py",
    "losses/video_nystrom_mmd.py": "losses/video_nystrom_mmd.py",
    "pipeline/self_forcing_training.py": "pipeline/self_forcing_training.py",
    "video_fd/replay/replay_video_encoder_vjp.py": "video_fd/replay/replay_video_encoder_vjp.py",
}


class DocumentationOnly(ast.NodeTransformer):
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__all__":
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


class RestoreNames(DocumentationOnly):
    def __init__(self, replay=False):
        self.symbols = {**SYMBOLS, **(REPLAY_LOCALS if replay else {})}

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target == "__all__":
                # New exports list readable and legacy aliases; it is not numerical code.
                return None
            if isinstance(node.value, ast.Name) and SYMBOLS.get(node.value.id) == target:
                return None
        return self.generic_visit(node)

    def visit_Name(self, node):
        node.id = self.symbols.get(node.id, node.id)
        return node

    def visit_Attribute(self, node):
        node.attr = SYMBOLS.get(node.attr, node.attr)
        return self.generic_visit(node)

    def visit_arg(self, node):
        node.arg = self.symbols.get(node.arg, node.arg)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.name = SYMBOLS.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        node.name = SYMBOLS.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        node.module = MODULES.get(node.module, node.module)
        for name in node.names:
            name.name = SYMBOLS.get(name.name, name.name)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            # Method lookup strings and diagnostic names were renamed with their symbols.
            pattern = r"\b(" + "|".join(map(re.escape, SYMBOLS)) + r")\b"
            node.value = re.sub(pattern, lambda m: SYMBOLS[m.group()], node.value)
        return node


def baseline_source(revision, path):
    return subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)


def config_helpers(release):
    spec = importlib.util.spec_from_file_location(
        "readability_config", ROOT / release / "code/utils/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    args = parser.parse_args()
    count = 0
    for release in ("1.3B", "14B"):
        for current, original in FILES.items():
            before = ast.parse(baseline_source(args.baseline, f"{release}/code/{original}"))
            after = ast.parse((ROOT / release / "code" / current).read_text())
            before = DocumentationOnly().visit(before)
            after = RestoreNames(replay=current == "trainer/feature_gradient_replay.py").visit(after)
            left, right = (ast.dump(tree, indent=2) for tree in (before, after))
            if left != right:
                diff = "\n".join(list(difflib.unified_diff(left.splitlines(), right.splitlines()))[:80])
                raise AssertionError(f"Executable syntax changed: {release}/{current}\n{diff}")
            count += 1
        helper = config_helpers(release)
        names = ["three_encoder.yaml"] if release == "1.3B" else ["train.yaml", "inference.yaml"]
        for name in names:
            path = f"{release}/configs/{name}"
            original = yaml.safe_load(baseline_source(args.baseline, path))
            resolved = helper.resolve_config_aliases(yaml.safe_load((ROOT / path).read_text()))
            assert resolved == original, f"Resolved settings changed: {path}"
            assert list(resolved) == list(original), f"Resolved key order changed: {path}"
            print(f"PASS {path}: all {len(original)} resolved fields and their order match")
    # Parse every Python source without importing CUDA models or creating bytecode.
    sources = list(ROOT.rglob("*.py"))
    for path in sources:
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
    print(f"PASS {count} core implementation files: executable AST matches after documented renames")
    print(f"PASS {len(sources)} Python files parse")


if __name__ == "__main__":
    main()
