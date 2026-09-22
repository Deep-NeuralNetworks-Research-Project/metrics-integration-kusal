"""The single evaluation path. Every paper number comes out of this script.

    python -m cdlib.cli.evaluate --dummy --split test --out results/eval.json
    python -m cdlib.cli.evaluate --checkpoint PATH --split test --out results/eval.json
    python -m cdlib.cli.evaluate --dummy --robustness suite

``--dummy`` scores the stand-in pair CNN so CI can lock the schema.
A real checkpoint is hashed and loaded only if it is that same module.
Team model classes are not in this slice yet; the script refuses to
invent scores for an unknown architecture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from cdlib.cli._dummy import DummyPairCNN
from cdlib.metrics.report import SCHEMA_KEYS, build_eval_report
from cdlib.metrics.tables import nuisance_markdown
from cdlib.utils.reproducibility import set_seed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dummy_batch(n: int = 4, h: int = 32, w: int = 32, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(n, 1, h, w, generator=g) > 0.85).float()
    mask[0].zero_()
    return {
        "img1": torch.rand(n, 3, h, w, generator=g),
        "img2": torch.rand(n, 3, h, w, generator=g),
        "mask": mask,
        "nuisance_label": torch.zeros(n, dtype=torch.int64),
        "meta": {"pair_id": "dummy", "dataset": "synthetic"},
    }


def load_model(checkpoint: Path | None, dummy: bool) -> tuple[torch.nn.Module, str]:
    if dummy or checkpoint is None:
        return DummyPairCNN().eval(), hashlib.sha256(b"dummy").hexdigest()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    sha = sha256_file(checkpoint)
    try:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(checkpoint, map_location="cpu")
    state = blob.get("state_dict", blob) if isinstance(blob, dict) else None
    model = DummyPairCNN()
    if not isinstance(state, dict):
        raise SystemExit(
            "checkpoint has no state_dict. This slice only loads DummyPairCNN weights; "
            "do not score an unknown architecture."
        )
    try:
        model.load_state_dict(state)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(
            "checkpoint does not match DummyPairCNN. Team model classes are not in this "
            f"slice yet, so no number was emitted. ({exc.__class__.__name__})"
        ) from exc
    return model.eval(), sha


def _parse_leftovers(unknown: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in unknown:
        if "=" in tok:
            key, value = tok.split("=", 1)
            out[key.lstrip("+")] = value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config", default="unnamed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--robustness", default="", help="suite to emit the full nuisance grid")
    args, unknown = parser.parse_known_args(argv)
    extra = _parse_leftovers(unknown)
    set_seed(args.seed)

    robustness = args.robustness in {"suite", "full"} or extra.get("robustness") in {"suite", "full"}
    use_dummy = args.dummy or args.checkpoint is None
    model, sha = load_model(args.checkpoint, dummy=use_dummy)
    batch = dummy_batch(args.batch_size, args.img_size, args.img_size, seed=args.seed)
    val = dummy_batch(args.batch_size, args.img_size, args.img_size, seed=args.seed + 1)
    report = build_eval_report(
        model,
        batch,
        split=args.split,
        seed=args.seed,
        checkpoint_sha256=sha,
        config=extra.get("exp_id", args.config),
        val_batch=val,
        robustness=robustness,
    )
    missing = SCHEMA_KEYS - report.keys()
    if missing:
        raise SystemExit(f"schema keys missing: {sorted(missing)}")

    text = json.dumps(report, indent=2)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
        if robustness and isinstance(report["nuisance"], dict):
            side = args.out.with_suffix(".nuisance.md")
            side.write_text(nuisance_markdown(report["nuisance"]) + "\n")
            print(f"wrote {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
