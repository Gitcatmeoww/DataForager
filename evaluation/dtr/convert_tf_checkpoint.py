"""Convert a released TAPAS dual-encoder TF checkpoint to a PyTorch state dict.

The checkpoints published with Herzig et al. (2021) are TF1 estimator
checkpoints. Rather than take on the original TF training stack, this script
uses TensorFlow purely as a file reader and hands the tensors to PyTorch.

That split means two interpreters: TensorFlow no longer ships wheels for the
Python version the project itself runs on. The work is therefore staged, and
the TF import is lazy so that this module imports cleanly without TF present.

    # 1. Under a scratch venv that has tensorflow (Python 3.11):
    python -m evaluation.dtr.convert_tf_checkpoint dump \
        --checkpoint evaluation/dtr/checkpoints/tapas_dual_encoder_proj_256_medium

    # 2. Under the project venv, which has torch:
    python -m evaluation.dtr.convert_tf_checkpoint convert \
        --checkpoint evaluation/dtr/checkpoints/tapas_dual_encoder_proj_256_medium

Variable layout of the TF checkpoint, verified against
tapas/models/table_retriever_model.py:

- bert/...           the TABLE tower, built first (line 319)
- bert_1/...         the QUERY tower, built second (line 349)
- table_projection   [proj_dim, hidden], applied with transpose_b=True
- text_projection    [proj_dim, hidden], the query-side projection
- Adam slots (adam_m, adam_v) and global_step, all discarded

Getting the tower order backwards produces a model that still trains and still
scores, but silently encodes queries with the table encoder. The order above is
not a guess; it follows the construction order in the original model_fn.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Filename of the intermediate dump written by the TF stage and read by the
# torch stage.
NPZ_NAME = "tf_weights.npz"

# TF scope of each tower. The table encoder is created first and so takes the
# unsuffixed 'bert' scope.
TABLE_TOWER_SCOPE = "bert"
QUERY_TOWER_SCOPE = "bert_1"

# TF projection variable -> attribute on TapasDualEncoder.
PROJECTIONS = {
    "table_projection": "table_projection",
    "text_projection": "query_projection",
}

# TF variables that carry optimizer state or bookkeeping rather than weights.
_SKIP_SUFFIXES = ("adam_m", "adam_v")
_SKIP_NAMES = ("global_step",)


def dump_tf_checkpoint(checkpoint_dir: Path) -> Path:
    """Read every non-optimizer tensor out of a TF checkpoint into an .npz.

    Runs under a scratch venv that has TensorFlow. TF is used only to read the
    checkpoint; no graph is built and nothing is executed.

    Args:
        checkpoint_dir: Directory holding model.ckpt.{index,data-*}.

    Returns:
        Path to the written .npz file.
    """
    import tensorflow as tf  # Lazy: only the dump stage needs TF.

    index_files = sorted(checkpoint_dir.glob("*.ckpt*.index"))
    if not index_files:
        raise FileNotFoundError(f"No TF checkpoint index found under {checkpoint_dir}")
    prefix = str(index_files[0])[: -len(".index")]

    reader = tf.train.load_checkpoint(prefix)
    shapes = reader.get_variable_to_shape_map()

    tensors = {}
    for name in sorted(shapes):
        if name in _SKIP_NAMES or name.endswith(_SKIP_SUFFIXES):
            continue
        tensors[name] = reader.get_tensor(name)

    out_path = checkpoint_dir / NPZ_NAME
    np.savez(out_path, **tensors)
    print(f"Dumped {len(tensors)} tensors from {prefix} to {out_path}")
    return out_path


def _tower_key(tf_name: str) -> tuple[str, bool]:
    """Map one TF tower variable name to its HF TapasModel key.

    Args:
        tf_name: Variable name with the tower scope already stripped, for
            example "encoder/layer_3/attention/self/query/kernel".

    Returns:
        A (key, transpose) pair, where transpose says whether the tensor is a
        TF dense kernel stored as [in, out] and so must be transposed to match
        the [out, in] layout of nn.Linear.
    """
    name = tf_name

    # TF names LayerNorm scale/offset gamma/beta; torch calls them weight/bias.
    name = name.replace("LayerNorm/gamma", "LayerNorm/weight")
    name = name.replace("LayerNorm/beta", "LayerNorm/bias")

    # encoder/layer_7/... -> encoder/layer/7/...
    name = re.sub(r"^encoder/layer_(\d+)/", r"encoder/layer/\1/", name)

    transpose = False
    if name.endswith("/kernel"):
        name = name[: -len("/kernel")] + "/weight"
        transpose = True

    # Bare embedding tables become the .weight of an nn.Embedding.
    if re.search(r"embeddings/(word_embeddings|position_embeddings|token_type_embeddings_\d+)$", name):
        name += "/weight"

    return name.replace("/", "."), transpose


def build_state_dict(tensors: dict[str, np.ndarray]) -> dict:
    """Turn the dumped TF tensors into a TapasDualEncoder state dict.

    Args:
        tensors: Mapping of TF variable name to array, as produced by
            dump_tf_checkpoint.

    Returns:
        A state dict keyed to match TapasDualEncoder.

    Raises:
        ValueError: If any tensor is left unconsumed, which would mean the
            mapping has silently dropped weights.
    """
    import torch

    state, consumed = {}, set()

    for tf_name, array in tensors.items():
        if tf_name in PROJECTIONS:
            # Stored as [proj_dim, hidden] and applied with transpose_b=True,
            # which is precisely the [out, in] layout nn.Linear expects. No
            # transpose here, unlike the dense kernels below.
            state[f"{PROJECTIONS[tf_name]}.weight"] = torch.from_numpy(array)
            consumed.add(tf_name)
            continue

        for scope, tower in ((TABLE_TOWER_SCOPE, "table_encoder"), (QUERY_TOWER_SCOPE, "query_encoder")):
            if tf_name.startswith(f"{scope}/"):
                key, transpose = _tower_key(tf_name[len(scope) + 1 :])
                tensor = torch.from_numpy(array)
                state[f"{tower}.{key}"] = tensor.T.contiguous() if transpose else tensor
                consumed.add(tf_name)
                break

    unconsumed = set(tensors) - consumed
    if unconsumed:
        raise ValueError(f"Unmapped TF variables: {sorted(unconsumed)}")

    return state


def load_config(checkpoint_dir: Path):
    """Build a TapasConfig from the checkpoint's bert_config.json."""
    from transformers import TapasConfig

    cfg = json.loads((checkpoint_dir / "bert_config.json").read_text())
    return TapasConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        intermediate_size=cfg["intermediate_size"],
        hidden_act=cfg["hidden_act"],
        hidden_dropout_prob=cfg["hidden_dropout_prob"],
        attention_probs_dropout_prob=cfg["attention_probs_dropout_prob"],
        max_position_embeddings=cfg["max_position_embeddings"],
        type_vocab_sizes=cfg["type_vocab_size"],
        initializer_range=cfg["initializer_range"],
    )


def convert(checkpoint_dir: Path) -> Path:
    """Convert a dumped checkpoint into a torch state dict on disk.

    Loads with strict=True so that any missing or unexpected key is a hard
    failure rather than a silently half-initialized model.

    Args:
        checkpoint_dir: Directory holding bert_config.json and the .npz dump.

    Returns:
        Path to the written .pt file.
    """
    import torch

    from evaluation.dtr.modeling import TapasDualEncoder

    npz_path = checkpoint_dir / NPZ_NAME
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run the 'dump' stage under a venv with TensorFlow first."
        )

    tensors = dict(np.load(npz_path))
    config = load_config(checkpoint_dir)
    proj_dim = tensors["table_projection"].shape[0]

    model = TapasDualEncoder(config, projection_dim=proj_dim)
    state = build_state_dict(tensors)

    expected = set(model.state_dict())
    if set(state) != expected:
        raise ValueError(
            f"Key mismatch. Missing: {sorted(expected - set(state))[:5]} "
            f"Unexpected: {sorted(set(state) - expected)[:5]}"
        )
    model.load_state_dict(state, strict=True)

    out_path = checkpoint_dir / "pytorch_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": config.to_dict(), "projection_dim": proj_dim}, out_path)
    print(f"Converted {len(state)} tensors (projection_dim={proj_dim}) to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=["dump", "convert"])
    parser.add_argument("--checkpoint", type=Path, required=True, help="Unzipped checkpoint directory")
    args = parser.parse_args()

    if args.stage == "dump":
        dump_tf_checkpoint(args.checkpoint)
    else:
        convert(args.checkpoint)


if __name__ == "__main__":
    main()
