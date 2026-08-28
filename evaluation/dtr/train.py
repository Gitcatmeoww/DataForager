"""Fine-tune DTR on a corpus split with in-batch negatives.

Follows the training setup in Herzig et al. (2021): batch 256, lr 1.25e-5, Adam,
linear decay with a 0.2 warmup ratio, dropout 0.2, early stopping on validation
Recall@10.

The batch size is the number of negatives each query sees, so it cannot be
traded for gradient accumulation. Gradient checkpointing is what buys the paper's
batch of 256 on a single machine, and it is also faster than a smaller batch
without it, since the uncheckpointed run thrashes memory.

    python -m evaluation.dtr.train --checkpoint <converted checkpoint> \
        --csv-dir evaluation/huggingface --output-dir evaluation/dtr/runs/dtr
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from evaluation.dtr.adapters import KaggleDSAdapter
from evaluation.dtr.index import DTRIndex, encode_query_batch
from evaluation.dtr.evaluate_dtr import recall_at_k
from evaluation.dtr.modeling import load_dual_encoder, resolve_device
from evaluation.dtr.serialization import (
    DEFAULT_MAX_QUERY_LENGTH,
    DEFAULT_MAX_TABLE_LENGTH,
    encode_queries,
    encode_tables,
)

DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1.25e-5
DEFAULT_WARMUP_RATIO = 0.2
DEFAULT_DROPOUT = 0.2


def group_aware_batches(examples, table_by_id, batch_size, rng):
    """Split examples into batches that never repeat a group_id.

    Tables from one Kaggle dataset are often near-identical, so two examples
    sharing a group can be equally valid answers to the same query. Letting them
    meet inside a batch would train the model to push them apart. This is the
    analogue of the original's rule that a mined negative must not contain the
    reference answer.

    Args:
        examples: RetrievalExamples to batch.
        table_by_id: Lookup from table_id to TableRecord.
        batch_size: Target examples per batch.
        rng: random.Random controlling the shuffle.

    Returns:
        A list of batches, each a list of RetrievalExamples. Only full batches
        are returned, since in-batch negatives depend on a fixed batch size.
    """
    remaining = list(examples)
    rng.shuffle(remaining)

    batches = []
    while True:
        current, used_groups, deferred = [], set(), []
        for example in remaining:
            group = table_by_id[example.table_id].group_id
            if group in used_groups:
                deferred.append(example)
                continue
            current.append(example)
            used_groups.add(group)
            if len(current) == batch_size:
                batches.append(current)
                current, used_groups = [], set()

        # A trailing partial batch is not usable on its own, so its members go
        # back into the pool for the next sweep.
        deferred.extend(current)

        # Each table's queries share a group, so one sweep can place only one of
        # them. Sweeping until the pool stops shrinking uses most of the epoch.
        if len(deferred) < batch_size or len(deferred) == len(remaining):
            return batches
        remaining = deferred


def collate(tokenizer, batch, table_by_id, negatives, max_query_length, max_table_length):
    """Tokenize one batch into query, table, and optional negative inputs.

    Args:
        negatives: Optional lookup from (query, table_id) to a negative
            table_id. Keyed per example rather than per table, because the
            original mines one negative for each question.
    """
    tables = [table_by_id[e.table_id] for e in batch]

    inputs = {
        "query_inputs": encode_queries(tokenizer, [e.query for e in batch], max_query_length),
        "table_inputs": encode_tables(
            tokenizer, [t.title for t in tables], [t.table for t in tables], max_table_length
        ),
    }

    if negatives:
        mined = [table_by_id[negatives[(e.query, e.table_id)]] for e in batch]
        inputs["negative_table_inputs"] = encode_tables(
            tokenizer, [t.title for t in mined], [t.table for t in mined], max_table_length
        )
    return inputs


def _to_device(inputs, device):
    return {
        name: {key: value.to(device) for key, value in batch.items()}
        for name, batch in inputs.items()
    }


@torch.no_grad()
def validation_recall(model, tokenizer, tables, examples, device, batch_size, k=10):
    """Recall@k on a split, used for early stopping."""
    index = DTRIndex.build(model, tokenizer, tables, device, batch_size=batch_size, show_progress=False)
    embeddings = encode_query_batch(
        model, tokenizer, [e.query for e in examples], device, batch_size=batch_size
    )
    ranking = index.search(embeddings, top_k=k)

    by_id = {t.table_id: t.recall_key for t in tables}
    gold = [by_id[e.table_id] for e in examples]
    model.train()
    return recall_at_k(index, ranking, gold, ks=(k,))[k]


def save_run(model, output_dir, checkpoint_dir, metrics):
    """Write the model, its config, and the tokenizer vocab to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config.to_dict(),
            "projection_dim": model.projection_dim,
        },
        output_dir / "pytorch_model.pt",
    )
    # Copied so the run directory loads through load_dual_encoder unaided.
    shutil.copy(Path(checkpoint_dir) / "vocab.txt", output_dir / "vocab.txt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def train(checkpoint, output_dir, csv_dir=None, train_split="train", val_split="validation",
          batch_size=DEFAULT_BATCH_SIZE, learning_rate=DEFAULT_LEARNING_RATE, epochs=10,
          max_steps=None, eval_every=200, val_queries=1000, dropout=DEFAULT_DROPOUT,
          max_query_length=DEFAULT_MAX_QUERY_LENGTH, max_table_length=DEFAULT_MAX_TABLE_LENGTH,
          hard_negatives=None, device=None, seed=42, encode_batch_size=32,
          gradient_checkpointing=True):
    """Fine-tune a converted checkpoint and keep the best model by validation Recall@10."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = resolve_device(device)

    model, tokenizer = load_dual_encoder(checkpoint, device=device)
    model.config.hidden_dropout_prob = dropout
    model.config.attention_probs_dropout_prob = dropout
    if gradient_checkpointing:
        model.table_encoder.gradient_checkpointing_enable()
        model.query_encoder.gradient_checkpointing_enable()
    model.train()

    adapter = KaggleDSAdapter(csv_dir=csv_dir)
    tables, examples = adapter.load(train_split)
    val_tables, val_examples = adapter.load(val_split)
    if val_queries:
        val_examples = val_examples[:val_queries]

    table_by_id = {t.table_id: t for t in tables}

    negatives = None
    if hard_negatives:
        records = json.loads(Path(hard_negatives).read_text())
        negatives = {(r["query"], r["table_id"]): r["negative_id"] for r in records}
        # Drop examples whose every candidate was filtered out as a sibling.
        examples = [e for e in examples if (e.query, e.table_id) in negatives]

    batches_per_epoch = len(group_aware_batches(examples, table_by_id, batch_size, rng))
    total_steps = max_steps or batches_per_epoch * epochs

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * DEFAULT_WARMUP_RATIO), total_steps
    )

    print(f"device={device} batch={batch_size} steps/epoch={batches_per_epoch} total_steps={total_steps}")
    print(f"train examples={len(examples)} tables={len(tables)} | val queries={len(val_examples)}")

    best_recall, step, done = -1.0, 0, False
    history = []

    for epoch in range(epochs):
        if done:
            break
        for batch in tqdm(
            group_aware_batches(examples, table_by_id, batch_size, rng),
            desc=f"epoch {epoch}", unit="step",
        ):
            inputs = collate(
                tokenizer, batch, table_by_id, negatives, max_query_length, max_table_length
            )
            loss = model(**_to_device(inputs, device)).loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % eval_every == 0 or step == total_steps:
                recall = validation_recall(
                    model, tokenizer, val_tables, val_examples, device, encode_batch_size
                )
                history.append({"step": step, "loss": float(loss), "val_recall@10": recall})
                print(f"  step {step}: loss={float(loss):.4f} val R@10={recall:.4f}")

                if recall > best_recall:
                    best_recall = recall
                    save_run(model, output_dir, checkpoint,
                             {"best_val_recall@10": best_recall, "step": step, "history": history})
                    print(f"  saved new best to {output_dir}")

            if step >= total_steps:
                done = True
                break

    print(f"\nBest validation Recall@10: {best_recall:.4f}")
    return best_recall


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--csv-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--val-queries", type=int, default=1000)
    parser.add_argument("--max-table-length", type=int, default=DEFAULT_MAX_TABLE_LENGTH)
    parser.add_argument("--hard-negatives", default=None, help="JSON from mine_hard_negatives")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    train(
        checkpoint=args.checkpoint, output_dir=args.output_dir, csv_dir=args.csv_dir,
        batch_size=args.batch_size, learning_rate=args.learning_rate, epochs=args.epochs,
        max_steps=args.max_steps, eval_every=args.eval_every, val_queries=args.val_queries,
        max_table_length=args.max_table_length, hard_negatives=args.hard_negatives,
        device=args.device, seed=args.seed,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )


if __name__ == "__main__":
    main()
