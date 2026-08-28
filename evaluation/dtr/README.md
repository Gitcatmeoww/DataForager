# DTR: a learned retrieval baseline

Re-implementation of the dense table retriever from Herzig et al. (2021),
*Open Domain Question Answering over Tables via Dense Retrieval* (NAACL), used
as the **learned** baseline next to the frozen-embedding baselines in
[`../eval_methods.py`](../eval_methods.py).

Every other baseline in the paper uses off-the-shelf representations
(`text-embedding-3-small`, BM25). DTR is instead fine-tuned on the KaggleDS
train split, which answers the obvious question of what happens when the
retriever is trained on the task-to-table distribution rather than merely
prompted.

## Results

Test split, 7,011 task queries over 2,337 tables. Recall is scored on the bare
table name, matching the other baselines.

| Method | Training data | R@1 | R@10 | R@20 | R@30 | R@40 | R@50 |
| ------ | ------------- | --- | ---- | ---- | ---- | ---- | ---- |
| DTR, no fine-tuning | none | .063 | .248 | .344 | .418 | .466 | .504 |
| **DTR** | KaggleDS train | .208 | **.575** | .691 | .758 | .795 | **.823** |
| DTR +hn | KaggleDS train | .206 | .565 | .686 | .743 | .783 | .810 |
| HySE over DTR | KaggleDS train | .209 | .566 | .683 | .749 | .787 | .817 |
| HySE over DTR +hn | KaggleDS train | .208 | .556 | .677 | .743 | .781 | .813 |

Three observations, all of which cut against the original paper's findings and
are worth stating plainly:

- **Fine-tuning is what matters.** It more than doubles R@10, from .248 to .575.
- **Hard negatives do not help here** (-1.0 pp R@10), where the paper reports
  +5 pp on NQ-Tables. Their corpus holds 169,898 tables, so a batch of 256
  in-batch negatives covers 0.15% of it; here the same batch covers 9.4% of the
  2,715 train tables, so in-batch negatives are already close to hard. Mining
  also costs 11.5% of the training pairs, whose every candidate was a sibling.
- **HySE does not transfer onto DTR** (+0.1 pp R@10, inside noise, and negative
  at higher k). Fine-tuning specializes the table tower on real Kaggle tables,
  and a hypothetical schema, with its invented column names and synthetic
  values, is off-distribution for it. A general-purpose encoder treats real and
  hypothetical tables alike, which is plausibly why HySE works there and not
  here.

## Pipeline

The stages are separate so each can be rerun without repeating the others.

```bash
# 1. Convert a released TF checkpoint. The dump stage needs a scratch venv with
#    TensorFlow; the convert stage runs in the project venv.
python -m evaluation.dtr.convert_tf_checkpoint dump    --checkpoint checkpoints/tapas_dual_encoder_proj_256_medium
python -m evaluation.dtr.convert_tf_checkpoint convert  --checkpoint checkpoints/tapas_dual_encoder_proj_256_medium

# 2. Fine-tune with in-batch negatives.
python -m evaluation.dtr.train --checkpoint checkpoints/tapas_dual_encoder_proj_256_medium \
    --csv-dir evaluation/huggingface --output-dir evaluation/dtr/runs/dtr

# 3. Mine one hard negative per training question, then retrain from the
#    pre-trained checkpoint, as the original does.
python -m evaluation.dtr.mine_hard_negatives --model evaluation/dtr/runs/dtr \
    --csv-dir evaluation/huggingface --output evaluation/dtr/runs/hard_negatives.json
python -m evaluation.dtr.train --checkpoint checkpoints/tapas_dual_encoder_proj_256_medium \
    --csv-dir evaluation/huggingface --output-dir evaluation/dtr/runs/dtr_hn \
    --hard-negatives evaluation/dtr/runs/hard_negatives.json

# 4. Evaluate, and evaluate the HySE fusion.
python -m evaluation.dtr.evaluate_dtr --checkpoint evaluation/dtr/runs/dtr \
    --split test --csv-dir evaluation/huggingface --index-path evaluation/dtr/runs/dtr_test_index.npz
PYTHONPATH=src python -m evaluation.dtr.hyse_over_dtr --model evaluation/dtr/runs/dtr \
    --split test --csv-dir evaluation/huggingface --index-path evaluation/dtr/runs/dtr_test_index.npz
```

Checkpoints come from the [TAPAS repository](https://github.com/google-research/tapas/blob/master/DENSE_TABLE_RETRIEVER.md),
under `https://storage.googleapis.com/tapas_models/2021_04_27/`. Only
`hyse_over_dtr` touches Postgres, to read cached hypothetical schemas; training,
indexing, and evaluation read the split CSVs and need no database.

## Details that are easy to get wrong

Verified against `tapas/models/table_retriever_model.py` rather than assumed:

- **`bert` is the table tower and `bert_1` the query tower.** The table model is
  constructed first. Swapping them yields a model that still trains and still
  scores, while encoding queries with the table encoder.
- **`[CLS]` is the pooled output**, dense plus tanh, not the raw first hidden
  state.
- **The query tower is given an empty table.** All seven TAPAS token type ids
  are zeroed and the query is capped at its own shorter length.
- **The projections are `[proj_dim, hidden]` and applied with `transpose_b`**,
  so they copy straight into `nn.Linear` with no transpose, unlike every dense
  kernel, which needs one.
- **ICT pre-training is load-bearing.** Start from
  `tapas_dual_encoder_proj_256_*`, never from a raw `google/tapas-*` checkpoint:
  the paper's DTR-pt ablation falls from 76.0 to 47.8 R@10.

The conversion is gated on a tensor-by-tensor diff against
[`xhluca/tapas-nq-hn-retriever-medium-{0,1}`](https://huggingface.co/xhluca/tapas-nq-hn-retriever-medium-0),
an independent conversion of the same checkpoints. Both towers match exactly.

### Sibling tables

Multi-table Kaggle datasets often hold near-identical tables, 1,459 of them with
identical schemas. A query written from a dataset description can be answered by
several of them, but only one is labelled gold, so the rest are false negatives.
Every table therefore carries a `group_id`, its `database_name`, and both the
batch sampler and the hard-negative miner refuse to put two members of a group
in opposition. This is the analogue of the original's rule that a mined negative
must not contain the reference answer. It matters: 11.5% of training questions
had no non-sibling candidate anywhere in their top 50.

## Deviations from the paper

1. `tapas_dual_encoder_proj_256_medium` rather than `large`, for compute.
2. Tables are serialized as title, header, and two example rows, because the
   corpus artifacts retain no more than that. It also makes the comparison
   against HySE exactly content-matched.
3. `max_seq_length` 256 rather than 512, a consequence of (2). Wide tables are
   narrowed to fit, following the original's trimming rule; the corpus reaches
   3,002 columns.
4. Gradient checkpointing, which is what makes the paper's batch of 256 fit on
   one machine.
5. Exhaustive search over 256-dimensional vectors. Same algorithm as the paper,
   far smaller corpus, but not comparable to the HNSW timings the other methods
   report.
6. Task queries are LLM-generated from dataset descriptions, so both DTR and
   HySE are measured against synthetic query phrasing.
