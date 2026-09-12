# NQ-Tables: the transfer corpus

NQ-Tables (Herzig et al., 2021) is the second evaluation corpus. Its questions
are real and human-written, which is what the Kaggle benchmark cannot offer, so
it is used as a **transfer test** rather than as the target setting: the
questions are factoid, not analytical, and FlexPrune has no metadata predicates
to compile against and is not evaluated here.

## Which release, and why two

Neither published release is sufficient alone.

| | `trl-lab/openti-nq-tables` | `ibm-research/NQTablesRetrieval` |
| --- | --- | --- |
| tables | 8,205, one parquet each | 169,898 |
| questions | 11,628 with answers | 11,628 |
| qrels | one target per instance | official, a few with two golds |
| page title | **absent** | `meta_data` |

We evaluate on the OpenTI corpus, whose 8,205 tables are exactly the union of
gold tables. That matches the KaggleDS protocol, where `eval_data_test.csv`
holds 2,337 tables each carrying its own three task queries, so every table is
gold for something and the corpus contains no distractors either.

The consequence is that **published DTR numbers do not apply**. Herzig et al.
report .449 / .798 / .911 (R@1/10/50, `medium` with hard negatives) over all
169,898 tables; at 8,205 the haystack is 20.7x smaller and recall is not
comparable. DTR has to be run here rather than cited. That costs inference only,
since the released NQ checkpoints are already fine-tuned.

## The title problem

OpenTI drops the Wikipedia page title. In NQ-Tables that is where the entity
lives. The gold table for

> when was the last time easter was in the month of march

has the header `Year | Western | Eastern` and the word *Easter* appears nowhere
in its cells — only in the title, **List of dates for Easter**. Confirmed absent
from every field OpenTI ships: all 11,628 instances' metadata carries only
`split`, `original_question_id`, `interaction_id`, and `all_answer_texts`;
`tables_index.parquet` carries only shape and hash columns; no sampled parquet
carries schema metadata. Separately, 61.4% of tables have at least one blank or
placeholder column name (`''`, `_2`, `_3`), so headers alone do not identify a
table either.

Dropping the title would both cripple retrieval and put the released TAPAS
checkpoints off-distribution, since DTR serializes title, header, and rows.

## The join

`build_title_map.py` recovers the titles from the IBM release. The two use
unrelated table ids, but OpenTI's `original_question_id` is IBM's `qid`, so
tables join through the questions that point at them:

    openti table -> its instances -> qid -> IBM qrels -> did -> corpus title

Every OpenTI table is gold for at least one question, so this covers the corpus.
Content hashing breaks the tie for the few questions IBM labels with two gold
tables, and then serves as an independent check — the id path and the content
path share no inputs, so their agreement is evidence rather than restatement.

```bash
python -m evaluation.nq.build_title_map \
    --data-dir <dir holding both releases> \
    --output evaluation/nq/data/openti_title_map.parquet
```

Result, over all 8,205 tables:

| | |
| --- | --- |
| resolved by qid alone | 8,154 (99.4%) |
| resolved by qid + content tiebreak | 51 (0.6%) |
| carrying a title | **8,205 (100%)** |
| confirmed by the independent content hash | **8,205 (100%)** |
| distinct dids | 8,205, so the map is a bijection |

The committed `data/openti_title_map.parquet` (677 KB) is that output, so the
corpus is reproducible without re-downloading IBM's 860 MB
`corpus_structure.jsonl`.

## The adapter

`NQTablesAdapter` implements the same `CorpusAdapter` protocol as
`KaggleDSAdapter`, so the DTR index, the evaluator, and the HySE fusion read
this corpus the way they read KaggleDS.

```python
from evaluation.nq.adapters import NQTablesAdapter

tables, examples = NQTablesAdapter(data_dir).load("test")
# 8,205 TableRecords, 966 RetrievalExamples over 959 questions
```

Loading all 8,205 tables takes about 14 seconds and is cached on the adapter,
so a second `load` for another split re-reads nothing.

Three choices differ from KaggleDS and change how results read:

| | KaggleDS | NQ-Tables |
| --- | --- | --- |
| corpus scope | per split | **one corpus, all splits** |
| `recall_key` | `table_name` | the did, a content hash |
| `group_id` | `database_name` | the did minus its hash, i.e. the Wikipedia page |

**The corpus is not split-scoped.** NQ-Tables has one 8,205-table corpus that
every split retrieves against; only the questions are split-scoped. Narrowing
test to its own 896 gold tables would shrink the haystack ninefold and inflate
recall.

**Recall is scored on the did.** It is a content hash, so unlike KaggleDS's
`table_name` it cannot collide across unrelated tables — the C3 problem does not
arise here.

**Groups are Wikipedia pages.** 1,698 of the 8,205 tables share a page with
another table, in groups of up to 12. Tables from one page can answer the same
question, so they must never be opposed as negatives, which is the same role
`database_name` plays for KaggleDS.

### Running it

```bash
python -m evaluation.dtr.evaluate_dtr \
    --checkpoint evaluation/dtr/checkpoints/tapas_nq_hn_retriever_medium \
    --corpus nq --data-dir <openti release> --split test \
    --index-path evaluation/dtr/runs/nq_test_index_medium.npz
```

DTR needs no fine-tuning here: the released `tapas_nq_hn_retriever_*`
checkpoints are already trained on NQ train, so this is inference only.

| DTR `medium` +hn | R@1 | R@10 | R@50 |
| --- | --- | --- | --- |
| here, 8,205 tables | .553 | .852 | .923 |
| Herzig et al., 169,898 tables | .449 | .798 | .911 |

The two rows are not comparable and the second is listed only as a sanity
check. Ours is higher because the corpus is 20.7x smaller, and the gap narrows
as k grows (+10.4, +5.4, +1.2 pp), which is what removing distractors looks
like.

### Multi-gold questions

A few questions carry two gold tables: 7 of 966 in test, 60 in train, 1 in dev.
OpenTI stores one instance per (question, gold table) pair, which is why its
11,628 instances match IBM's 11,628 qrel rows exactly.

The adapter emits one `RetrievalExample` per pair, which is what training wants,
and tags examples from one question with a shared `query_id`. `evaluate_dtr`
groups on it via `group_by_question` and counts a hit when *any* of a question's
gold tables is retrieved, which is why test reports 959 queries rather than 966.

`recall_at_k` accepts either a bare gold key or a set of them. The bare form is
untouched, and the KaggleDS numbers behind Table 1 were re-derived through the
changed code and are bitwise identical at every cutoff (R@10 0.57466837826273).

## Two defects in the upstream releases

Both cost real debugging time; both are worth knowing before touching these
files.

1. **IBM's qrels CSV-quote three dids.** They ship as
   `"Whenever_I_Call_You_""Friend""_5DE9FEBACB12C224"` and match no corpus row
   until unescaped. Both the `.jsonl` and the `.tsv` forms carry it, so it is
   upstream of the serialization — 3 distinct dids over 6 qrel rows, train 4,
   dev 1, **test 1**. Untreated, one of the 966 test queries has an
   unresolvable gold and scores as an automatic miss, which reads as a slightly
   worse retrieval number rather than as a bug.

   `unquote_did` handles it, and requiring *both* a leading and a trailing
   quote is what keeps it safe: 56 corpus ids genuinely begin with a quote
   (`"Heroes"_(David_Bowie_song)_19471739CF2A662`), but none end with one,
   since every id ends in a 16 character hex hash. Verified: 0 of those 56 are
   altered.

   `require_all_resolved` then makes any future recurrence fatal instead of
   silent, since the failure mode is a quietly smaller corpus.

   Note when reading the `.tsv` yourself that Python's `csv` module interprets
   quotes by default, so it will appear to have clean ids while the `.jsonl`
   looks broken. Split on tabs to see what is actually stored.
2. **IBM's corpus holds duplicate titles for one table.** The same content
   appears under, for example, both `Orange_Is_the_New_Black_ED31ED6C18A3E20B`
   and `List_of_Orange_Is_the_New_Black_episodes_ED31ED6C18A3E20B` — note the
   shared hash suffix. Joining on content alone picks an id the qrels do not
   score against, which is why the join runs through qids instead.

Separately, `ibm-research`'s `corpus_linearized` config is unusable: its ids use
spaces where the qrels and the other two configs use underscores, and its text
concatenates the last header to the first cell (`ChildrenLesley Joseph`) with no
row boundaries. Use `corpus_structure`, whose `cells` is a flat row-major list
to be reshaped by `len(headers)`.

## Attribution

NQ-Tables is derived from Natural Questions and released by Herzig et al. under
CC BY-SA 4.0. `ibm-research/NQTablesRetrieval` is CC BY-4.0; the committed title
map contains page titles derived from it.

> J. Herzig, T. Müller, S. Krichene, and J. Eisenschlos. Open Domain Question
> Answering over Tables via Dense Retrieval. NAACL 2021, pp. 512–519.
