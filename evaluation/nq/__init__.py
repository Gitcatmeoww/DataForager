"""NQ-Tables corpus support for the transfer evaluation.

NQ-Tables (Herzig et al., 2021) is the second evaluation corpus: an external
table collection with real, human-written questions, used as a transfer test
rather than as the target setting.

The corpus is assembled from two published releases, because neither is
sufficient alone:

- trl-lab/openti-nq-tables supplies the tables themselves, as one parquet per
  table, plus the questions and their answers. It drops the Wikipedia page
  title, which is where an NQ table's entity lives.
- ibm-research/NQTablesRetrieval supplies the titles, along with official
  qrels and the full 169,898-table corpus.

build_title_map joins the two so the OpenTI tables can carry their titles.
"""
