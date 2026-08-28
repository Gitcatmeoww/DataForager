"""Unit tests for the DTR markdown parser (pure logic, no model / network)."""

from evaluation.dtr.serialization import parse_markdown_table

SIMPLE = """| date | value |
| --- | --- |
| 2024-01-01 | 5 |
| 2024-01-02 | 7 |
| 2024-01-03 | 9 |"""


def test_parses_header_and_rows():
    frame = parse_markdown_table(SIMPLE)

    assert list(frame.columns) == ["date", "value"]
    assert frame.shape == (3, 2)
    assert frame.iloc[0]["date"] == "2024-01-01"


def test_max_rows_truncates_data_rows_only():
    # The train split ships nine example rows and test only two, so callers pin
    # this to keep the two splits serialized the same way.
    frame = parse_markdown_table(SIMPLE, max_rows=2)

    assert frame.shape == (2, 2)
    assert list(frame.columns) == ["date", "value"]


def test_separator_row_is_not_data():
    assert "---" not in parse_markdown_table(SIMPLE)["date"].tolist()


def test_ragged_rows_are_padded_and_clipped():
    markdown = """| a | b | c |
| --- | --- | --- |
| 1 |
| 1 | 2 | 3 | 4 |"""

    frame = parse_markdown_table(markdown)

    assert frame.shape == (2, 3)
    assert frame.iloc[0].tolist() == ["1", "", ""]
    assert frame.iloc[1].tolist() == ["1", "2", "3"]


def test_duplicate_and_blank_headers_are_made_unique():
    # The tokenizer indexes columns by name, so names must be unique.
    frame = parse_markdown_table("| x |  | x |\n| --- | --- | --- |\n| 1 | 2 | 3 |")

    assert len(set(frame.columns)) == 3
    assert frame.columns[0] == "x"


def test_empty_input_returns_empty_frame():
    assert parse_markdown_table("").empty
    assert parse_markdown_table(None).empty


def test_header_only_table_has_columns_but_no_rows():
    frame = parse_markdown_table("| a | b |\n| --- | --- |")

    assert list(frame.columns) == ["a", "b"]
    assert frame.shape[0] == 0


def test_cells_keep_inner_whitespace_but_are_stripped():
    frame = parse_markdown_table("| a |\n| --- |\n|   hello world   |")

    assert frame.iloc[0]["a"] == "hello world"
