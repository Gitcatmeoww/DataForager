-- Enable the `pgvector` extension
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS corpus_raw_metadata_with_embedding;

CREATE TABLE corpus_raw_metadata_with_embedding (
    table_name TEXT PRIMARY KEY,
    col_num INT,
    popularity INT,
    time_granu TEXT[],
    geo_granu TEXT[],
    comb_embed VECTOR(1536),
    query_embed VECTOR(1536)
);

-- Add index for the embeddings using HNSW cosine distance
-- Note: Vectors with up to 2,000 dimensions can be indexed
-- `m` - the max number of connections per layer (16 by default)
-- `ef_construction` - the size of the dynamic candidate list for constructing the graph (64 by default). A higher value of ef_construction provides better recall at the cost of index build time / insert speed
CREATE INDEX IF NOT EXISTS comb_embed_idx ON corpus_raw_metadata_with_embedding USING hnsw (comb_embed vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS query_embed_idx ON corpus_raw_metadata_with_embedding USING hnsw (query_embed vector_cosine_ops) WITH (m = 16, ef_construction = 64);