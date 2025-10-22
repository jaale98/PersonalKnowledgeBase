CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS notes (
  id          BIGSERIAL PRIMARY KEY,
  title       TEXT NOT NULL,
  body        TEXT NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  embedding   VECTOR(1536)
);

CREATE INDEX IF NOT EXISTS idx_notes_created_at_desc ON notes (created_at DESC);

CREATE TABLE IF NOT EXISTS tags (
  id   BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS note_tags (
  note_id BIGINT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  tag_id  BIGINT NOT NULL REFERENCES tags(id)  ON DELETE CASCADE,
  PRIMARY KEY (note_id, tag_id)
);

DROP INDEX IF EXISTS idx_notes_embedding_ivfflat;
CREATE INDEX idx_notes_embedding_ivfflat
  ON notes
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Optional: FTS (toggle later)
-- ALTER TABLE notes ADD COLUMN IF NOT EXISTS fts tsvector
--   GENERATED ALWAYS AS (to_tsvector('simple', coalesce(title,'') || ' ' || coalesce(body,''))) STORED;
-- CREATE INDEX IF NOT EXISTS idx_notes_fts ON notes USING GIN(fts);

CREATE INDEX IF NOT EXISTS idx_notes_title_trgm ON notes USING GIN (title gin_trgm_ops);