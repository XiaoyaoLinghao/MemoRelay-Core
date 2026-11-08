CREATE TABLE IF NOT EXISTS source_semantic_exclusions (
    event_id TEXT NOT NULL,
    event_revision INTEGER NOT NULL,
    part_index INTEGER NOT NULL,
    body_sha256 TEXT NOT NULL CHECK(length(body_sha256)=64 AND body_sha256 NOT GLOB '*[^0-9a-f]*'),
    reason TEXT NOT NULL CHECK(reason IN ('clock_reading','historical_recall')),
    proof_sha256 TEXT NOT NULL CHECK(length(proof_sha256)=64 AND proof_sha256 NOT GLOB '*[^0-9a-f]*'),
    reviewer TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    PRIMARY KEY(event_id,event_revision,part_index),
    FOREIGN KEY(event_id,event_revision,part_index) REFERENCES event_parts(event_id,revision,part_index)
);
