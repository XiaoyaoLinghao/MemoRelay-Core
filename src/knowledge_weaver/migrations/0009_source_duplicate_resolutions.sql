-- Reviewed native compression duplicates. Original sources/jobs/attempts stay intact.
CREATE TABLE IF NOT EXISTS source_duplicate_resolutions (
    failed_job_id TEXT PRIMARY KEY REFERENCES jobs(job_id),
    canonical_job_id TEXT NOT NULL REFERENCES jobs(job_id),
    source_parts_json TEXT NOT NULL,
    proof_sha256 TEXT NOT NULL CHECK(length(proof_sha256)=64 AND proof_sha256 NOT GLOB '*[^0-9a-f]*'),
    reviewer TEXT NOT NULL CHECK(reviewer GLOB 'operator:*'),
    reviewed_at TEXT NOT NULL,
    CHECK(failed_job_id <> canonical_job_id)
);
