use rusqlite::Connection;

pub fn initialize(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    conn.execute_batch("PRAGMA foreign_keys=ON;")?;
    conn.execute_batch("PRAGMA busy_timeout=5000;")?;

    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS memories (
            id           TEXT PRIMARY KEY,
            namespace    TEXT NOT NULL,
            content      TEXT NOT NULL,
            metadata     TEXT,
            embedding    BLOB NOT NULL,
            status       TEXT NOT NULL DEFAULT 'active',
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            accessed_at  TEXT NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            hash         TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
        CREATE INDEX IF NOT EXISTS idx_memories_status    ON memories(status);
        CREATE INDEX IF NOT EXISTS idx_memories_hash      ON memories(namespace, hash);
        CREATE INDEX IF NOT EXISTS idx_memories_created   ON memories(created_at);
        ",
    )?;

    // Add superseded_by column (idempotent migration)
    conn.execute_batch(
        "ALTER TABLE memories ADD COLUMN superseded_by TEXT;",
    )
    .ok(); // ignore if column already exists

    // Add tags and memory_type columns (idempotent migration)
    conn.execute_batch(
        "ALTER TABLE memories ADD COLUMN tags TEXT NOT NULL DEFAULT '[]';",
    )
    .ok();
    conn.execute_batch(
        "ALTER TABLE memories ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'fact';",
    )
    .ok();
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);",
    )?;

    // Add importance column for memory scoring (default 5 = neutral)
    conn.execute_batch(
        "ALTER TABLE memories ADD COLUMN importance INTEGER NOT NULL DEFAULT 5;",
    )
    .ok();

    // Add episode_id for episodic memory grouping
    conn.execute_batch(
        "ALTER TABLE memories ADD COLUMN episode_id TEXT;",
    )
    .ok();
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_memories_episode ON memories(namespace, episode_id);",
    )?;

    // Add event_date for temporal filtering (first-class indexed column)
    conn.execute_batch(
        "ALTER TABLE memories ADD COLUMN event_date TEXT;",
    )
    .ok();
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_memories_event_date ON memories(namespace, event_date);",
    )?;
    // Backfill event_date from metadata JSON for existing memories
    conn.execute_batch(
        "UPDATE memories SET event_date = json_extract(metadata, '$.event_date')
         WHERE event_date IS NULL
           AND json_extract(metadata, '$.event_date') IS NOT NULL;",
    ).ok();

    // Conversations table for chat history
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS conversations (
            id          TEXT PRIMARY KEY,
            namespace   TEXT NOT NULL,
            title       TEXT NOT NULL,
            mode        TEXT NOT NULL DEFAULT 'rag',
            messages    TEXT NOT NULL DEFAULT '[]',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_namespace ON conversations(namespace);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated   ON conversations(updated_at);
        ",
    )?;

    // FTS5 full-text search table
    conn.execute_batch(
        "
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
            memory_id UNINDEXED,
            namespace UNINDEXED,
            content,
            tokenize='porter unicode61'
        );
        ",
    )?;

    // Calibration observations + fitted models
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS calibration_observations (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace            TEXT NOT NULL,
            prediction_kind      TEXT NOT NULL,
            prediction_id        TEXT,
            predicted_confidence REAL NOT NULL,
            outcome              REAL,
            metadata             TEXT,
            observed_at          TEXT NOT NULL,
            resolved_at          TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_cal_obs_ns_kind
            ON calibration_observations(namespace, prediction_kind);
        CREATE INDEX IF NOT EXISTS idx_cal_obs_resolved
            ON calibration_observations(prediction_kind, resolved_at);

        CREATE TABLE IF NOT EXISTS calibration_models (
            namespace       TEXT NOT NULL,
            prediction_kind TEXT NOT NULL,
            samples         INTEGER NOT NULL,
            avg_prediction  REAL NOT NULL,
            avg_outcome     REAL NOT NULL,
            mse             REAL NOT NULL,
            slope           REAL NOT NULL,
            intercept       REAL NOT NULL,
            updated_at      TEXT NOT NULL,
            PRIMARY KEY(namespace, prediction_kind)
        );

        CREATE TABLE IF NOT EXISTS calibration_bins (
            namespace       TEXT NOT NULL,
            prediction_kind TEXT NOT NULL,
            bin_index       INTEGER NOT NULL,
            sample_count    INTEGER NOT NULL,
            avg_prediction  REAL NOT NULL,
            avg_outcome     REAL NOT NULL,
            updated_at      TEXT NOT NULL,
            PRIMARY KEY(namespace, prediction_kind, bin_index)
        );

        CREATE TABLE IF NOT EXISTS correction_events (
            id               TEXT PRIMARY KEY,
            namespace        TEXT NOT NULL,
            target_memory_id TEXT NOT NULL,
            new_memory_id    TEXT NOT NULL,
            reason           TEXT,
            metadata         TEXT,
            created_at       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_correction_events_ns
            ON correction_events(namespace, created_at);
        CREATE INDEX IF NOT EXISTS idx_correction_events_target
            ON correction_events(target_memory_id, created_at);

        CREATE TABLE IF NOT EXISTS contradiction_ledger (
            id               TEXT PRIMARY KEY,
            namespace        TEXT NOT NULL,
            old_memory_id    TEXT NOT NULL,
            new_memory_id    TEXT NOT NULL,
            resolution       TEXT NOT NULL,
            provenance       TEXT,
            created_at       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_contradiction_ledger_ns
            ON contradiction_ledger(namespace, created_at);
        CREATE INDEX IF NOT EXISTS idx_contradiction_ledger_memories
            ON contradiction_ledger(old_memory_id, new_memory_id);

        CREATE TABLE IF NOT EXISTS causal_edges (
            id               TEXT PRIMARY KEY,
            namespace        TEXT NOT NULL,
            source_memory_id TEXT NOT NULL,
            target_memory_id TEXT NOT NULL,
            relation_type    TEXT NOT NULL,
            confidence       REAL NOT NULL,
            evidence         TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            UNIQUE(namespace, source_memory_id, target_memory_id, relation_type)
        );
        CREATE INDEX IF NOT EXISTS idx_causal_edges_ns
            ON causal_edges(namespace, relation_type);
        CREATE INDEX IF NOT EXISTS idx_causal_edges_source
            ON causal_edges(source_memory_id);
        CREATE INDEX IF NOT EXISTS idx_causal_edges_target
            ON causal_edges(target_memory_id);

        CREATE TABLE IF NOT EXISTS state_transitions (
            id               TEXT PRIMARY KEY,
            namespace        TEXT NOT NULL,
            from_memory_id   TEXT NOT NULL,
            to_memory_id     TEXT NOT NULL,
            transition_type  TEXT NOT NULL,
            confidence       REAL NOT NULL,
            evidence         TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            UNIQUE(namespace, from_memory_id, to_memory_id, transition_type)
        );
        CREATE INDEX IF NOT EXISTS idx_state_transitions_ns
            ON state_transitions(namespace, transition_type);
        CREATE INDEX IF NOT EXISTS idx_state_transitions_from
            ON state_transitions(from_memory_id);
        CREATE INDEX IF NOT EXISTS idx_state_transitions_to
            ON state_transitions(to_memory_id);

        CREATE TABLE IF NOT EXISTS claim_revisions (
            id               TEXT PRIMARY KEY,
            namespace        TEXT NOT NULL,
            memory_id        TEXT NOT NULL,
            revision_number  INTEGER NOT NULL,
            operation        TEXT NOT NULL,
            content          TEXT NOT NULL,
            metadata         TEXT,
            tags             TEXT NOT NULL,
            memory_type      TEXT NOT NULL,
            importance       INTEGER NOT NULL,
            status           TEXT NOT NULL,
            superseded_by    TEXT,
            hash             TEXT NOT NULL,
            embedding        BLOB,
            actor            TEXT,
            reason           TEXT,
            provenance       TEXT,
            created_at       TEXT NOT NULL,
            UNIQUE(namespace, memory_id, revision_number)
        );
        CREATE INDEX IF NOT EXISTS idx_claim_revisions_ns_memory
            ON claim_revisions(namespace, memory_id, revision_number DESC);
        CREATE INDEX IF NOT EXISTS idx_claim_revisions_created
            ON claim_revisions(created_at DESC);

        CREATE TABLE IF NOT EXISTS procedure_revisions (
            id               TEXT PRIMARY KEY,
            namespace        TEXT NOT NULL,
            procedure_name   TEXT NOT NULL,
            revision_number  INTEGER NOT NULL,
            operation        TEXT NOT NULL,
            spec             TEXT NOT NULL,
            actor            TEXT,
            reason           TEXT,
            created_at       TEXT NOT NULL,
            UNIQUE(namespace, procedure_name, revision_number)
        );
        CREATE INDEX IF NOT EXISTS idx_procedure_revisions_ns_name
            ON procedure_revisions(namespace, procedure_name, revision_number DESC);

        CREATE TABLE IF NOT EXISTS memory_audit_log (
            id               TEXT PRIMARY KEY,
            namespace        TEXT NOT NULL,
            entity_type      TEXT NOT NULL,
            entity_id        TEXT NOT NULL,
            operation        TEXT NOT NULL,
            actor            TEXT,
            reason           TEXT,
            details          TEXT,
            created_at       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memory_audit_ns_created
            ON memory_audit_log(namespace, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_memory_audit_entity
            ON memory_audit_log(entity_type, entity_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS identity_entities (
            id               TEXT PRIMARY KEY,
            namespace        TEXT NOT NULL,
            canonical_name   TEXT NOT NULL,
            normalized_name  TEXT NOT NULL,
            language         TEXT NOT NULL DEFAULT 'und',
            confidence       REAL NOT NULL,
            ambiguity        REAL NOT NULL DEFAULT 0.0,
            metadata         TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            UNIQUE(namespace, normalized_name)
        );
        CREATE INDEX IF NOT EXISTS idx_identity_entities_ns_name
            ON identity_entities(namespace, normalized_name);

        CREATE TABLE IF NOT EXISTS identity_aliases (
            id               TEXT PRIMARY KEY,
            namespace        TEXT NOT NULL,
            entity_id        TEXT NOT NULL,
            alias            TEXT NOT NULL,
            normalized_alias TEXT NOT NULL,
            language         TEXT NOT NULL DEFAULT 'und',
            confidence       REAL NOT NULL,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            UNIQUE(namespace, normalized_alias, entity_id)
        );
        CREATE INDEX IF NOT EXISTS idx_identity_aliases_ns_alias
            ON identity_aliases(namespace, normalized_alias);
        CREATE INDEX IF NOT EXISTS idx_identity_aliases_entity
            ON identity_aliases(entity_id);

        CREATE TABLE IF NOT EXISTS working_memories (
            id                  TEXT PRIMARY KEY,
            namespace           TEXT NOT NULL,
            content             TEXT NOT NULL,
            metadata            TEXT,
            provisional_score   REAL NOT NULL DEFAULT 0.5,
            status              TEXT NOT NULL DEFAULT 'pending',
            conversation_id     TEXT,
            expires_at          TEXT,
            committed_memory_id TEXT,
            created_at          TEXT NOT NULL,
            updated_at          TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_working_memories_ns_status
            ON working_memories(namespace, status, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_working_memories_expires
            ON working_memories(status, expires_at);
        CREATE INDEX IF NOT EXISTS idx_working_memories_conversation
            ON working_memories(namespace, conversation_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS plan_traces (
            id                 TEXT PRIMARY KEY,
            namespace          TEXT NOT NULL,
            goal               TEXT NOT NULL,
            status             TEXT NOT NULL,
            priority           INTEGER NOT NULL DEFAULT 5,
            due_at             TEXT,
            metadata           TEXT,
            outcome            TEXT,
            outcome_confidence REAL,
            finished_at        TEXT,
            created_at         TEXT NOT NULL,
            updated_at         TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_plan_traces_ns_status
            ON plan_traces(namespace, status, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_plan_traces_due
            ON plan_traces(namespace, due_at);

        CREATE TABLE IF NOT EXISTS plan_checkpoints (
            id              TEXT PRIMARY KEY,
            namespace       TEXT NOT NULL,
            plan_id         TEXT NOT NULL,
            checkpoint_key  TEXT NOT NULL,
            title           TEXT NOT NULL,
            order_index     INTEGER NOT NULL DEFAULT 0,
            status          TEXT NOT NULL,
            expected_by     TEXT,
            completed_at    TEXT,
            evidence        TEXT,
            metadata        TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            UNIQUE(namespace, plan_id, checkpoint_key)
        );
        CREATE INDEX IF NOT EXISTS idx_plan_checkpoints_plan
            ON plan_checkpoints(namespace, plan_id, order_index ASC, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_plan_checkpoints_status
            ON plan_checkpoints(namespace, status, updated_at DESC);

        CREATE TABLE IF NOT EXISTS plan_recovery_branches (
            id                   TEXT PRIMARY KEY,
            namespace            TEXT NOT NULL,
            plan_id              TEXT NOT NULL,
            source_checkpoint_id TEXT,
            branch_label         TEXT NOT NULL,
            trigger_reason       TEXT NOT NULL,
            status               TEXT NOT NULL,
            branch_plan          TEXT,
            metadata             TEXT,
            resolution_notes     TEXT,
            created_at           TEXT NOT NULL,
            updated_at           TEXT NOT NULL,
            resolved_at          TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_plan_branches_plan
            ON plan_recovery_branches(namespace, plan_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_plan_branches_status
            ON plan_recovery_branches(namespace, status, updated_at DESC);

        CREATE TABLE IF NOT EXISTS plan_procedure_bindings (
            id              TEXT PRIMARY KEY,
            namespace       TEXT NOT NULL,
            plan_id         TEXT NOT NULL,
            procedure_name  TEXT NOT NULL,
            binding_role    TEXT NOT NULL,
            confidence      REAL NOT NULL,
            metadata        TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            UNIQUE(namespace, plan_id, procedure_name, binding_role)
        );
        CREATE INDEX IF NOT EXISTS idx_plan_proc_bindings_plan
            ON plan_procedure_bindings(namespace, plan_id, updated_at DESC);
        ",
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let conn = Connection::open_in_memory().unwrap();
        initialize(&conn).unwrap();
        // Verify tables exist
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='memories'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);

        let cal_count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='calibration_models'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(cal_count, 1);
    }

    #[test]
    fn test_schema_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        initialize(&conn).unwrap();
        initialize(&conn).unwrap(); // should not fail
    }
}
