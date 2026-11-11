//! Skills: named, versioned instructions an assistant loads on demand.
//!
//! A skill is a short description plus a body of instructions. Only the name
//! and description are shown to a model up front; the body is fetched when the
//! model decides the skill applies. Skills are scoped to one namespace.

use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};

pub const MAX_NAME_LEN: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Skill {
    pub id: String,
    pub namespace: String,
    pub name: String,
    pub description: String,
    pub content: String,
    pub version: i64,
    pub created_at: String,
    pub updated_at: String,
}

/// Names are lowercase letters, digits and hyphens, so they are safe as
/// folder names and URL segments on every platform.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME_LEN
        && !name.starts_with('-')
        && !name.ends_with('-')
        && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

const COLUMNS: &str = "id, namespace, name, description, content, version, created_at, updated_at";

fn from_row(row: &Row) -> rusqlite::Result<Skill> {
    Ok(Skill {
        id: row.get(0)?,
        namespace: row.get(1)?,
        name: row.get(2)?,
        description: row.get(3)?,
        content: row.get(4)?,
        version: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
    })
}

/// Skills of one namespace, most recently updated first.
pub fn list(conn: &Connection, namespace: &str) -> rusqlite::Result<Vec<Skill>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {COLUMNS} FROM skills WHERE namespace = ?1 ORDER BY updated_at DESC, name ASC"
    ))?;
    let rows = stmt.query_map(params![namespace], from_row)?;
    rows.collect()
}

/// Skills of every namespace.
pub fn list_all(conn: &Connection) -> rusqlite::Result<Vec<Skill>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {COLUMNS} FROM skills ORDER BY namespace ASC, name ASC"
    ))?;
    let rows = stmt.query_map([], from_row)?;
    rows.collect()
}

pub fn get(conn: &Connection, namespace: &str, name: &str) -> rusqlite::Result<Option<Skill>> {
    conn.query_row(
        &format!("SELECT {COLUMNS} FROM skills WHERE namespace = ?1 AND name = ?2"),
        params![namespace, name],
        from_row,
    )
    .optional()
}

/// Create a skill or update it. The version goes up only when the
/// description or content actually changes.
pub fn upsert(
    conn: &Connection,
    namespace: &str,
    name: &str,
    description: &str,
    content: &str,
) -> rusqlite::Result<Skill> {
    let now = Utc::now().to_rfc3339();
    match get(conn, namespace, name)? {
        Some(existing) if existing.description == description && existing.content == content => {
            Ok(existing)
        }
        Some(existing) => {
            conn.execute(
                "UPDATE skills SET description = ?1, content = ?2, version = version + 1, updated_at = ?3
                 WHERE id = ?4",
                params![description, content, now, existing.id],
            )?;
            Ok(Skill {
                description: description.to_string(),
                content: content.to_string(),
                version: existing.version + 1,
                updated_at: now,
                ..existing
            })
        }
        None => {
            let skill = Skill {
                id: ulid::Ulid::new().to_string(),
                namespace: namespace.to_string(),
                name: name.to_string(),
                description: description.to_string(),
                content: content.to_string(),
                version: 1,
                created_at: now.clone(),
                updated_at: now,
            };
            put(conn, &skill)?;
            Ok(skill)
        }
    }
}

/// Store a skill exactly as given, replacing any skill with the same
/// namespace and name. Used by restore, which must keep versions and dates.
pub fn put(conn: &Connection, skill: &Skill) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO skills (id, namespace, name, description, content, version, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(namespace, name) DO UPDATE SET
            description = excluded.description,
            content = excluded.content,
            version = excluded.version,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at",
        params![
            skill.id,
            skill.namespace,
            skill.name,
            skill.description,
            skill.content,
            skill.version,
            skill.created_at,
            skill.updated_at
        ],
    )?;
    Ok(())
}

/// Returns whether a skill was removed.
pub fn delete(conn: &Connection, namespace: &str, name: &str) -> rusqlite::Result<bool> {
    let n = conn.execute(
        "DELETE FROM skills WHERE namespace = ?1 AND name = ?2",
        params![namespace, name],
    )?;
    Ok(n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(crate::schema::SKILLS_SCHEMA).unwrap();
        conn
    }

    #[test]
    fn names_are_restricted_to_safe_slugs() {
        assert!(valid_name("deploy-api"));
        assert!(valid_name("a1"));
        for bad in ["", "Deploy", "has space", "../etc", "-lead", "trail-", "a/b", "é"] {
            assert!(!valid_name(bad), "{bad:?} should be rejected");
        }
        assert!(!valid_name(&"a".repeat(MAX_NAME_LEN + 1)));
    }

    #[test]
    fn upsert_bumps_version_only_on_change() {
        let c = conn();
        let first = upsert(&c, "default", "deploy", "Ship the API", "steps").unwrap();
        assert_eq!(first.version, 1);
        let same = upsert(&c, "default", "deploy", "Ship the API", "steps").unwrap();
        assert_eq!(same.version, 1);
        assert_eq!(same.updated_at, first.updated_at);
        let changed = upsert(&c, "default", "deploy", "Ship the API", "new steps").unwrap();
        assert_eq!(changed.version, 2);
        assert_eq!(get(&c, "default", "deploy").unwrap().unwrap().content, "new steps");
    }

    #[test]
    fn skills_are_scoped_to_their_namespace() {
        let c = conn();
        upsert(&c, "default", "deploy", "d", "a").unwrap();
        upsert(&c, "work", "deploy", "d", "b").unwrap();
        assert_eq!(list(&c, "default").unwrap().len(), 1);
        assert_eq!(get(&c, "work", "deploy").unwrap().unwrap().content, "b");
        assert_eq!(list_all(&c).unwrap().len(), 2);
        assert!(delete(&c, "default", "deploy").unwrap());
        assert!(!delete(&c, "default", "deploy").unwrap());
        assert_eq!(list_all(&c).unwrap().len(), 1);
    }

    #[test]
    fn put_keeps_the_given_version_and_dates() {
        let c = conn();
        upsert(&c, "default", "deploy", "old", "old").unwrap();
        let restored = Skill {
            id: "x".into(),
            namespace: "default".into(),
            name: "deploy".into(),
            description: "restored".into(),
            content: "body".into(),
            version: 7,
            created_at: "2026-01-01T00:00:00+00:00".into(),
            updated_at: "2026-02-01T00:00:00+00:00".into(),
        };
        put(&c, &restored).unwrap();
        let got = get(&c, "default", "deploy").unwrap().unwrap();
        assert_eq!((got.version, got.description.as_str()), (7, "restored"));
        assert_eq!(got.updated_at, restored.updated_at);
        assert_eq!(list(&c, "default").unwrap().len(), 1);
    }
}
