from __future__ import annotations

import argparse
import ast
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "MOOPer"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "mooper.db"

SOURCE = "MOOPer"
ENTITY_DIR = Path("knowledgeGraph") / "entity"
RELATION_DIR = Path("knowledgeGraph") / "relation"
INTERACTION_DIR = Path("interaction")


ENTITY_SPECS = [
    ("course", "course_id", "course.csv", "name"),
    ("chapter", "chapter_id", "chapter.csv", "name"),
    ("exercise", "exercise_id", "exercise.csv", "name"),
    ("challenge", "challenge_id", "challenge.csv", "name"),
    ("topic", "topic_id", "topics.csv", "topic_name"),
    ("discipline", "discipline_id", "discipline.csv", "name"),
    ("subdiscipline", "sub_discipline_id", "subdiscipline.csv", "name"),
    ("teacher", "teacher_id", "teacher.csv", None),
    ("student", "student_id", "student.csv", None),
    ("school", "school_id", "school.csv", "name"),
    ("department", "department_id", "department.csv", "name"),
]

RAW_ENTITY_TABLES = {
    "course.csv": "raw_courses",
    "chapter.csv": "raw_chapters",
    "exercise.csv": "raw_exercises",
    "challenge.csv": "raw_challenges",
    "topics.csv": "raw_topics",
    "discipline.csv": "raw_disciplines",
    "subdiscipline.csv": "raw_subdisciplines",
    "teacher.csv": "raw_teachers",
    "student.csv": "raw_students",
    "school.csv": "raw_schools",
    "department.csv": "raw_departments",
}

RAW_RELATION_TABLES = {
    "challenge_teacher.csv": "raw_challenge_teacher",
    "challenge_topic.csv": "raw_challenge_topic",
    "chapter_teacher.csv": "raw_chapter_teacher",
    "course_chapter.csv": "raw_course_chapter",
    "course_teacher.csv": "raw_course_teacher",
    "discipline_course.csv": "raw_discipline_course",
    "discipline_exercise.csv": "raw_discipline_exercise",
    "exercise_challenge.csv": "raw_exercise_challenge",
    "exercise_course.csv": "raw_exercise_course",
    "exercise_teacher.csv": "raw_exercise_teacher",
    "student_institution.csv": "raw_student_institution",
    "sub_discipline.csv": "raw_sub_discipline",
    "teacher_institution.csv": "raw_teacher_institution",
    "topic_cluster.csv": "raw_topic_cluster",
}

LEARNING_RESOURCE_TYPES = ("course", "chapter", "exercise", "challenge", "topic", "discipline", "subdiscipline")


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    options = {
        "dtype": str,
        "keep_default_na": False,
        "encoding": "utf-8-sig",
        "low_memory": False,
    }
    options.update(kwargs)
    return pd.read_csv(path, **options)


def read_csv_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        chunksize=chunksize,
        low_memory=False,
    )


def stable_id(entity_type: str, values: pd.Series) -> pd.Series:
    return SOURCE.lower() + ":" + entity_type + ":" + values.astype(str)


def compact_json(row: Mapping[str, Any]) -> str:
    return json.dumps({key: value for key, value in row.items() if value != ""}, ensure_ascii=False)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def connect(db_path: Path, replace: bool) -> sqlite3.Connection:
    ensure_parent(db_path)
    if db_path.exists():
        if not replace:
            raise SystemExit("Database already exists. Use --replace to rebuild: {0}".format(db_path))
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def import_raw_csv(conn: sqlite3.Connection, path: Path, table_name: str, chunksize: Optional[int] = None) -> int:
    if chunksize:
        total = 0
        first = True
        for chunk in read_csv_chunks(path, chunksize):
            chunk.to_sql(table_name, conn, if_exists="replace" if first else "append", index=False)
            total += len(chunk)
            first = False
        if first:
            pd.DataFrame().to_sql(table_name, conn, if_exists="replace", index=False)
        return total

    df = read_csv(path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    return len(df)


def import_raw_tables(conn: sqlite3.Connection, raw_root: Path, include_outputs: bool, chunksize: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    for filename, table_name in RAW_ENTITY_TABLES.items():
        counts[table_name] = import_raw_csv(conn, raw_root / ENTITY_DIR / filename, table_name)

    for filename, table_name in RAW_RELATION_TABLES.items():
        counts[table_name] = import_raw_csv(conn, raw_root / RELATION_DIR / filename, table_name)

    counts["challenge_interactions"] = import_raw_csv(
        conn,
        raw_root / INTERACTION_DIR / "challenge_interaction.csv",
        "challenge_interactions",
        chunksize=chunksize,
    )
    counts["discussions"] = import_raw_csv(conn, raw_root / INTERACTION_DIR / "discussions.csv", "discussions")

    if include_outputs:
        counts["outputs"] = import_raw_csv(
            conn,
            raw_root / INTERACTION_DIR / "outputs.csv",
            "outputs",
            chunksize=chunksize,
        )

    return counts


def entity_frame(raw_root: Path, entity_type: str, id_field: str, filename: str, title_field: Optional[str]) -> pd.DataFrame:
    df = read_csv(raw_root / ENTITY_DIR / filename)
    out = pd.DataFrame()
    out["id"] = stable_id(entity_type, df[id_field])
    out["source"] = SOURCE
    out["entity_type"] = entity_type
    out["raw_id"] = df[id_field]
    out["title"] = df[title_field] if title_field else ""
    out["description"] = df["description"] if "description" in df.columns else ""
    out["visits"] = df["visits"] if "visits" in df.columns else ""
    out["score"] = df["score"] if "score" in df.columns else ""
    out["difficulty"] = df["difficulty"] if "difficulty" in df.columns else ""
    out["praises_count"] = df["praises_count"] if "praises_count" in df.columns else ""
    out["status"] = df["status"] if "status" in df.columns else ""
    out["province"] = df["province"] if "province" in df.columns else ""
    out["gender"] = df["gender"] if "gender" in df.columns else ""
    out["technical_title"] = df["technical_title"] if "technical_title" in df.columns else ""
    out["brief_introduction"] = df["brief_introduction"] if "brief_introduction" in df.columns else ""
    out["learning_notes"] = df["learning_notes"] if "learning_notes" in df.columns else ""
    out["created_at"] = df["created_at"] if "created_at" in df.columns else ""
    out["publish_time"] = df["publish_time"] if "publish_time" in df.columns else ""
    out["raw_table"] = "raw_" + entity_type + "s"
    out["raw_json"] = df.apply(lambda row: compact_json(row.to_dict()), axis=1)
    return out


def build_entities(conn: sqlite3.Connection, raw_root: Path) -> Dict[str, int]:
    frames = [entity_frame(raw_root, *spec) for spec in ENTITY_SPECS]
    entities = pd.concat(frames, ignore_index=True)
    entities.to_sql("entities", conn, if_exists="replace", index=False)

    resources = entities[entities["entity_type"].isin(LEARNING_RESOURCE_TYPES)].copy()
    resources = resources.rename(columns={"entity_type": "resource_type"})
    resources.to_sql("resources", conn, if_exists="replace", index=False)
    return {"entities": len(entities), "resources": len(resources)}


def id_value(entity_type: str, raw_id: Any) -> str:
    return "{0}:{1}:{2}".format(SOURCE.lower(), entity_type, raw_id)


def edge_frame(
    df: pd.DataFrame,
    source_type: str,
    source_field: str,
    relation: str,
    target_type: str,
    target_field: str,
    attrs: Sequence[str] = (),
) -> pd.DataFrame:
    df = df[(df[source_field].astype(str).str.strip() != "") & (df[target_field].astype(str).str.strip() != "")].copy()
    out = pd.DataFrame()
    out["source_id"] = stable_id(source_type, df[source_field])
    out["source_type"] = source_type
    out["relation"] = relation
    out["target_id"] = stable_id(target_type, df[target_field])
    out["target_type"] = target_type
    out["source"] = SOURCE
    for attr in attrs:
        out[attr] = df[attr] if attr in df.columns else ""
    if "position" not in out.columns:
        out["position"] = ""
    if "created_at" not in out.columns:
        out["created_at"] = ""
    return out


def parse_topic_cluster_edges(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        cluster_key = str(row.get("cluster_key", ""))
        raw_value = str(row.get("cluster_value", "")).strip()
        if not cluster_key or not raw_value:
            continue
        try:
            topic_ids = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(topic_ids, list):
            continue
        for topic_id in topic_ids:
            rows.append(
                {
                    "source_id": id_value("topic_cluster", cluster_key),
                    "source_type": "topic_cluster",
                    "relation": "contains_topic",
                    "target_id": id_value("topic", topic_id),
                    "target_type": "topic",
                    "source": SOURCE,
                    "position": "",
                    "created_at": "",
                }
            )
    return pd.DataFrame(rows)


def build_graph_edges(conn: sqlite3.Connection, raw_root: Path) -> Dict[str, int]:
    relation_root = raw_root / RELATION_DIR
    frames = [
        edge_frame(read_csv(relation_root / "course_chapter.csv"), "course", "course_id", "has_chapter", "chapter", "chapter_id", ["position"]),
        edge_frame(read_csv(relation_root / "exercise_course.csv"), "course", "course_id", "has_exercise", "exercise", "exercise_id", ["position"]),
        edge_frame(read_csv(relation_root / "exercise_challenge.csv"), "exercise", "exercise_id", "has_challenge", "challenge", "challenge_id", ["position"]),
        edge_frame(read_csv(relation_root / "challenge_topic.csv"), "challenge", "challenge_id", "has_topic", "topic", "topic_id", ["created_at"]),
        edge_frame(read_csv(relation_root / "discipline_course.csv"), "course", "course_id", "belongs_to_subdiscipline", "subdiscipline", "sub_discipline_id", ["created_at"]),
        # The released CSV is named discipline_exercise.csv, but its header is course_id,sub_discipline_id,created_at.
        edge_frame(read_csv(relation_root / "discipline_exercise.csv"), "course", "course_id", "belongs_to_subdiscipline", "subdiscipline", "sub_discipline_id", ["created_at"]),
        edge_frame(read_csv(relation_root / "sub_discipline.csv"), "subdiscipline", "sub_discipline_id", "belongs_to_discipline", "discipline", "discipline_id"),
        edge_frame(read_csv(relation_root / "course_teacher.csv"), "course", "course_id", "has_teacher", "teacher", "creator_id"),
        edge_frame(read_csv(relation_root / "chapter_teacher.csv"), "chapter", "chapter_id", "has_teacher", "teacher", "creator_id"),
        edge_frame(read_csv(relation_root / "challenge_teacher.csv"), "challenge", "challenge_id", "has_teacher", "teacher", "creator_id"),
        edge_frame(read_csv(relation_root / "exercise_teacher.csv"), "exercise", "exercise_id", "has_teacher", "teacher", "creator_id"),
    ]

    student_institution = read_csv(relation_root / "student_institution.csv")
    frames.append(edge_frame(student_institution, "student", "student_id", "belongs_to_school", "school", "school_id"))
    frames.append(edge_frame(student_institution, "student", "student_id", "belongs_to_department", "department", "department_id"))

    teacher_institution = read_csv(relation_root / "teacher_institution.csv")
    frames.append(edge_frame(teacher_institution, "teacher", "teacher_id", "belongs_to_school", "school", "school_id"))
    frames.append(edge_frame(teacher_institution, "teacher", "teacher_id", "belongs_to_department", "department", "department_id"))

    frames.append(parse_topic_cluster_edges(read_csv(relation_root / "topic_cluster.csv")))

    edges = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    edges = edges.drop_duplicates()
    edges.to_sql("graph_edges", conn, if_exists="replace", index=False)

    resource_edges = edges[
        edges["source_type"].isin(LEARNING_RESOURCE_TYPES)
        & edges["target_type"].isin(tuple(LEARNING_RESOURCE_TYPES) + ("teacher",))
    ].copy()
    resource_edges.to_sql("resource_edges", conn, if_exists="replace", index=False)
    return {"graph_edges": len(edges), "resource_edges": len(resource_edges)}


def join_names(rows: Iterable[Any]) -> str:
    values = []
    seen = set()
    for value in rows:
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none"} and text not in seen:
            values.append(text)
            seen.add(text)
    return "|".join(values)


def build_course_catalog(conn: sqlite3.Connection, raw_root: Path) -> Dict[str, int]:
    entity_root = raw_root / ENTITY_DIR
    relation_root = raw_root / RELATION_DIR

    courses = read_csv(entity_root / "course.csv")
    course_chapter = read_csv(relation_root / "course_chapter.csv")
    exercise_course = read_csv(relation_root / "exercise_course.csv")
    exercise_challenge = read_csv(relation_root / "exercise_challenge.csv")
    challenge_topic = read_csv(relation_root / "challenge_topic.csv")
    topics = read_csv(entity_root / "topics.csv")
    subdisciplines = read_csv(entity_root / "subdiscipline.csv")
    disciplines = read_csv(entity_root / "discipline.csv")
    discipline_course = pd.concat(
        [
            read_csv(relation_root / "discipline_course.csv"),
            read_csv(relation_root / "discipline_exercise.csv"),
        ],
        ignore_index=True,
    ).drop_duplicates()
    sub_discipline = read_csv(relation_root / "sub_discipline.csv")

    chapter_counts = course_chapter.groupby("course_id")["chapter_id"].nunique().rename("chapter_count")
    exercise_counts = exercise_course.groupby("course_id")["exercise_id"].nunique().rename("exercise_count")

    challenge_by_course = exercise_course.merge(exercise_challenge, on="exercise_id", how="left")
    challenge_counts = challenge_by_course.groupby("course_id")["challenge_id"].nunique().rename("challenge_count")

    topic_by_course = challenge_by_course.merge(challenge_topic, on="challenge_id", how="left")
    topic_by_course = topic_by_course.merge(topics, on="topic_id", how="left")
    topic_names = topic_by_course.groupby("course_id")["topic_name"].apply(join_names).rename("knowledge_point_names")
    topic_ids = topic_by_course.groupby("course_id")["topic_id"].apply(join_names).rename("knowledge_point_raw_ids")

    sub_by_course = discipline_course.merge(subdisciplines, on="sub_discipline_id", how="left")
    sub_names = sub_by_course.groupby("course_id")["name"].apply(join_names).rename("subdiscipline_names")
    sub_ids = sub_by_course.groupby("course_id")["sub_discipline_id"].apply(join_names).rename("subdiscipline_raw_ids")

    disc_by_course = sub_by_course.merge(sub_discipline, on="sub_discipline_id", how="left")
    disc_by_course = disc_by_course.merge(disciplines, on="discipline_id", how="left", suffixes=("", "_discipline"))
    disc_names = disc_by_course.groupby("course_id")["name_discipline"].apply(join_names).rename("discipline_names")
    disc_ids = disc_by_course.groupby("course_id")["discipline_id"].apply(join_names).rename("discipline_raw_ids")

    catalog = courses.set_index("course_id")
    for series in [chapter_counts, exercise_counts, challenge_counts, topic_names, topic_ids, sub_names, sub_ids, disc_names, disc_ids]:
        catalog = catalog.join(series, how="left")

    catalog = catalog.reset_index()
    catalog.insert(0, "id", stable_id("course", catalog["course_id"]))
    catalog.insert(1, "source", SOURCE)
    catalog = catalog.rename(columns={"course_id": "raw_id", "name": "title"})
    for column in ["chapter_count", "exercise_count", "challenge_count"]:
        catalog[column] = catalog[column].fillna("0").astype(str)
    for column in [
        "knowledge_point_names",
        "knowledge_point_raw_ids",
        "subdiscipline_names",
        "subdiscipline_raw_ids",
        "discipline_names",
        "discipline_raw_ids",
    ]:
        catalog[column] = catalog[column].fillna("")

    catalog.to_sql("course_catalog", conn, if_exists="replace", index=False)
    return {"course_catalog": len(catalog)}


def create_indexes_and_views(conn: sqlite3.Connection) -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_entities_id ON entities(id)",
        "CREATE INDEX IF NOT EXISTS idx_entities_type_raw ON entities(entity_type, raw_id)",
        "CREATE INDEX IF NOT EXISTS idx_resources_type ON resources(resource_type)",
        "CREATE INDEX IF NOT EXISTS idx_resources_title ON resources(title)",
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id, relation)",
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id, relation)",
        "CREATE INDEX IF NOT EXISTS idx_resource_edges_source ON resource_edges(source_id, relation)",
        "CREATE INDEX IF NOT EXISTS idx_resource_edges_target ON resource_edges(target_id, relation)",
        "CREATE INDEX IF NOT EXISTS idx_course_catalog_title ON course_catalog(title)",
        "CREATE INDEX IF NOT EXISTS idx_course_catalog_disciplines ON course_catalog(discipline_names, subdiscipline_names)",
        "CREATE INDEX IF NOT EXISTS idx_challenge_interactions_user ON challenge_interactions(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_challenge_interactions_challenge ON challenge_interactions(challenge_id)",
        "CREATE INDEX IF NOT EXISTS idx_challenge_interactions_user_challenge ON challenge_interactions(user_id, challenge_id)",
        "CREATE INDEX IF NOT EXISTS idx_discussions_challenge ON discussions(challenge_id)",
        "CREATE VIEW IF NOT EXISTS v_course_knowledge_points AS "
        "SELECT DISTINCT ce.source_id AS course_id, e.id AS topic_id, e.title AS topic_name "
        "FROM graph_edges ce "
        "JOIN graph_edges ge_ex ON ge_ex.source_id = ce.target_id AND ge_ex.relation = 'has_challenge' "
        "JOIN graph_edges ge ON ge.source_id = ge_ex.target_id AND ge.relation = 'has_topic' "
        "JOIN entities e ON e.id = ge.target_id "
        "WHERE ce.relation = 'has_exercise'",
        "CREATE VIEW IF NOT EXISTS v_user_challenge_performance AS "
        "SELECT user_id, challenge_id, COUNT(*) AS attempt_count, "
        "AVG(CAST(NULLIF(final_score, '') AS REAL)) AS avg_final_score, "
        "MAX(CAST(NULLIF(final_score, '') AS REAL)) AS best_final_score, "
        "MIN(open_time) AS first_open_time, MAX(end_time) AS last_end_time "
        "FROM challenge_interactions GROUP BY user_id, challenge_id",
    ]
    for statement in statements:
        conn.execute(statement)
    conn.commit()


def write_manifest(conn: sqlite3.Connection, db_path: Path, raw_root: Path, counts: Mapping[str, int], include_outputs: bool) -> None:
    manifest = {
        "source": SOURCE,
        "raw_root": str(raw_root),
        "database": str(db_path),
        "include_outputs": include_outputs,
        "counts": dict(counts),
        "schema_note": (
            "The database imports original MOOPer CSV columns into raw_* tables and builds normalized "
            "entities, resources, graph_edges, resource_edges, and course_catalog tables. It does not "
            "infer absent commercial fields such as price, grade level, availability, or external rating."
        ),
    }
    pd.DataFrame(
        [{"key": key, "value": json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value} for key, value in manifest.items()]
    ).to_sql("build_manifest", conn, if_exists="replace", index=False)

    manifest_path = db_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a complete SQLite database from the MOOPer CSV dataset.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help="Path to the extracted MOOPer directory.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Output SQLite database path.")
    parser.add_argument("--replace", action="store_true", help="Replace the output database if it already exists.")
    parser.add_argument("--chunksize", type=int, default=200000, help="Chunk size for large interaction CSV imports.")
    parser.add_argument("--include-outputs", action="store_true", help="Import interaction/outputs.csv. This file is very large.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    db_path = args.db_path.resolve()
    if not raw_root.exists():
        raise SystemExit("MOOPer raw root not found: {0}".format(raw_root))

    conn = connect(db_path, replace=args.replace)
    try:
        counts: Dict[str, int] = {}
        counts.update(import_raw_tables(conn, raw_root, include_outputs=args.include_outputs, chunksize=args.chunksize))
        counts.update(build_entities(conn, raw_root))
        counts.update(build_graph_edges(conn, raw_root))
        counts.update(build_course_catalog(conn, raw_root))
        create_indexes_and_views(conn)
        write_manifest(conn, db_path, raw_root, counts, include_outputs=args.include_outputs)
        conn.commit()
    finally:
        conn.close()

    print("Built MOOPer SQLite database: {0}".format(db_path))
    for name in sorted(counts):
        print("{0}: {1}".format(name, counts[name]))


if __name__ == "__main__":
    main()
