from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "MOOPer"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mooper"


ENTITY_DIR = Path("knowledgeGraph") / "entity"
RELATION_DIR = Path("knowledgeGraph") / "relation"
INTERACTION_DIR = Path("interaction")


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    value = value.strip()
    return value if value != "" else None


def compact_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: clean_value(value) for key, value in row.items() if clean_value(value) is not None}


def read_csv_rows(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield compact_row(row)


def load_lookup(path: Path, key_field: str) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in read_csv_rows(path):
        key = row.get(key_field)
        if key is not None:
            lookup[str(key)] = row
    return lookup


def load_one_to_many(path: Path, parent_field: str, child_field: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = defaultdict(list)
    for row in read_csv_rows(path):
        parent = row.get(parent_field)
        child = row.get(child_field)
        if parent is not None and child is not None:
            result[str(parent)].append(str(child))
    return dict(result)


def load_many_to_many(path: Path, fields: Sequence[str]) -> List[Dict[str, Any]]:
    return [row for row in read_csv_rows(path) if all(row.get(field) is not None for field in fields)]


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def resource_id(resource_type: str, raw_id: Any) -> str:
    return "mooper:{0}:{1}".format(resource_type, raw_id)


def edge(source_type: str, source_id: Any, relation: str, target_type: str, target_id: Any, **attrs: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "source_id": resource_id(source_type, source_id),
        "source_type": source_type,
        "relation": relation,
        "target_id": resource_id(target_type, target_id),
        "target_type": target_type,
        "source": "MOOPer",
    }
    for key, value in attrs.items():
        cleaned = clean_value(value)
        if cleaned is not None:
            payload[key] = cleaned
    return payload


def simple_resource(resource_type: str, raw_id_field: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    raw_id = row[raw_id_field]
    payload = {
        "id": resource_id(resource_type, raw_id),
        "source": "MOOPer",
        "resource_type": resource_type,
        "raw_id": str(raw_id),
        "raw_fields": dict(row),
    }
    name = row.get("name") or row.get("topic_name")
    if name is not None:
        payload["title"] = name
    description = row.get("description")
    if description is not None:
        payload["description"] = description
    return payload


def build_resources(raw_root: Path, output_dir: Path) -> Dict[str, int]:
    entity_root = raw_root / ENTITY_DIR

    specs = [
        ("course", "course_id", entity_root / "course.csv"),
        ("chapter", "chapter_id", entity_root / "chapter.csv"),
        ("exercise", "exercise_id", entity_root / "exercise.csv"),
        ("challenge", "challenge_id", entity_root / "challenge.csv"),
    ]

    counts: Dict[str, int] = {}
    for resource_type, raw_id_field, path in specs:
        rows = (simple_resource(resource_type, raw_id_field, row) for row in read_csv_rows(path))
        counts["{0}_resources".format(resource_type)] = write_jsonl(output_dir / "{0}s.jsonl".format(resource_type), rows)

    topic_rows = (
        {
            "id": resource_id("topic", row["topic_id"]),
            "source": "MOOPer",
            "resource_type": "topic",
            "raw_id": str(row["topic_id"]),
            "title": row.get("topic_name"),
            "raw_fields": dict(row),
        }
        for row in read_csv_rows(entity_root / "topics.csv")
        if row.get("topic_id") is not None
    )
    counts["topics"] = write_jsonl(output_dir / "knowledge_points.jsonl", topic_rows)
    return counts


def build_relations(raw_root: Path, output_dir: Path) -> Dict[str, int]:
    relation_root = raw_root / RELATION_DIR

    relation_specs = [
        ("course_chapter.csv", "course", "course_id", "has_chapter", "chapter", "chapter_id", ["position"]),
        ("exercise_course.csv", "course", "course_id", "has_exercise", "exercise", "exercise_id", ["position"]),
        ("exercise_challenge.csv", "exercise", "exercise_id", "has_challenge", "challenge", "challenge_id", ["position"]),
        ("challenge_topic.csv", "challenge", "challenge_id", "has_topic", "topic", "topic_id", ["created_at"]),
        ("discipline_course.csv", "course", "course_id", "belongs_to_subdiscipline", "subdiscipline", "sub_discipline_id", ["created_at"]),
        ("sub_discipline.csv", "subdiscipline", "sub_discipline_id", "belongs_to_discipline", "discipline", "discipline_id", []),
        ("course_teacher.csv", "course", "course_id", "has_teacher", "teacher", "teacher_id", []),
        ("chapter_teacher.csv", "chapter", "chapter_id", "has_teacher", "teacher", "teacher_id", []),
        ("challenge_teacher.csv", "challenge", "challenge_id", "has_teacher", "teacher", "teacher_id", []),
        ("exercise_teacher.csv", "exercise", "exercise_id", "has_teacher", "teacher", "teacher_id", []),
        # The file name says "exercise", but the released CSV header is
        # course_id,sub_discipline_id,created_at. Preserve the header meaning.
        ("discipline_exercise.csv", "course", "course_id", "belongs_to_subdiscipline", "subdiscipline", "sub_discipline_id", ["created_at"]),
    ]

    def iter_edges() -> Iterator[Dict[str, Any]]:
        for filename, source_type, source_field, relation, target_type, target_field, attr_fields in relation_specs:
            path = relation_root / filename
            if not path.exists():
                continue
            for row in read_csv_rows(path):
                if row.get(source_field) is None or row.get(target_field) is None:
                    continue
                attrs = {field: row.get(field) for field in attr_fields}
                yield edge(source_type, row[source_field], relation, target_type, row[target_field], **attrs)

    count = write_jsonl(output_dir / "resource_edges.jsonl", iter_edges())
    return {"resource_edges": count}


def build_course_catalog(raw_root: Path, output_dir: Path) -> Dict[str, int]:
    entity_root = raw_root / ENTITY_DIR
    relation_root = raw_root / RELATION_DIR

    courses = load_lookup(entity_root / "course.csv", "course_id")
    chapters = load_lookup(entity_root / "chapter.csv", "chapter_id")
    exercises = load_lookup(entity_root / "exercise.csv", "exercise_id")
    challenges = load_lookup(entity_root / "challenge.csv", "challenge_id")
    topics = load_lookup(entity_root / "topics.csv", "topic_id")
    subdisciplines = load_lookup(entity_root / "subdiscipline.csv", "sub_discipline_id")
    disciplines = load_lookup(entity_root / "discipline.csv", "discipline_id")

    course_chapters = load_many_to_many(relation_root / "course_chapter.csv", ["course_id", "chapter_id"])
    course_exercises = load_many_to_many(relation_root / "exercise_course.csv", ["course_id", "exercise_id"])
    exercise_challenges = load_one_to_many(relation_root / "exercise_challenge.csv", "exercise_id", "challenge_id")
    challenge_topics = load_one_to_many(relation_root / "challenge_topic.csv", "challenge_id", "topic_id")
    course_subdisciplines = load_one_to_many(relation_root / "discipline_course.csv", "course_id", "sub_discipline_id")
    sub_to_discipline = load_one_to_many(relation_root / "sub_discipline.csv", "sub_discipline_id", "discipline_id")

    chapters_by_course: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in course_chapters:
        chapter_id = str(row["chapter_id"])
        chapter = chapters.get(chapter_id)
        if not chapter:
            continue
        chapters_by_course[str(row["course_id"])].append(
            {
                "id": resource_id("chapter", chapter_id),
                "raw_id": chapter_id,
                "title": chapter.get("name"),
                "position": row.get("position"),
            }
        )

    exercises_by_course: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    topics_by_course: Dict[str, Set[str]] = defaultdict(set)
    challenges_by_course: Dict[str, Set[str]] = defaultdict(set)

    for row in course_exercises:
        course_id = str(row["course_id"])
        exercise_id = str(row["exercise_id"])
        exercise_row = exercises.get(exercise_id)
        if exercise_row:
            exercises_by_course[course_id].append(
                {
                    "id": resource_id("exercise", exercise_id),
                    "raw_id": exercise_id,
                    "title": exercise_row.get("name"),
                    "position": row.get("position"),
                    "visits": exercise_row.get("visits"),
                    "status": exercise_row.get("status"),
                    "created_at": exercise_row.get("created_at"),
                    "publish_time": exercise_row.get("publish_time"),
                }
            )
        for challenge_id in exercise_challenges.get(exercise_id, []):
            challenges_by_course[course_id].add(challenge_id)
            for topic_id in challenge_topics.get(challenge_id, []):
                topics_by_course[course_id].add(topic_id)

    def make_course(row: Mapping[str, Any]) -> Dict[str, Any]:
        course_id = str(row["course_id"])
        sub_ids = course_subdisciplines.get(course_id, [])
        discipline_ids: Set[str] = set()
        for sub_id in sub_ids:
            discipline_ids.update(sub_to_discipline.get(sub_id, []))

        topic_items = []
        for topic_id in sorted(topics_by_course.get(course_id, set()), key=lambda item: int(float(item)) if item.replace(".", "", 1).isdigit() else item):
            topic = topics.get(topic_id)
            if topic:
                topic_items.append({"id": resource_id("topic", topic_id), "raw_id": topic_id, "title": topic.get("topic_name")})

        return {
            "id": resource_id("course", course_id),
            "source": "MOOPer",
            "resource_type": "course",
            "raw_id": course_id,
            "title": row.get("name"),
            "description": row.get("description"),
            "visits": row.get("visits"),
            "created_at": row.get("created_at"),
            "learning_notes": row.get("learning_notes"),
            "publish_time": row.get("publish_time"),
            "subdisciplines": [
                {
                    "id": resource_id("subdiscipline", sub_id),
                    "raw_id": sub_id,
                    "title": subdisciplines.get(sub_id, {}).get("name"),
                }
                for sub_id in sub_ids
            ],
            "disciplines": [
                {
                    "id": resource_id("discipline", discipline_id),
                    "raw_id": discipline_id,
                    "title": disciplines.get(discipline_id, {}).get("name"),
                }
                for discipline_id in sorted(discipline_ids)
            ],
            "chapters": chapters_by_course.get(course_id, []),
            "exercises": exercises_by_course.get(course_id, []),
            "knowledge_points": topic_items,
            "challenge_ids": [resource_id("challenge", challenge_id) for challenge_id in sorted(challenges_by_course.get(course_id, set()))],
            "raw_fields": dict(row),
        }

    rows = (make_course(row) for row in courses.values())
    count = write_jsonl(output_dir / "course_catalog.jsonl", rows)
    return {"course_catalog": count}


def build_interactions(raw_root: Path, output_dir: Path, include_outputs: bool, max_outputs: Optional[int]) -> Dict[str, int]:
    interaction_root = raw_root / INTERACTION_DIR

    def challenge_interactions() -> Iterator[Dict[str, Any]]:
        for row in read_csv_rows(interaction_root / "challenge_interaction.csv"):
            payload = {
                "id": "mooper:challenge_interaction:{0}".format(row.get("c_interaction_id")),
                "source": "MOOPer",
                "interaction_type": "challenge_interaction",
                "user_id": row.get("user_id"),
                "resource_id": resource_id("challenge", row.get("challenge_id")),
                "raw_fields": row,
            }
            yield payload

    counts = {
        "challenge_interactions": write_jsonl(output_dir / "challenge_interactions.jsonl", challenge_interactions())
    }

    def discussions() -> Iterator[Dict[str, Any]]:
        for row in read_csv_rows(interaction_root / "discussions.csv"):
            challenge_id = row.get("challenge_id")
            payload = {
                "id": "mooper:discussion:{0}".format(row.get("discuss_id")),
                "source": "MOOPer",
                "interaction_type": "discussion",
                "user_id": row.get("user_id"),
                "resource_id": resource_id("challenge", challenge_id) if challenge_id is not None else None,
                "raw_fields": row,
            }
            yield compact_row(payload)

    counts["discussions"] = write_jsonl(output_dir / "discussions.jsonl", discussions())

    if include_outputs:
        def outputs() -> Iterator[Dict[str, Any]]:
            emitted = 0
            for row in read_csv_rows(interaction_root / "outputs.csv"):
                if max_outputs is not None and emitted >= max_outputs:
                    break
                emitted += 1
                yield {
                    "id": "mooper:output:{0}".format(row.get("id")),
                    "source": "MOOPer",
                    "interaction_type": "output",
                    "challenge_interaction_id": row.get("c_interaction_id"),
                    "raw_fields": row,
                }

        counts["outputs"] = write_jsonl(output_dir / "outputs.jsonl", outputs())

    return counts


def build_manifest(raw_root: Path, output_dir: Path, counts: Mapping[str, int]) -> None:
    manifest = {
        "source": "MOOPer",
        "raw_root": str(raw_root),
        "outputs": dict(counts),
        "schema_note": (
            "The processed files preserve MOOPer fields and only add stable ids, "
            "resource types, relation names, and source metadata. Fields absent from "
            "MOOPer, such as price, grade level, and commercial rating, are not inferred."
        ),
    }
    write_json(output_dir / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MOOPer CSV files into recommendation-ready JSONL files.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help="Path to the extracted MOOPer directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for processed JSONL files.")
    parser.add_argument("--skip-interactions", action="store_true", help="Skip challenge interactions and discussions.")
    parser.add_argument("--include-outputs", action="store_true", help="Also convert interaction/outputs.csv. This file is very large.")
    parser.add_argument("--max-outputs", type=int, default=None, help="Optional cap for converted output rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()

    if not raw_root.exists():
        raise SystemExit("MOOPer raw root not found: {0}".format(raw_root))

    ensure_output_dir(output_dir)

    counts: Dict[str, int] = {}
    counts.update(build_resources(raw_root, output_dir))
    counts.update(build_relations(raw_root, output_dir))
    counts.update(build_course_catalog(raw_root, output_dir))
    if not args.skip_interactions:
        counts.update(build_interactions(raw_root, output_dir, args.include_outputs, args.max_outputs))
    build_manifest(raw_root, output_dir, counts)

    print("Processed MOOPer data into {0}".format(output_dir))
    for name in sorted(counts):
        print("{0}: {1}".format(name, counts[name]))


if __name__ == "__main__":
    main()
