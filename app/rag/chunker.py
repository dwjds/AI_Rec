from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from app.core.config import settings
from app.db.sqlite import connect_resource_db, rows_to_dicts


@dataclass
class RagChunk:
    id: str
    chunk_type: str
    source: str
    source_resource_id: str
    source_resource_type: str
    title: str
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MooperChunker:
    """Build RAG chunks from natural MOOPer resource units."""

    def iter_chunks(self, include_types: Optional[Iterable[str]] = None) -> Iterator[RagChunk]:
        selected = set(include_types or ["course", "chapter", "exercise", "knowledge_point"])
        if "course" in selected:
            yield from self.iter_course_chunks()
        if "chapter" in selected:
            yield from self.iter_chapter_chunks()
        if "exercise" in selected:
            yield from self.iter_exercise_chunks()
        if "knowledge_point" in selected:
            yield from self.iter_knowledge_point_chunks()

    def iter_course_chunks(self) -> Iterator[RagChunk]:
        with connect_resource_db() as conn:
            rows = rows_to_dicts(conn.execute("SELECT * FROM course_catalog ORDER BY raw_id").fetchall())
        for course in rows:
            disciplines = self._split_pipe(course.get("discipline_names"))
            subdisciplines = self._split_pipe(course.get("subdiscipline_names"))
            knowledge_points = self._split_pipe(course.get("knowledge_point_names"))
            metadata = self._metadata(
                {
                    "course_id": course.get("id"),
                    "raw_id": course.get("raw_id"),
                    "resource_type": "course",
                    "title": course.get("title"),
                    "discipline_names": "|".join(disciplines),
                    "subdiscipline_names": "|".join(subdisciplines),
                    "knowledge_point_names": "|".join(knowledge_points[:80]),
                    "chapter_count": self._safe_int(course.get("chapter_count")),
                    "exercise_count": self._safe_int(course.get("exercise_count")),
                    "challenge_count": self._safe_int(course.get("challenge_count")),
                    "visits": self._safe_int(course.get("visits")),
                    "publish_time": course.get("publish_time"),
                    "created_at": course.get("created_at"),
                }
            )
            content = self._join_lines(
                [
                    ("课程标题", course.get("title")),
                    ("课程描述", course.get("description")),
                    ("学习说明", course.get("learning_notes")),
                    ("学科", "、".join(disciplines)),
                    ("子学科", "、".join(subdisciplines)),
                    ("覆盖知识点", "、".join(knowledge_points[:80])),
                    ("章节数量", course.get("chapter_count")),
                    ("练习数量", course.get("exercise_count")),
                    ("挑战数量", course.get("challenge_count")),
                    ("访问量", course.get("visits")),
                ]
            )
            yield RagChunk(
                id="chunk:course:{0}".format(course.get("raw_id")),
                chunk_type="course",
                source="MOOPer",
                source_resource_id=str(course.get("id")),
                source_resource_type="course",
                title=str(course.get("title") or ""),
                content=content,
                metadata=metadata,
            )

    def iter_chapter_chunks(self) -> Iterator[RagChunk]:
        with connect_resource_db() as conn:
            rows = rows_to_dicts(
                conn.execute(
                    """
                    SELECT
                        chapter.id AS chapter_id,
                        chapter.raw_id AS chapter_raw_id,
                        chapter.title AS chapter_title,
                        chapter.description AS chapter_description,
                        edge.position AS chapter_position,
                        course.id AS course_id,
                        course.raw_id AS course_raw_id,
                        course.title AS course_title,
                        course.knowledge_point_names AS course_knowledge_points
                    FROM resource_edges edge
                    JOIN resources chapter ON chapter.id = edge.target_id
                    JOIN course_catalog course ON course.id = edge.source_id
                    WHERE edge.relation = 'has_chapter'
                    ORDER BY course.raw_id, CAST(NULLIF(edge.position, '') AS INTEGER), chapter.raw_id
                    """
                ).fetchall()
            )
        for row in rows:
            knowledge_points = self._split_pipe(row.get("course_knowledge_points"))
            metadata = self._metadata(
                {
                    "chapter_id": row.get("chapter_id"),
                    "raw_id": row.get("chapter_raw_id"),
                    "resource_type": "chapter",
                    "title": row.get("chapter_title"),
                    "course_id": row.get("course_id"),
                    "course_raw_id": row.get("course_raw_id"),
                    "course_title": row.get("course_title"),
                    "chapter_position": self._safe_int(row.get("chapter_position")),
                    "knowledge_point_names": "|".join(knowledge_points[:60]),
                    "knowledge_point_scope": "inherited_from_course",
                }
            )
            content = self._join_lines(
                [
                    ("章节标题", row.get("chapter_title")),
                    ("所属课程", row.get("course_title")),
                    ("章节简介", row.get("chapter_description")),
                    ("关联知识点", "、".join(knowledge_points[:60])),
                    ("章节顺序", row.get("chapter_position")),
                ]
            )
            yield RagChunk(
                id="chunk:chapter:{0}".format(row.get("chapter_raw_id")),
                chunk_type="chapter",
                source="MOOPer",
                source_resource_id=str(row.get("chapter_id")),
                source_resource_type="chapter",
                title=str(row.get("chapter_title") or ""),
                content=content,
                metadata=metadata,
            )

    def iter_exercise_chunks(self) -> Iterator[RagChunk]:
        with connect_resource_db() as conn:
            rows = rows_to_dicts(
                conn.execute(
                    """
                    SELECT
                        exercise.id AS exercise_id,
                        exercise.raw_id AS exercise_raw_id,
                        exercise.title AS exercise_title,
                        exercise.description AS exercise_description,
                        exercise.visits AS exercise_visits,
                        exercise.status AS exercise_status,
                        edge.position AS exercise_position,
                        course.id AS course_id,
                        course.raw_id AS course_raw_id,
                        course.title AS course_title,
                        GROUP_CONCAT(DISTINCT challenge.title) AS challenge_titles,
                        GROUP_CONCAT(DISTINCT NULLIF(challenge.difficulty, '')) AS challenge_difficulties,
                        GROUP_CONCAT(DISTINCT topic.title) AS topic_names
                    FROM resource_edges edge
                    JOIN resources exercise ON exercise.id = edge.target_id
                    JOIN course_catalog course ON course.id = edge.source_id
                    LEFT JOIN resource_edges ex_ch ON ex_ch.source_id = exercise.id AND ex_ch.relation = 'has_challenge'
                    LEFT JOIN resources challenge ON challenge.id = ex_ch.target_id
                    LEFT JOIN resource_edges ch_topic ON ch_topic.source_id = challenge.id AND ch_topic.relation = 'has_topic'
                    LEFT JOIN resources topic ON topic.id = ch_topic.target_id
                    WHERE edge.relation = 'has_exercise'
                    GROUP BY exercise.id, course.id, edge.position
                    ORDER BY course.raw_id, CAST(NULLIF(edge.position, '') AS INTEGER), exercise.raw_id
                    """
                ).fetchall()
            )
        for row in rows:
            topics = self._split_csv(row.get("topic_names"))
            challenge_titles = self._split_csv(row.get("challenge_titles"))
            difficulties = self._split_csv(row.get("challenge_difficulties"))
            metadata = self._metadata(
                {
                    "exercise_id": row.get("exercise_id"),
                    "raw_id": row.get("exercise_raw_id"),
                    "resource_type": "exercise",
                    "title": row.get("exercise_title"),
                    "course_id": row.get("course_id"),
                    "course_raw_id": row.get("course_raw_id"),
                    "course_title": row.get("course_title"),
                    "exercise_position": self._safe_int(row.get("exercise_position")),
                    "knowledge_point_names": "|".join(topics[:80]),
                    "difficulty": "|".join(difficulties),
                    "visits": self._safe_int(row.get("exercise_visits")),
                    "status": row.get("exercise_status"),
                }
            )
            content = self._join_lines(
                [
                    ("练习标题", row.get("exercise_title")),
                    ("所属课程", row.get("course_title")),
                    ("训练目标", self._summarize_targets(row.get("exercise_description"), challenge_titles)),
                    ("关联知识点", "、".join(topics[:80])),
                    ("难度", "、".join(difficulties)),
                    ("访问量", row.get("exercise_visits")),
                ]
            )
            yield RagChunk(
                id="chunk:exercise:{0}:course:{1}:pos:{2}".format(
                    row.get("exercise_raw_id"),
                    row.get("course_raw_id"),
                    self._safe_int(row.get("exercise_position")),
                ),
                chunk_type="exercise",
                source="MOOPer",
                source_resource_id=str(row.get("exercise_id")),
                source_resource_type="exercise",
                title=str(row.get("exercise_title") or ""),
                content=content,
                metadata=metadata,
            )

    def iter_knowledge_point_chunks(self) -> Iterator[RagChunk]:
        with connect_resource_db() as conn:
            rows = rows_to_dicts(
                conn.execute(
                    """
                    SELECT
                        topic.id AS topic_id,
                        topic.raw_id AS topic_raw_id,
                        topic.title AS topic_title,
                        COUNT(DISTINCT course.course_id) AS related_course_count,
                        GROUP_CONCAT(DISTINCT catalog.title) AS related_course_titles,
                        GROUP_CONCAT(DISTINCT catalog.discipline_names) AS discipline_names,
                        GROUP_CONCAT(DISTINCT catalog.subdiscipline_names) AS subdiscipline_names,
                        COUNT(DISTINCT ch_topic.source_id) AS related_challenge_count
                    FROM resources topic
                    LEFT JOIN v_course_knowledge_points course ON course.topic_id = topic.id
                    LEFT JOIN course_catalog catalog ON catalog.id = course.course_id
                    LEFT JOIN resource_edges ch_topic ON ch_topic.target_id = topic.id AND ch_topic.relation = 'has_topic'
                    WHERE topic.resource_type = 'topic'
                    GROUP BY topic.id
                    ORDER BY topic.raw_id
                    """
                ).fetchall()
            )
        for row in rows:
            course_titles = self._split_csv(row.get("related_course_titles"))
            disciplines = self._split_pipe(",".join(self._split_csv(row.get("discipline_names"))))
            subdisciplines = self._split_pipe(",".join(self._split_csv(row.get("subdiscipline_names"))))
            metadata = self._metadata(
                {
                    "topic_id": row.get("topic_id"),
                    "raw_id": row.get("topic_raw_id"),
                    "resource_type": "knowledge_point",
                    "title": row.get("topic_title"),
                    "related_course_count": self._safe_int(row.get("related_course_count")),
                    "related_challenge_count": self._safe_int(row.get("related_challenge_count")),
                    "related_course_titles": "|".join(course_titles[:50]),
                    "discipline_names": "|".join(disciplines[:20]),
                    "subdiscipline_names": "|".join(subdisciplines[:30]),
                }
            )
            content = self._join_lines(
                [
                    ("知识点名称", row.get("topic_title")),
                    ("相关课程", "、".join(course_titles[:50])),
                    ("学科", "、".join(disciplines[:20])),
                    ("子学科", "、".join(subdisciplines[:30])),
                    ("相关课程数量", row.get("related_course_count")),
                    ("相关任务数量", row.get("related_challenge_count")),
                ]
            )
            yield RagChunk(
                id="chunk:knowledge_point:{0}".format(row.get("topic_raw_id")),
                chunk_type="knowledge_point",
                source="MOOPer",
                source_resource_id=str(row.get("topic_id")),
                source_resource_type="knowledge_point",
                title=str(row.get("topic_title") or ""),
                content=content,
                metadata=metadata,
            )

    def write_jsonl(self, output_path: Path | None = None, include_types: Optional[Iterable[str]] = None) -> Dict[str, int]:
        target = output_path or settings.chunks_path
        target.parent.mkdir(parents=True, exist_ok=True)
        id_map_path = settings.chunk_id_map_path
        id_map_path.parent.mkdir(parents=True, exist_ok=True)

        counts: Dict[str, int] = {}
        chunk_id_map: Dict[str, Dict[str, str]] = {}
        with target.open("w", encoding="utf-8") as fp:
            for chunk in self.iter_chunks(include_types=include_types):
                data = chunk.to_dict()
                fp.write(json.dumps(data, ensure_ascii=False) + "\n")
                counts[chunk.chunk_type] = counts.get(chunk.chunk_type, 0) + 1
                chunk_id_map[chunk.id] = {
                    "source_resource_id": chunk.source_resource_id,
                    "source_resource_type": chunk.source_resource_type,
                    "title": chunk.title,
                }
        id_map_path.write_text(json.dumps(chunk_id_map, ensure_ascii=False, indent=2), encoding="utf-8")
        counts["total"] = sum(counts.values())
        return counts

    def _join_lines(self, pairs: Iterable[tuple[str, Any]]) -> str:
        lines = []
        for label, value in pairs:
            text = self._clean_text(value)
            if text:
                lines.append("{0}：{1}".format(label, text))
        return "\n".join(lines)

    def _metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in data.items():
            if value is None:
                cleaned[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned

    def _clean_text(self, value: Any, max_length: int = 1600) -> str:
        text = str(value or "").replace("\ufeff", "").strip()
        text = " ".join(text.split())
        return text[:max_length]

    def _split_pipe(self, value: Any) -> List[str]:
        if not value:
            return []
        parts: List[str] = []
        for item in str(value).replace(",", "|").split("|"):
            text = item.strip()
            if text and text not in parts:
                parts.append(text)
        return parts

    def _split_csv(self, value: Any) -> List[str]:
        if not value:
            return []
        parts: List[str] = []
        for item in str(value).split(","):
            text = item.strip()
            if text and text not in parts:
                parts.append(text)
        return parts

    def _safe_int(self, value: Any) -> int:
        try:
            return int(float(str(value or "0")))
        except ValueError:
            return 0

    def _summarize_targets(self, description: Any, challenge_titles: List[str]) -> str:
        text = self._clean_text(description, max_length=500)
        if text:
            return text
        return "、".join(challenge_titles[:20])
