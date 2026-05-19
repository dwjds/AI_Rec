from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.db.sqlite import connect_resource_db, row_to_dict, rows_to_dicts


LEARNING_RESOURCE_TYPES = {"course", "chapter", "exercise", "challenge", "topic", "discipline", "subdiscipline"}


class ResourceStore:
    """Read-only access layer for the MOOPer resource database."""

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        with connect_resource_db() as conn:
            row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
            return row_to_dict(row)

    def get_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        with connect_resource_db() as conn:
            row = conn.execute("SELECT * FROM resources WHERE id = ?", (resource_id,)).fetchone()
            return row_to_dict(row)

    def get_course(self, course_id: str) -> Optional[Dict[str, Any]]:
        normalized_id = self._normalize_id("course", course_id)
        with connect_resource_db() as conn:
            row = conn.execute("SELECT * FROM course_catalog WHERE id = ? OR raw_id = ?", (normalized_id, course_id)).fetchone()
            return self._normalize_course(row_to_dict(row))

    def list_courses(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        with connect_resource_db() as conn:
            rows = conn.execute(
                """
                SELECT * FROM course_catalog
                ORDER BY CAST(NULLIF(visits, '') AS INTEGER) DESC, title ASC
                LIMIT ? OFFSET ?
                """,
                (self._limit(limit), max(0, int(offset))),
            ).fetchall()
            return [self._normalize_course(item) for item in rows_to_dicts(rows)]

    def search_courses(
        self,
        query: str,
        limit: int = 10,
        discipline: Optional[str] = None,
        subdiscipline: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        terms = self._terms(query)
        where_parts = []
        params: List[Any] = []

        for term in terms:
            like = "%{0}%".format(term)
            where_parts.append(
                """
                (
                    title LIKE ?
                    OR description LIKE ?
                    OR learning_notes LIKE ?
                    OR knowledge_point_names LIKE ?
                    OR subdiscipline_names LIKE ?
                    OR discipline_names LIKE ?
                )
                """
            )
            params.extend([like, like, like, like, like, like])

        if discipline:
            where_parts.append("discipline_names LIKE ?")
            params.append("%{0}%".format(discipline))
        if subdiscipline:
            where_parts.append("subdiscipline_names LIKE ?")
            params.append("%{0}%".format(subdiscipline))

        where_sql = " AND ".join(where_parts) if where_parts else "1 = 1"
        params.append(self._limit(limit))

        with connect_resource_db() as conn:
            rows = conn.execute(
                """
                SELECT *,
                    (
                        CASE WHEN title LIKE ? THEN 8 ELSE 0 END
                    ) AS title_boost
                FROM course_catalog
                WHERE {where_sql}
                ORDER BY title_boost DESC,
                    CAST(NULLIF(visits, '') AS INTEGER) DESC,
                    CAST(NULLIF(challenge_count, '') AS INTEGER) DESC,
                    title ASC
                LIMIT ?
                """.format(where_sql=where_sql),
                ["%{0}%".format(query.strip())] + params,
            ).fetchall()
            return [self._normalize_course(item) for item in rows_to_dicts(rows)]

    def search_resources(
        self,
        query: str,
        resource_types: Optional[Sequence[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        terms = self._terms(query)
        selected_types = [item for item in (resource_types or []) if item in LEARNING_RESOURCE_TYPES]
        params: List[Any] = []
        where_parts = []

        for term in terms:
            like = "%{0}%".format(term)
            where_parts.append("(title LIKE ? OR description LIKE ? OR learning_notes LIKE ?)")
            params.extend([like, like, like])

        if selected_types:
            placeholders = ", ".join(["?"] * len(selected_types))
            where_parts.append("resource_type IN ({0})".format(placeholders))
            params.extend(selected_types)

        where_sql = " AND ".join(where_parts) if where_parts else "1 = 1"
        params.append(self._limit(limit))

        with connect_resource_db() as conn:
            rows = conn.execute(
                """
                SELECT * FROM resources
                WHERE {where_sql}
                ORDER BY CAST(NULLIF(visits, '') AS INTEGER) DESC, title ASC
                LIMIT ?
                """.format(where_sql=where_sql),
                params,
            ).fetchall()
            return rows_to_dicts(rows)

    def search_knowledge_points(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        terms = self._terms(query)
        params: List[Any] = []
        where_parts = ["entity_type = 'topic'"]
        for term in terms:
            where_parts.append("title LIKE ?")
            params.append("%{0}%".format(term))
        params.append(self._limit(limit))

        with connect_resource_db() as conn:
            rows = conn.execute(
                """
                SELECT * FROM entities
                WHERE {where_sql}
                ORDER BY title ASC
                LIMIT ?
                """.format(where_sql=" AND ".join(where_parts)),
                params,
            ).fetchall()
            return rows_to_dicts(rows)

    def get_course_knowledge_points(self, course_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        normalized_id = self._normalize_id("course", course_id)
        with connect_resource_db() as conn:
            rows = conn.execute(
                """
                SELECT topic_id AS id, topic_name AS title
                FROM v_course_knowledge_points
                WHERE course_id = ?
                ORDER BY topic_name ASC
                LIMIT ?
                """,
                (normalized_id, self._limit(limit, upper=500)),
            ).fetchall()
            return rows_to_dicts(rows)

    def list_course_chapters(self, course_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return self._list_targets(
            source_id=self._normalize_id("course", course_id),
            relation="has_chapter",
            limit=limit,
        )

    def list_course_exercises(self, course_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return self._list_targets(
            source_id=self._normalize_id("course", course_id),
            relation="has_exercise",
            limit=limit,
        )

    def list_exercise_challenges(self, exercise_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return self._list_targets(
            source_id=self._normalize_id("exercise", exercise_id),
            relation="has_challenge",
            limit=limit,
        )

    def get_course_detail(self, course_id: str) -> Optional[Dict[str, Any]]:
        course = self.get_course(course_id)
        if course is None:
            return None
        course["chapters"] = self.list_course_chapters(course["id"], limit=200)
        course["exercises"] = self.list_course_exercises(course["id"], limit=200)
        course["knowledge_points"] = self.get_course_knowledge_points(course["id"], limit=200)
        return course

    def get_neighbors(
        self,
        resource_id: str,
        relation: Optional[str] = None,
        direction: str = "out",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        if direction not in {"out", "in"}:
            raise ValueError("direction must be 'out' or 'in'.")

        id_column = "source_id" if direction == "out" else "target_id"
        target_column = "target_id" if direction == "out" else "source_id"
        where_parts = ["e.{0} = ?".format(id_column)]
        params: List[Any] = [resource_id]
        if relation:
            where_parts.append("e.relation = ?")
            params.append(relation)
        params.append(self._limit(limit))

        with connect_resource_db() as conn:
            rows = conn.execute(
                """
                SELECT e.relation, e.position, r.*
                FROM resource_edges e
                JOIN resources r ON r.id = e.{target_column}
                WHERE {where_sql}
                ORDER BY CAST(NULLIF(e.position, '') AS INTEGER) ASC, r.title ASC
                LIMIT ?
                """.format(target_column=target_column, where_sql=" AND ".join(where_parts)),
                params,
            ).fetchall()
            return rows_to_dicts(rows)

    def _list_targets(self, source_id: str, relation: str, limit: int) -> List[Dict[str, Any]]:
        with connect_resource_db() as conn:
            rows = conn.execute(
                """
                SELECT e.position, r.*
                FROM resource_edges e
                JOIN resources r ON r.id = e.target_id
                WHERE e.source_id = ? AND e.relation = ?
                ORDER BY CAST(NULLIF(e.position, '') AS INTEGER) ASC, r.title ASC
                LIMIT ?
                """,
                (source_id, relation, self._limit(limit, upper=500)),
            ).fetchall()
            return rows_to_dicts(rows)

    def _terms(self, query: str) -> List[str]:
        terms = [item.strip() for item in str(query or "").replace("，", " ").replace(",", " ").split()]
        return [item for item in terms if item][:8]

    def _limit(self, value: int, lower: int = 1, upper: int = 100) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = 10
        return max(lower, min(upper, number))

    def _normalize_id(self, entity_type: str, value: str) -> str:
        text = str(value)
        prefix = "mooper:{0}:".format(entity_type)
        return text if text.startswith(prefix) else prefix + text

    def _split_pipe(self, value: Any) -> List[str]:
        if not value:
            return []
        return [item for item in str(value).split("|") if item]

    def _safe_int(self, value: Any) -> int:
        try:
            return int(str(value or "0"))
        except ValueError:
            return 0

    def _normalize_course(self, course: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if course is None:
            return None
        normalized = dict(course)
        normalized["visits"] = self._safe_int(normalized.get("visits"))
        normalized["chapter_count"] = self._safe_int(normalized.get("chapter_count"))
        normalized["exercise_count"] = self._safe_int(normalized.get("exercise_count"))
        normalized["challenge_count"] = self._safe_int(normalized.get("challenge_count"))
        normalized["knowledge_points"] = self._split_pipe(normalized.get("knowledge_point_names"))
        normalized["knowledge_point_ids"] = self._split_pipe(normalized.get("knowledge_point_raw_ids"))
        normalized["disciplines"] = self._split_pipe(normalized.get("discipline_names"))
        normalized["subdisciplines"] = self._split_pipe(normalized.get("subdiscipline_names"))
        return normalized
