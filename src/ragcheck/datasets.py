"""Dataset loading: a corpus JSONL plus a questions JSONL.

``corpus.jsonl``::

    {"id": "doc-1", "text": "...", "title": "optional", "metadata": {}}

``questions.jsonl``::

    {"id": "q1", "question": "...", "answer": "...", "relevant_doc_ids": ["doc-1"]}

(``ground_truth`` is accepted as an alias of ``answer``.)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ragcheck.types import Document, EvalCase

__all__ = ["Dataset", "DatasetError", "load_dataset", "load_jsonl_dataset", "load_sample"]

BUILTIN_PREFIX = "builtin:"


class DatasetError(ValueError):
    """Raised for malformed dataset files, with file and line information."""


@dataclass
class Dataset:
    documents: list[Document]
    cases: list[EvalCase]
    name: str = "dataset"
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cases)

    def subset(self, n: int) -> Dataset:
        return Dataset(self.documents, self.cases[:n], self.name, self.source, self.metadata)


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(obj, dict):
                raise DatasetError(f"{path}:{lineno}: expected a JSON object")
            yield lineno, obj


def load_jsonl_dataset(
    corpus_path: str | Path,
    questions_path: str | Path,
    name: str | None = None,
    strict: bool = True,
) -> Dataset:
    """Load and validate a dataset. With ``strict``, unknown doc ids are an error."""
    corpus_path, questions_path = Path(corpus_path), Path(questions_path)
    documents: list[Document] = []
    for lineno, obj in _iter_jsonl(corpus_path):
        try:
            documents.append(Document.model_validate(obj))
        except ValidationError as exc:
            raise DatasetError(f"{corpus_path}:{lineno}: {exc.errors()[0]['msg']}") from exc
    ids = [d.id for d in documents]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise DatasetError(f"{corpus_path}: duplicate document ids {dupes}")
    known = set(ids)
    cases: list[EvalCase] = []
    for lineno, raw in _iter_jsonl(questions_path):
        obj = dict(raw)
        if "answer" in obj:
            obj["ground_truth"] = obj.pop("answer")
        try:
            case = EvalCase.model_validate(obj)
        except ValidationError as exc:
            raise DatasetError(f"{questions_path}:{lineno}: {exc.errors()[0]['msg']}") from exc
        unknown = [d for d in case.relevant_doc_ids if d not in known]
        if unknown and strict:
            raise DatasetError(f"{questions_path}:{lineno}: unknown relevant_doc_ids {unknown}")
        cases.append(case)
    return Dataset(
        documents=documents,
        cases=cases,
        name=name or corpus_path.parent.name,
        source=str(corpus_path.parent),
    )


def load_sample() -> Dataset:
    """The bundled synthetic corpus: docs for *Larkspur*, a fictional task-queue product."""
    root = resources.files("ragcheck") / "data" / "sample"
    with (
        resources.as_file(root / "corpus.jsonl") as corpus,
        resources.as_file(root / "questions.jsonl") as questions,
    ):
        ds = load_jsonl_dataset(corpus, questions, name="larkspur-sample")
    ds.source = BUILTIN_PREFIX + "sample"
    return ds


def load_dataset(path: str | Path) -> Dataset:
    """Load ``builtin:sample`` or a directory containing corpus.jsonl and questions.jsonl."""
    spec = str(path)
    if spec.startswith(BUILTIN_PREFIX):
        if spec != BUILTIN_PREFIX + "sample":
            raise DatasetError(f"unknown builtin dataset {spec!r}; available: builtin:sample")
        return load_sample()
    p = Path(path)
    if not p.is_dir():
        raise DatasetError(f"dataset path {p} is not a directory")
    corpus, questions = p / "corpus.jsonl", p / "questions.jsonl"
    for f in (corpus, questions):
        if not f.exists():
            raise DatasetError(f"missing {f}")
    return load_jsonl_dataset(corpus, questions, name=p.name)
