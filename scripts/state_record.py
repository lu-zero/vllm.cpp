#!/usr/bin/env python3
"""Parsing and validation primitives for the structured state record."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


MANIFEST_HEADER = ("schema_version", "shard_id", "index_path", "event_prefix")
EVENT_HEADER = (
    "event_id",
    "occurred_at",
    "kind",
    "subject_ids",
    "phase",
    "outcome",
    "commit",
    "pr",
    "spec",
    "evidence_path",
    "supersedes",
    "summary",
    "next_action",
)

MANIFEST_MAX_BYTES = 64 * 1024
SHARD_MAX_BYTES = 256 * 1024
SHARD_MAX_EVENTS = 512
EVENT_MAX_BYTES = 32 * 1024
SUMMARY_MAX_CHARS = 200
NEXT_ACTION_MAX_CHARS = 240
SHARD_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{3}$")
NEW_EVENT_ID_RE = re.compile(r"^STATE-(\d{8}T\d{6})-\d{3}$")
LEGACY_EVENT_ID_RE = re.compile(r"^STATE-LEGACY-\d{6}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PR_RE = re.compile(r"^pr:[1-9]\d*$")
SUBJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
KINDS = frozenset({"legacy_import", "checkpoint", "decision", "handoff", "correction"})
PHASES = frozenset(
    {"spec", "implementation", "review", "verification", "integration", "handoff", "operations"}
)
OUTCOMES = frozenset(
    {"started", "checkpoint", "passed", "failed", "blocked", "landed", "closed", "superseded"}
)
REQUIRED_SECTIONS = ("Context", "Outcome", "Evidence", "Next action")
LEGACY_BEGIN = b"<!-- legacy-payload:begin -->\n"
LEGACY_END = b"<!-- legacy-payload:end -->\n"


@dataclass(frozen=True)
class Shard:
    schema_version: str
    shard_id: str
    index_path: str
    event_prefix: str


@dataclass(frozen=True)
class Event:
    event_id: str
    occurred_at: str
    kind: str
    subject_ids: str
    phase: str
    outcome: str
    commit: str
    pr: str
    spec: str
    evidence_path: str
    supersedes: str
    summary: str
    next_action: str


def parse_manifest(root: Path) -> tuple[list[Shard], list[str]]:
    path = root / ".agents/state.csv"
    errors: list[str] = []
    try:
        if path.stat().st_size > MANIFEST_MAX_BYTES:
            return [], [f"{path}: manifest exceeds the 64 KiB limit"]
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = tuple(next(reader, ()))
            if header != MANIFEST_HEADER:
                return [], [f"{path}: expected manifest header {MANIFEST_HEADER!r}"]
            shards: list[Shard] = []
            for line_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                location = f"{path}:{line_number}"
                if len(row) != len(MANIFEST_HEADER):
                    errors.append(
                        f"{location}: expected {len(MANIFEST_HEADER)} columns, "
                        f"found {len(row)}"
                    )
                    continue
                shard = Shard(*row)
                if any(any(control in field for control in ("\r", "\n", "\0")) for field in row):
                    errors.append(f"{location}: control character in manifest scalar")
                if shard.schema_version != "1":
                    errors.append(f"{location}: schema version must be 1")
                if SHARD_ID_RE.fullmatch(shard.shard_id) is None:
                    errors.append(f"{location}: invalid shard id {shard.shard_id!r}")
                expected_index = f".agents/state-index/{shard.shard_id}.csv"
                if shard.index_path != expected_index:
                    errors.append(
                        f"{location}: index path must be {expected_index!r}"
                    )
                period = shard.shard_id[:7]
                expected_prefix = f".agents/state-events/{period}/"
                if shard.event_prefix != expected_prefix:
                    errors.append(
                        f"{location}: event prefix must be {expected_prefix!r}"
                    )
                shards.append(shard)
    except (OSError, UnicodeError, csv.Error, TypeError) as exc:
        errors.append(f"{path}: cannot parse manifest: {exc}")
        return [], errors
    shard_ids = [shard.shard_id for shard in shards]
    if shard_ids != sorted(set(shard_ids)):
        errors.append(f"{path}: shard IDs must be unique and in increasing order")
    if errors:
        return [], errors
    return shards, errors


def _has_control(field: str) -> bool:
    return any(control in field for control in ("\r", "\n", "\0"))


def _valid_utc_timestamp(value: str) -> bool:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _valid_calendar_date(value: str) -> bool:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _validate_evidence(root: Path, event: Event, *, legacy: bool) -> list[str]:
    path = root / event.evidence_path
    errors: list[str] = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"{path}: cannot read evidence: {exc}"]
    if not legacy and len(raw) > EVENT_MAX_BYTES:
        errors.append(f"{path}: post-cutover event exceeds the 32 KiB limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        return [*errors, f"{path}: evidence is not UTF-8: {exc}"]
    marker = f"<!-- state-event: {event.event_id} -->"
    if marker not in text.splitlines():
        errors.append(f"{path}: event marker must match {event.event_id}")
    if legacy:
        try:
            read_legacy_payload(path)
        except ValueError as exc:
            errors.append(f"{path}: {exc}")
        return errors
    for section in REQUIRED_SECTIONS:
        match = re.search(
            rf"(?ms)^## {re.escape(section)}\s*\n(.+?)(?=^## |\Z)", text
        )
        if match is None or not match.group(1).strip():
            errors.append(f"{path}: required section {section!r} is missing or empty")
    return errors


def _event_scalar_errors(event: Event, shard: Shard, location: str) -> list[str]:
    errors: list[str] = []
    fields = tuple(getattr(event, name) for name in EVENT_HEADER)
    if any(_has_control(field) for field in fields):
        errors.append(f"{location}: control character in event scalar")
    legacy = event.kind == "legacy_import"
    new_match = NEW_EVENT_ID_RE.fullmatch(event.event_id)
    if not legacy and new_match is None:
        errors.append(f"{location}: invalid post-cutover event ID {event.event_id!r}")
    if (
        legacy
        and LEGACY_EVENT_ID_RE.fullmatch(event.event_id) is None
        and new_match is None
    ):
        errors.append(f"{location}: invalid legacy event ID {event.event_id!r}")
    if event.kind not in KINDS:
        errors.append(f"{location}: invalid kind {event.kind!r}")
    if legacy:
        if new_match is not None:
            if not _valid_utc_timestamp(event.occurred_at):
                errors.append(
                    f"{location}: timestamped legacy event requires an RFC 3339 UTC timestamp"
                )
            else:
                compact = datetime.strptime(
                    event.occurred_at, "%Y-%m-%dT%H:%M:%SZ"
                ).strftime("%Y%m%dT%H%M%S")
                if new_match.group(1) != compact:
                    errors.append(
                        f"{location}: event ID timestamp disagrees with occurred_at"
                    )
        elif (
            event.occurred_at
            and not _valid_calendar_date(event.occurred_at)
            and not _valid_utc_timestamp(event.occurred_at)
        ):
            errors.append(f"{location}: invalid legacy timestamp {event.occurred_at!r}")
        if event.subject_ids and any(
            SUBJECT_RE.fullmatch(subject) is None
            for subject in event.subject_ids.split(";")
        ):
            errors.append(f"{location}: invalid subject ID syntax")
        if event.phase and event.phase not in PHASES:
            errors.append(f"{location}: invalid phase {event.phase!r}")
        if event.outcome and event.outcome not in OUTCOMES:
            errors.append(f"{location}: invalid outcome {event.outcome!r}")
    else:
        if not _valid_utc_timestamp(event.occurred_at):
            errors.append(f"{location}: invalid timestamp {event.occurred_at!r}")
        elif new_match is not None:
            compact = datetime.strptime(
                event.occurred_at, "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%Y%m%dT%H%M%S")
            if new_match.group(1) != compact:
                errors.append(f"{location}: event ID timestamp disagrees with occurred_at")
        if not event.subject_ids:
            errors.append(f"{location}: subject IDs are required")
        elif any(
            SUBJECT_RE.fullmatch(subject) is None
            for subject in event.subject_ids.split(";")
        ):
            errors.append(f"{location}: invalid subject ID syntax")
        if event.phase not in PHASES:
            errors.append(f"{location}: invalid phase {event.phase!r}")
        if event.outcome not in OUTCOMES:
            errors.append(f"{location}: invalid outcome {event.outcome!r}")
        if not event.summary:
            errors.append(f"{location}: summary is required")
        if not event.next_action:
            errors.append(f"{location}: next action is required")
    if event.commit and COMMIT_RE.fullmatch(event.commit) is None:
        errors.append(f"{location}: commit must be a lowercase 40-character SHA")
    if event.pr and PR_RE.fullmatch(event.pr) is None:
        errors.append(f"{location}: PR must use pr:<number>")
    expected_evidence = f"{shard.event_prefix}{event.event_id}.md"
    if event.evidence_path != expected_evidence:
        errors.append(f"{location}: evidence path must be {expected_evidence!r}")
    if len(event.summary) > SUMMARY_MAX_CHARS:
        errors.append(f"{location}: summary exceeds {SUMMARY_MAX_CHARS} characters")
    if len(event.next_action) > NEXT_ACTION_MAX_CHARS:
        errors.append(
            f"{location}: next action exceeds {NEXT_ACTION_MAX_CHARS} characters"
        )
    return errors


def parse_events(root: Path, shards: list[Shard]) -> tuple[list[Event], list[str]]:
    events: list[Event] = []
    errors: list[str] = []
    for shard in shards:
        path = root / shard.index_path
        try:
            if path.stat().st_size > SHARD_MAX_BYTES:
                errors.append(f"{path}: index shard exceeds the 256 KiB limit")
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = tuple(next(reader, ()))
                if header != EVENT_HEADER:
                    errors.append(f"{path}: expected event header {EVENT_HEADER!r}")
                    continue
                shard_events: list[Event] = []
                for line_number, row in enumerate(reader, start=2):
                    if not row:
                        continue
                    location = f"{path}:{line_number}"
                    if len(row) != len(EVENT_HEADER):
                        errors.append(
                            f"{location}: expected {len(EVENT_HEADER)} columns, "
                            f"found {len(row)}"
                        )
                        continue
                    event = Event(*row)
                    errors.extend(_event_scalar_errors(event, shard, location))
                    errors.extend(
                        _validate_evidence(
                            root, event, legacy=event.kind == "legacy_import"
                        )
                    )
                    shard_events.append(event)
        except (OSError, UnicodeError, csv.Error, TypeError) as exc:
            errors.append(f"{path}: cannot parse event index: {exc}")
            continue
        if len(shard_events) > SHARD_MAX_EVENTS:
            errors.append(f"{path}: index shard exceeds the 512-event limit")
        event_ids = [event.event_id for event in shard_events]
        if event_ids != sorted(set(event_ids)):
            errors.append(f"{path}: event IDs must be unique and in increasing order")
        events.extend(shard_events)
    if errors:
        return [], errors
    return events, []


def read_legacy_payload(path: Path) -> bytes:
    raw = path.read_bytes()
    if raw.count(LEGACY_BEGIN) != 1 or raw.count(LEGACY_END) != 1:
        raise ValueError("legacy payload must contain one begin and one end marker")
    start = raw.index(LEGACY_BEGIN) + len(LEGACY_BEGIN)
    end = raw.index(LEGACY_END, start)
    return raw[start:end]
