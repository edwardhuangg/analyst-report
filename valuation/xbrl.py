"""Minimal XBRL instance reader: contexts, dimensions and numeric facts."""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

XBRLI = "{http://www.xbrl.org/2003/instance}"
XBRLDI = "{http://xbrl.org/2006/xbrldi}"
XSI_NIL = "{http://www.w3.org/2001/XMLSchema-instance}nil"


@dataclass(frozen=True)
class Context:
    start: Optional[date]
    end: date
    dims: frozenset  # of (axis, member) pairs, both "prefix:Name"

    @property
    def days(self) -> int:
        return (self.end - self.start).days if self.start else 0


def _prefix_for(uri: str, nsmap: dict) -> str:
    if "fasb.org/us-gaap" in uri:
        return "us-gaap"
    if "xbrl.sec.gov/dei" in uri:
        return "dei"
    if "fasb.org/srt" in uri:
        return "srt"
    return nsmap.get(uri, "ext")


class Instance:
    def __init__(self, raw: bytes):
        nsmap = {}
        for _event, (pfx, uri) in ET.iterparse(io.BytesIO(raw), events=("start-ns",)):
            nsmap.setdefault(uri, pfx)
        root = ET.fromstring(raw)
        self.contexts: dict[str, Context] = {}
        for c in root.iter(XBRLI + "context"):
            p = c.find(XBRLI + "period")
            inst, st, en = (p.find(XBRLI + t) for t in ("instant", "startDate", "endDate"))
            end = (en if en is not None else inst).text.strip()
            dims = frozenset((m.get("dimension"), (m.text or "").strip())
                             for m in c.iter(XBRLDI + "explicitMember"))
            self.contexts[c.get("id")] = Context(
                date.fromisoformat(st.text.strip()) if st is not None else None,
                date.fromisoformat(end[:10]), dims)
        # facts[concept][context] = value
        self.facts: dict[str, dict[Context, float]] = defaultdict(dict)
        self.text: dict[str, str] = {}
        for el in root:
            cref = el.get("contextRef")
            if not cref or el.get(XSI_NIL) == "true":
                continue
            uri, local = el.tag[1:].split("}")
            concept = f"{_prefix_for(uri, nsmap)}:{local}"
            ctx = self.contexts.get(cref)
            val = (el.text or "").strip()
            if el.get("unitRef") is not None:
                try:
                    self.facts[concept].setdefault(ctx, float(val))
                except ValueError:
                    pass
            elif not ctx.dims:
                self.text.setdefault(concept, val)

    # ------------------------------------------------------------ lookups
    def dei(self, name: str) -> Optional[str]:
        return self.text.get("dei:" + name)

    def value(self, concepts: Iterable[str], *, start: Optional[date] = None, end: date,
              dims: frozenset = frozenset(), tol: int = 4) -> Optional[float]:
        """First concept with a fact for this period (duration if start given, else instant)."""
        for concept in concepts:
            if ":" not in concept:
                concept = "us-gaap:" + concept
            for ctx, v in self.facts.get(concept, {}).items():
                if ctx.dims != dims or abs((ctx.end - end).days) > tol:
                    continue
                if start is None and ctx.start is None:
                    return v
                if start is not None and ctx.start is not None and abs((ctx.start - start).days) <= tol:
                    return v
        return None

    def durations(self, concepts: Iterable[str], min_days: int, max_days: int) -> list[Context]:
        """Distinct undimensioned duration contexts for these concepts, newest first."""
        seen = {}
        for concept in concepts:
            for ctx in self.facts.get("us-gaap:" + concept, {}):
                if not ctx.dims and ctx.start and min_days <= ctx.days <= max_days:
                    seen[(ctx.start, ctx.end)] = ctx
        return sorted(seen.values(), key=lambda c: c.end, reverse=True)

    def dimensional(self, concept: str) -> dict[Context, float]:
        if ":" not in concept:
            concept = "us-gaap:" + concept
        return {c: v for c, v in self.facts.get(concept, {}).items() if c.dims}

    def concepts_matching(self, pattern: str) -> list[str]:
        rx = re.compile(pattern)
        return [c for c in self.facts if rx.search(c)]


def humanize_member(member: str) -> str:
    """'hon:EnergyAndSustainabilitySolutionsMember' -> 'Energy & Sustainability Solutions'."""
    name = member.split(":")[-1]
    # case-insensitive (MRK: 'PharmaceuticalsegmentMember' showed as 'Pharmaceuticalsegment')
    name = re.sub(r"(?i)(reportable)?(operating)?(segments?)?member$", "", name)
    name = re.sub(r"(?<=[a-z])and(?=[A-Z])", " And ", name)  # DUK: 'UtilitiesandInfrastructure'
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=\.)(?=[A-Z][a-z])", " ", name)  # 'U.S.Domestic'
    name = re.sub(r"\bAnd\b", "&", name)
    return re.sub(r"\s+", " ", name).strip() or member
