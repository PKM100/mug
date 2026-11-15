"""Live PubMed retrieval via NCBI E-utilities (esearch + efetch)."""
from __future__ import annotations

import asyncio
import re
import time
import xml.etree.ElementTree as ET

import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_STOP = set("""a an and are as at be by can could did do does for from has have how i in is it its
may might of on or patient patients should than that the their there these this to was were what
when which while who whom why will with would most likely following best next step management
year old man woman presents shows year-old day days week weeks month months hour hours history
right left examination physical following department emergency comes brought physician visit
because due after over since ago""".split())


def keyword_query(question: str, n: int = 6) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", question.lower())
    seen: list[str] = []
    for w in words:
        if w not in _STOP and w not in seen:
            seen.append(w)
    return " ".join(seen[:n])


def clean_query(q: str) -> str:
    q = q.strip().strip("`").splitlines()[0] if q.strip() else ""
    q = re.sub(r"^(query|search)\s*:\s*", "", q, flags=re.I)
    q = q.replace('"', "")  # LLM quoting is often unbalanced; PubMed term mapping does better without it
    return re.sub(r"\s+", " ", q).strip()


def relaxations(term: str) -> list[str]:
    """Progressively broader variants of an AND-chained query: first 3 clauses, first 2, then OR of all."""
    parts = [p.strip() for p in re.split(r"\s+AND\s+", term) if p.strip()]
    if any(p.count("(") != p.count(")") for p in parts):  # AND nested inside parentheses: don't split
        return []
    out: list[str] = []
    for n in (3, 2):
        if len(parts) > n:
            out.append(" AND ".join(parts[:n]))
    if len(parts) > 1:
        out.append(" OR ".join(f"({p})" for p in parts))
    return out


MAX_CHARS = 2500


# Coarse evidence hierarchy from PubMed publication types (lower = stronger design).
EVIDENCE_LEVELS = [
    (1, "Systematic review / meta-analysis", ("Meta-Analysis", "Systematic Review")),
    (2, "Randomized controlled trial", ("Randomized Controlled Trial",)),
    (3, "Clinical trial", ("Clinical Trial", "Controlled Clinical Trial", "Clinical Trial, Phase III",
                           "Clinical Trial, Phase II", "Pragmatic Clinical Trial")),
    (4, "Observational study", ("Observational Study", "Multicenter Study", "Comparative Study",
                                "Validation Study", "Evaluation Study")),
    (5, "Guideline", ("Practice Guideline", "Guideline", "Consensus Development Conference")),
    (6, "Case report", ("Case Reports",)),
    (7, "Review / opinion", ("Review", "Editorial", "Comment", "Letter")),
]


def evidence_quality(pub_types: list[str], retracted: bool, year: str, now_year: int | None = None) -> dict:
    """Study-design level, retraction status and age from PubMed metadata. Retracted beats everything."""
    import datetime
    now_year = now_year or datetime.date.today().year
    level, label = 8, "Other / unclassified"
    for lv, lab, types in EVIDENCE_LEVELS:
        if any(t in pub_types for t in types):
            level, label = lv, lab
            break
    age = now_year - int(year) if year.isdigit() else None
    flags = []
    if retracted:
        flags.append("retracted")
    if age is not None and age > 15:
        flags.append("older than 15 years")
    return {"level": level, "label": "Retracted" if retracted else label, "retracted": retracted,
            "age_years": age, "flags": flags}


def _cap(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS].rsplit(" ", 1)[0] + " …"


class PubMedClient:
    def __init__(self, api_key: str | None = None, http: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0),
                                               headers={"User-Agent": "mug-app/0.1 (research prototype)"})
        # NCBI allows 3 req/s without an API key, 10 with one; exceeding it returns HTTP 429.
        self._interval = 0.11 if api_key else 0.35
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def _get(self, url: str, params: dict, retries: int = 3) -> httpx.Response:
        for attempt in range(retries + 1):
            async with self._lock:
                wait = self._next - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._next = time.monotonic() + self._interval
            r = await self._http.get(url, params=params)
            if r.status_code in (429, 500, 502, 503) and attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        raise RuntimeError("unreachable")

    def _params(self, **kw) -> dict:
        p = {"tool": "mug-app", **kw}
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    async def esearch(self, term: str, k: int) -> list[str]:
        r = await self._get(f"{EUTILS}/esearch.fcgi", self._params(
            db="pubmed", term=f"({term}) AND hasabstract", retmax=k, sort="relevance", retmode="json"))
        return r.json().get("esearchresult", {}).get("idlist", [])

    async def efetch(self, pmids: list[str]) -> list[dict]:
        if not pmids:
            return []
        r = await self._get(f"{EUTILS}/efetch.fcgi", self._params(
            db="pubmed", id=",".join(pmids), retmode="xml"))
        root = ET.fromstring(r.text)
        out = []
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID") or ""
            title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
            parts = []
            for ab in art.iter("AbstractText"):
                label = ab.get("Label")
                txt = "".join(ab.itertext()).strip()
                if txt:
                    parts.append(f"{label}: {txt}" if label else txt)
            if not parts:
                continue
            journal = art.findtext(".//Journal/ISOAbbreviation") or art.findtext(".//Journal/Title") or ""
            year = art.findtext(".//PubDate/Year") or (art.findtext(".//PubDate/MedlineDate") or "")[:4]
            pub_types = [t.text or "" for t in art.iter("PublicationType")]
            retracted = ("Retracted Publication" in pub_types or any(
                c.get("RefType") == "RetractionIn" for c in art.iter("CommentsCorrections")))
            out.append({
                "id": f"PMID:{pmid}",
                "source": "pubmed",
                "title": title.strip(),
                "content": _cap(f"{title.strip()} {' '.join(parts)}"),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "meta": {"journal": journal, "year": year, "pub_types": pub_types,
                         "quality": evidence_quality(pub_types, retracted, year)},
            })
        order = {p: i for i, p in enumerate(pmids)}
        out.sort(key=lambda d: order.get(d["id"][5:], 1e9))
        return out

    async def search(self, term: str, k: int, fallback: str | None = None) -> tuple[str, list[dict]]:
        """Returns (term actually used, abstracts). Relaxes the query step by step on zero hits."""
        if k <= 0:
            return term, []
        tried: list[str] = []
        for t in [term, *relaxations(term), fallback]:
            if not t or t in tried:
                continue
            tried.append(t)
            ids = await self.esearch(t, k)
            if ids:
                return t, await self.efetch(ids)
        return term, []
