"""Prompt builders. MCQ prompts mirror the MUG paper's research code verbatim where
possible so that served results are comparable with the published numbers."""
from __future__ import annotations

LETTER_PROMPT = (
    "Answer the multiple-choice question with ONLY the letter of the best "
    "option. Output a single character and nothing else."
)

MARGINAL_UTILITY_INSTRUCTION = (
    "You are a CONTEXT EVALUATOR sitting between a retriever and an expert answerer.\n"
    "The answerer already has broad expert knowledge. Your job is to keep ONLY the "
    "passages that would CHANGE or DECIDE the expert's answer to this specific question "
    "— i.e. passages containing the particular finding, number, or fact the question "
    "hinges on. Do NOT keep a passage merely because it is on-topic or generally "
    "relevant; redundant background hurts the answerer. If the question can be answered "
    "from general expert knowledge without any passage, keep NONE."
)


def refs_block(context: list[dict]) -> str:
    if not context:
        return ""
    refs = "\n".join(f"[Doc {i+1}] {c['content'].strip()}" for i, c in enumerate(context))
    return f"Relevant references:\n{refs}\n\n"


def mcq_prompt(question: str, options: dict, context: list[dict]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"{refs_block(context)}{LETTER_PROMPT}\n\nQuestion: {question}\n\nOptions:\n{opts}\n\nAnswer:"


def open_prompt(question: str, context: list[dict]) -> str:
    if context:
        guide = ("Use the references only where they are reliable and relevant; cite them inline as "
                 "[Doc n]. If they conflict with established medical knowledge, say so.")
    else:
        guide = "Answer from your own medical knowledge."
    return (
        f"{refs_block(context)}You are assisting a clinician/researcher. {guide} "
        "Give a direct, precise answer in at most 120 words. Lead with the bottom-line answer "
        "in the first sentence.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def equivalence_prompt(question: str, a: str, b: str) -> str:
    return (
        "Two answers to the same medical question are shown. Decide whether they reach the SAME "
        "bottom-line conclusion (same diagnosis/treatment/claim/direction of effect). Ignore wording, "
        "length, citations and extra detail.\n\n"
        f"Question: {question}\n\nAnswer 1: {a}\n\nAnswer 2: {b}\n\n"
        "Do they reach the same bottom-line conclusion? Answer with ONLY 'yes' or 'no'.\nAnswer:"
    )


def entail_prompt(question: str, answer: str, context: list[dict]) -> str:
    refs = "\n".join(f"[Doc {i+1}] {c['content'].strip()}" for i, c in enumerate(context))
    return (
        f"References:\n{refs}\n\n"
        f"Question: {question}\n"
        f"Proposed answer: {answer}\n\n"
        "Do the references provide evidence that DIRECTLY supports this proposed "
        "answer? Answer with ONLY 'yes' or 'no'.\nAnswer:"
    )


def evaluator_prompt(question: str, candidates: list[dict]) -> str:
    lines = [MARGINAL_UTILITY_INSTRUCTION, "", f"Question: {question}", "", "Candidate passages:"]
    for i, c in enumerate(candidates):
        text = c["content"].strip().replace("\n", " ")
        if len(text) > 700:
            text = text[:700] + "..."
        lines.append(f"[{i}] {text}")
    lines += [
        "",
        "Respond with a JSON object ONLY, of the form:",
        '{"needs_context": true/false, "keep": [ids...], '
        '"decisive_sentence": {"id": "the single most decisive sentence copied verbatim"}}',
        "If no passage is needed, use needs_context=false and keep=[].",
    ]
    return "\n".join(lines)


def pubmed_query_prompt(question: str, options: dict[str, str] | None = None) -> str:
    opts = ""
    if options:
        opts = "\nCandidate answers: " + "; ".join(options.values())
    return (
        "Write a PubMed search query to find abstracts that would help answer this question.\n"
        "Rules: 2-3 core biomedical concepts joined with AND (use OR inside parentheses for synonyms or "
        "competing candidate answers); no quotes, no field tags, no explanation. For a clinical vignette, "
        "query the key finding(s) and the condition or intervention being asked about, not demographics.\n\n"
        f"Question: {question}{opts}\n\nQuery:"
    )


def verify_prompt(question: str, answer: str, context: list[dict]) -> str:
    def tag(c: dict) -> str:
        q = (c.get("meta") or {}).get("quality")
        if not q:
            return ""
        year = (c.get("meta") or {}).get("year") or "n.d."
        return f"({q['label']}, {year}{'; ' + ', '.join(q['flags']) if q['flags'] else ''}) "
    refs = "\n".join(f"[Doc {i+1}] {tag(c)}{c.get('full_content', c['content']).strip()}" for i, c in enumerate(context))
    return (
        "You are an independent evidence auditor for a medical QA system. Another model changed its answer "
        "after reading the references below. Decide whether the references contain a specific statement "
        "(a finding, result, number, or explicit fact) that DIRECTLY establishes the proposed answer to THIS "
        "question. Topical relevance, background, or general plausibility is NOT enough; for a study question "
        "the reference must report the study's own result.\n\n"
        f"References:\n{refs}\n\nQuestion: {question}\nProposed answer: {answer}\n\n"
        'Respond with JSON ONLY: {"supported": true/false, "doc": n or null, '
        '"quote": "the decisive sentence copied verbatim, or empty"}'
    )
