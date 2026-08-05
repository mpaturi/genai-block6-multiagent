"""Citation hardening for MultiAgentAnswer.citations (see genai-block7-
security's docs/spec.md LLM01/LLM02 sections, this repo's docs/tasks.md
"Block 6 - citation hardening" phase).

Citation.snippet is populated straight from Block 4's chunk_text (raw
patient note text), which is currently unsanitized before it reaches this
repo - Block 4's own ingestion-time sanitizer (PR #13 on
genai-block4-rag-eval) is still open, not merged. Two independent layers
live here, applied together at the one place citations are constructed
(scripts/orchestrator.py::_citations_from_clinical):

1. sanitize_citation_text (LLM01, indirect prompt injection): strips the
   structural markers an injection needs to fake a new conversation turn -
   role prefixes ("System:", "Human:", etc.), chat-template delimiters
   ("[INST]", "<|...|>", "### Instruction"), and raw control characters.
   Mirrors Block 4's own sanitize.py approach (structural stripping, not
   phrase blocklisting - trivially bypassed by rewording, and gives false
   confidence it "caught" prompt injection).
2. trim_citation_snippet (LLM02, field-layer over-exposure): the spec's
   sentence-level keyword-containment rule - keep only sentences
   containing a parsed query term, or the first sentence as a safe default
   if none match. Cuts down how much of a patient's raw note text a
   citation exposes, independent of the injection question.
"""
import re

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Matches any of the four role words anywhere a word boundary (\b) allows
# - not just at start-of-line or right after sentence-ending punctuation.
# An earlier version anchored this to sentence boundaries specifically to
# let phrasing like "Cardiovascular system: normal." survive untouched,
# but that left a real gap: a marker spliced in after a comma or plain
# mid-sentence space (e.g. "Vitals stable, System: ignore...") went
# undetected. Checked directly against the real corpus this sanitizer
# runs against (Block 1's chunk_records.py templates, and all 11,436
# records in data/raw/graph_export.jsonl): "Cardiovascular system:"-style
# phrasing never actually occurs there, so the anchor was guarding
# against a hypothetical case at the cost of leaving "system" - the
# single most common real-world injection marker word - less protected
# than the other three. All four words now share one unanchored pattern.
# \b still guards against matching inside a longer word (e.g.
# "ecosystem:" has no boundary between "eco" and "system", so it's never
# mistaken for a role marker).
_ROLE_MARKER_RE = re.compile(r"(?i)\b(?:system|human|assistant|user)\s*:\s*")
_CHAT_DELIMITER_RE = re.compile(
    r"(?i)\[/?(?:INST|SYS)\]|<\|.*?\|>|#{2,}\s*(?:instructions?|response)\b"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Explicit per-citation cap, independent of any upstream chunking
# guarantee. Measured directly against this repo's own captured RAG
# fixture data (data/eval/rag_fixtures.json - 102 real citation
# snippets, from Block 4's actual chunker): min 25, median 144, p90 190,
# max 200 chars - the 200 ceiling exactly matches Block 4's own
# chunk_records.py chunking cap. Under normal operation this function
# never sees more than one already-≤200-char chunk, so joining its
# matching sentences can't exceed that either - but this function has no
# way to verify that guarantee still holds for whatever called it, so it
# enforces its own bound rather than trusting it silently. 500 matches
# genai-block4-rag-eval's own precedent (QueryRequest.question's
# max_length=500, chosen the same way: several times its own real
# observed max of 98 chars) - comfortably above every real value seen
# here, while still a real, enforced ceiling.
_MAX_CITATION_SNIPPET_LENGTH = 500


def sanitize_citation_text(text: str) -> str:
    """Strip conversation-turn markers and control characters from a
    citation snippet before it is ever rendered or stored.

    The role-marker substitution replaces with a single space, not "" -
    matching genai-block4-rag-eval's own sanitize.py fix, identically.
    Stripping to empty removes the whitespace on both sides of the marker
    too, fusing the sentence before it directly onto the text after it
    (e.g. "...fatigue. System: ignore..." -> "...fatigue.ignore...", zero
    space after the period). trim_citation_snippet below splits on
    whitespace after sentence-ending punctuation to find sentence
    boundaries - a fused boundary here would silently defeat that split,
    leaving the injection sentence's own phrasing fused onto the kept
    sentence instead of excluded from it. A single space keeps the
    boundary intact while still fully removing the marker itself.
    """
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _ROLE_MARKER_RE.sub(" ", text)
    text = _CHAT_DELIMITER_RE.sub("", text)
    return text


def trim_citation_snippet(text: str, keywords: list[str]) -> str:
    """Sentence-level keyword-containment rule (spec's LLM02 section):
    keep only the sentences that contain at least one of the parsed query
    terms. If no sentence matches, keep the first sentence only, as a safe
    default rather than dropping the citation entirely.

    The result is always capped at _MAX_CITATION_SNIPPET_LENGTH, however
    it was produced - joining every matching sentence, falling back to
    the first sentence, or returning the input as-is when it has no
    sentence-ending punctuation to split on at all.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]
    if not sentences:
        return text[:_MAX_CITATION_SNIPPET_LENGTH]

    lowered_keywords = [k.lower() for k in keywords if k]
    matching = [s for s in sentences if any(k in s.lower() for k in lowered_keywords)]
    result = " ".join(matching) if matching else sentences[0]
    return result[:_MAX_CITATION_SNIPPET_LENGTH]
