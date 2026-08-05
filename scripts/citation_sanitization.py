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
# Matches a role marker at start-of-line (^, with (?m) - kept in case a
# newline ever does appear) OR right after sentence-ending punctuation
# (./!/?) - the realistic shape for this corpus. Citation.snippet comes
# from Block 1's note-generation templates by way of Block 4's chunking
# (see this module's docstring), and that pipeline never emits a newline
# inside a note - a marker planted mid-note only ever follows a sentence
# ending, never a line start (same finding genai-block4-rag-eval's own
# sanitize.py fix made against its corpus). The lookbehind is zero-width,
# so the punctuation itself is never consumed/stripped - only the role
# marker and colon are removed. A plain mid-sentence word immediately
# followed by a colon (e.g. "Cardiovascular system: normal.") is NOT
# matched, since neither alternative is satisfied there - "system" isn't
# at start-of-line, and the character before it (through any run of
# whitespace) is an ordinary word character, not one of .!?.
_ROLE_MARKER_RE = re.compile(r"(?im)(?:^|(?<=[.!?]))\s*(system|human|assistant|user)\s*:\s*")
_CHAT_DELIMITER_RE = re.compile(
    r"(?i)\[/?(?:INST|SYS)\]|<\|.*?\|>|#{2,}\s*(?:instructions?|response)\b"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def sanitize_citation_text(text: str) -> str:
    """Strip conversation-turn markers and control characters from a
    citation snippet before it is ever rendered or stored."""
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _ROLE_MARKER_RE.sub("", text)
    text = _CHAT_DELIMITER_RE.sub("", text)
    return text


def trim_citation_snippet(text: str, keywords: list[str]) -> str:
    """Sentence-level keyword-containment rule (spec's LLM02 section):
    keep only the sentences that contain at least one of the parsed query
    terms. If no sentence matches, keep the first sentence only, as a safe
    default rather than dropping the citation entirely.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]
    if not sentences:
        return text

    lowered_keywords = [k.lower() for k in keywords if k]
    matching = [s for s in sentences if any(k in s.lower() for k in lowered_keywords)]
    if matching:
        return " ".join(matching)
    return sentences[0]
