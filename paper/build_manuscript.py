#!/usr/bin/env python3
"""Concatenate the 5 DRAFT_v1_*.md files into a single manuscript markdown.

Output: manuscript_v1.md — venue-agnostic, includes:
  - Title block
  - Abstract (from ABSTRACT.md)
  - Sections 1-8 + Acknowledgements
  - Inline LaTeX-style citations
  - Notes/debug content stripped (only paper-grade prose retained)
"""
import re
from pathlib import Path

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem/paper")
OUT = ROOT / "manuscript_v1.md"

TITLE = "Stratified machine-learning bias correction of FABDEM transfers between contrasting Chilean climate regimes: evidence from ICESat-2 over Mediterranean and humid temperate watersheds"
AUTHOR = "Francisco Parra Ortiz"
AFFILIATION = "[Institutional affiliation to be added at submission]"
EMAIL = "gran.huja@gmail.com"
DATE = "2026-05-16"


def extract_section_body(md_file):
    """Read a draft file and strip metadata / notes / word counts.

    Convention used in the drafts:
      - First block before first `## ` is metadata; skip
      - Body starts at first `## ` (Section heading)
      - Body ends at:
          - First `---` line followed by `## Word count` or similar meta
          - Or first `## Word count` / `## Notes` etc.
    """
    text = Path(md_file).read_text()
    # Find first H2 heading (the actual section start)
    lines = text.split("\n")
    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and not line.startswith("## Word count") \
                and "Continuation of" not in line \
                and not line.startswith("## Target"):
            start_idx = i
            break
    if start_idx is None:
        return ""
    # Find end: first appearance of '---' followed by '## Word count' or
    # '## Notes' or '## Cumulative' or `## Citations` etc.
    end_idx = len(lines)
    in_body = True
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if line == "---":
            # Look ahead a few lines to see what follows
            for j in range(i + 1, min(i + 6, len(lines))):
                next_line = lines[j].strip()
                if next_line.startswith("## Word count") \
                        or next_line.startswith("## Citations") \
                        or next_line.startswith("## Cumulative") \
                        or next_line.startswith("## Notes for") \
                        or next_line.startswith("## Notes") \
                        or next_line.startswith("## Final draft") \
                        or next_line.startswith("## Changes vs"):
                    end_idx = i
                    in_body = False
                    break
            if not in_body:
                break
    body = "\n".join(lines[start_idx:end_idx]).rstrip()
    return body


# Build manuscript
print(f"Building unified manuscript → {OUT}")
parts = []

# Title block
parts.append(f"""---
title: "{TITLE}"
author: {AUTHOR}
affiliation: {AFFILIATION}
email: {EMAIL}
date: {DATE}
keywords: FABDEM, bias correction, ICESat-2 ATL08, XGBoost, spatial-block cross-validation, Chile, out-of-distribution generalization
---

# {TITLE}

**{AUTHOR}**
*{AFFILIATION}*
{EMAIL}

**Draft v1 — {DATE}**

---

## Abstract""")

# Pull abstract from ABSTRACT.md (just the prose paragraph)
abs_text = (ROOT / "ABSTRACT.md").read_text()
# The abstract is the single paragraph between the first --- divider and the next ---
match = re.search(r"^---\s*$\s*(.+?)\s*^---\s*$", abs_text, flags=re.MULTILINE | re.DOTALL)
if match:
    abstract_para = match.group(1).strip()
    # Strip the title block above
    abstract_para = abstract_para.replace("# Abstract — submission-ready draft v3 (post tex-review)", "").strip()
    # Find the actual abstract paragraph (last block before ---)
    paragraphs = [p.strip() for p in abstract_para.split("\n\n") if p.strip()]
    # The last paragraph is the abstract
    paragraphs = [p for p in paragraphs if not p.startswith("**")]
    if paragraphs:
        parts.append(paragraphs[-1])

parts.append("\n**Keywords:** FABDEM; bias correction; ICESat-2 ATL08; XGBoost; spatial-block cross-validation; Chile; out-of-distribution generalization\n\n---\n")

# Sections 1-8
section_files = [
    "DRAFT_v1_sections_1_to_3.md",
    "DRAFT_v1_section_4.md",
    "DRAFT_v1_section_5.md",
    "DRAFT_v1_section_6.md",
    "DRAFT_v1_sections_7_and_8.md",
]

for sf in section_files:
    body = extract_section_body(ROOT / sf)
    if body:
        parts.append(body)
        parts.append("\n")
        print(f"  + {sf}: {len(body.split())} words")

# References note
parts.append("""---

## References

All citations resolve to entries in the accompanying `references.bib` file (29 entries, verified against OpenAlex / CrossRef via the `verify-refs` skill). The reference list will be auto-generated at LaTeX compile time using the Taylor & Francis or Elsevier bibliography style appropriate for the chosen venue.

---

*End of manuscript v1.*
""")

OUT.write_text("\n".join(parts))
n_words = len(OUT.read_text().split())
print(f"\n→ {OUT} ({n_words:,} words, {OUT.stat().st_size/1024:.1f} KB)")
print(f"\nManuscript structure:")
for line in OUT.read_text().split("\n"):
    if line.startswith("## ") or line.startswith("# "):
        print(f"  {line}")
