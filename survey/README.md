# Agent Characteristics Survey

## Survey Overview

This directory contains the data, codebook, coding records, and inter-coder analysis for Section 3 and Appendix B of the paper. We reviewed a purposive sample of 85 papers and coded 23 of the 85 reviewed sources (agent-risk taxonomies, surveys, and governance frameworks) alongside 30 incidents from the AI Incident Database (AIID) covering 2025 and 2026 reports of tool-using AI agents.

Two language-model coders from different model families coded each item independently against a fixed 20-code codebook:
- First coder: Gemini 3.8 Flash.
- Second coder: Claude Sonnet 5.5, run blind to the first coder's results.

Both coders evaluated the sources and incident records directly from the primary URLs. For each source, characteristics were coded as top-level category (T), named or discussed (N), or absent (0). For each incident, characteristics were coded as contributing to the incident (Y) or absent (0). Every non-zero cell includes a pinpoint reference and a supporting quote.

The counts are a structured two-coder reading of a purposive sample, not a measured consensus of the field.

## File Manifest

- `sources.tsv`: Tab-separated table of the 23 coded sources: identifier (`S01` to `S23`), citation key in the paper's bibliography, author and year, title, type, publication, the version consulted as recorded at coding time (`Not recorded` where the coders did not record it), the URL the coders fetched, and the access dates. The column `arxiv_version_served_on_coding_dates` is inferred, not recorded: for an arXiv source fetched by its unversioned URL, it gives the latest arXiv version as of 2026-10-03 from the arXiv submission history. Where both columns name an arXiv version, they agree.
- `incidents.tsv`: Tab-separated table of the 30 coded incidents. Columns include AI Incident Database incident identifier (`aiid`), canonical citation URL, title as recorded in database reports, and incident date.
- `codebook.md`: Verbatim definitions and scoring rules for all 20 candidate agent characteristics (`C1` through `C20`).
- `coding/sources-coder1.tsv`: Coding matrix for sources by the first coder (Gemini 3.8 Flash), containing values T, N, or 0 across characteristics `C1` to `C20`.
- `coding/sources-coder2.tsv`: Coding matrix for sources by the second coder (Claude Sonnet 5.5), containing values T, N, or 0 across characteristics `C1` to `C20`.
- `coding/incidents-coder1.tsv`: Coding matrix for incidents by the first coder (Gemini 3.8 Flash), containing values Y or 0 across characteristics `C1` to `C20`.
- `coding/incidents-coder2.tsv`: Coding matrix for incidents by the second coder (Claude Sonnet 5.5), containing values Y or 0 across characteristics `C1` to `C20`.
- `coding/evidence-coder1.tsv`: Evidence entries from the first coder. Columns include item identifier, characteristic code, coded value, pinpoint location, and excerpted quote (trimmed to 300 characters where applicable).
- `coding/evidence-coder2.tsv`: Evidence entries from the second coder. Columns include item identifier, characteristic code, coded value, pinpoint location, and excerpted quote (trimmed to 300 characters where applicable).
- `intercoder.py`: Standalone Python script that reads the four TSV matrices in `coding/` and computes percent agreement, Cohen's kappa, per-characteristic statistics, and reporting counts.

## Inter-Coder Agreement

The table below summarizes inter-coder agreement and Cohen's kappa between the two independent model coders:

| Material | Coding | Cells | Agreement | Cohen's kappa |
|---|---|---|---|---|
| 23 sources | T / N / 0 | 460 | 0.711 | 0.538 |
| 23 sources | any mention (T or N) versus 0 | 460 | 0.837 | 0.674 |
| 23 sources | top level (T) versus not | 460 | 0.854 | 0.588 |
| 30 incidents | Y / 0 | 600 | 0.877 | 0.701 |

A Cohen's kappa between 0.54 and 0.70 reflects moderate to substantial agreement. Characteristics analyzed in detail in the paper, such as irreversible actions (C3), multi-agent interaction (C8), observability gaps (C9), and open versus closed access (C12), show high agreement across sources (kappa = 0.91).

## Replication Instructions

To reproduce the inter-coder analysis and verify all statistical metrics, execute `intercoder.py` using Python 3:

```bash
python survey/intercoder.py
```

The script runs without external dependencies (standard library only). It outputs:
1. Overall agreement and Cohen's kappa across the 460 source cells and 600 incident cells.
2. Per-characteristic breakdown of top-level and any-mention counts.
3. Summary counts for reporting across the focus characteristics.

To omit the summary counts table and display only the raw agreement metrics, pass `--no-counts`:

```bash
python survey/intercoder.py --no-counts
```
