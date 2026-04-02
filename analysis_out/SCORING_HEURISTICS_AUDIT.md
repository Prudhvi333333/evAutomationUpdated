# Scoring Heuristics Audit

## Scope

This note focuses only on the remaining heuristic parts of evaluation:

- `compact_context_segments()` in `src/ev_llm_compare/prompts.py`
- `_segment_response_units()` in `src/ev_llm_compare/evaluation.py`

These are not arithmetic bugs. They are approximation layers that decide:

1. which evidence the judge sees
2. how many claims the judge is asked to label

If either approximation is poor, the final metrics can still drift even when the formula is correct.

## Why this matters

The judge does **not** see the full workbook. It sees:

- a compacted subset of retrieved evidence
- a claim-split version of the answer or reference

So evaluation quality still depends on:

- retrieval compaction quality
- claim segmentation quality
- availability of human golden answers

## Concrete examples

### Example 1: Large list answer collapsed into too few claims

Question:

`Show all Tier 1/2 suppliers in Georgia, list their EV Supply Chain Role and Product / Service.`

Observed reference segmentation:

- claim 1: `There are 18 Tier 1/2 companies in Georgia.`
- claim 2: one giant line containing the whole supplier list

Effect:

- `context_recall` can become unstable because one huge claim stands for many factual items.
- If the context supports 15 out of 18 companies but misses 3, the judge may still mark the giant claim unsupported.

### Example 2: Nicely formatted list splits well

Question:

`What locations does Novelis Inc. operate in, and what primary facility types are associated with each location?`

Observed reference segmentation:

- `Novelis Inc. is recorded at three Georgia locations.`
- `Gainesville, Hall County | Primary Facility Type: Manufacturing Plant`
- `Trenton, Dade County | Primary Facility Type: Manufacturing Plant`
- `Lawrenceville, Gwinnett County | Primary Facility Type: Manufacturing Plant`

Effect:

- This is a strong format for evaluation.
- `context_recall` and claim-level judging are much more interpretable here.

### Example 3: Compaction helps on analytic summary questions

Question:

`Which county have the highest total Employment among Tier 1 suppliers only?`

Compacted evidence selected:

- one `structured_match_summary` block with county totals

Effect:

- This is good.
- The summary is self-sufficient and short enough for the judge to reason over cleanly.

### Example 4: Compaction can hurt network/full-set questions

Question:

`Show all Vehicle Assembly OEMs in Georgia and the full set of Tier 1 suppliers connected to each within the state.`

Observed compacted evidence selected from the old artifact:

- `PPG Industries Inc.`
- `Teklas USA`
- `Suzuki Manufacturing of America Corp.`
- `Kia Georgia Inc.`

Effect:

- The question asks for a full network.
- Only four compact blocks are shown.
- Even with the current fix, scoring may still understate recall if the needed supplier graph does not survive compaction.

## Main residual risks

### 1. Claim splitting is format-sensitive

Relevant code:

- `src/ev_llm_compare/evaluation.py`, `_segment_response_units()`

Current behavior:

- bullet lines split well
- short multi-line blocks split well
- dense long paragraphs often become one or very few claims

Risk:

- large list answers can be judged too harshly because many facts are packed into one claim
- or too leniently if one merged claim is marked supported despite partial errors

### 2. Broad-context detection depends on keyword matching

Relevant code:

- `src/ev_llm_compare/prompts.py`, `_needs_broad_context()`

Current behavior:

- context expansion triggers on phrases like `all suppliers`, `network`, `connected to each`, `map all`

Risk:

- semantically broad questions that do not use these keywords may still get narrow context
- semantically narrow questions containing these keywords may get more context than needed

### 3. Evidence field compaction can drop the most useful columns

Relevant code:

- `src/ev_llm_compare/prompts.py`, `_compact_metadata_line()`
- `src/ev_llm_compare/prompts.py`, `_requested_fields()`

Current behavior:

- only a subset of metadata fields is emitted based on question keywords

Risk:

- if the question implies a needed field but does not literally mention it, the compact block may omit that field
- this can reduce groundedness/precision even when the raw row had the needed data

### 4. Company deduplication can hide needed repeated records

Relevant code:

- `src/ev_llm_compare/prompts.py`, `_select_compact_results()`

Current behavior:

- repeated `row_full` / `company_profile` entries for the same company are deduped

Risk:

- multi-site or multi-role companies may need more than one row to answer correctly
- dedup can suppress a valid second row that contains a different location, OEM link, or product

### 5. Evaluation is strongest only when golden answers exist

Relevant code:

- `src/ev_llm_compare/runner.py`
- `src/ev_llm_compare/evaluation.py`

Current behavior:

- if no golden answer exists, the system can generate a fallback reference

Risk:

- evaluation becomes judge-vs-generated-reference, not judge-vs-human-ground-truth
- this can bake retrieval/model bias into the score

## What would fix or improve this

### A. Better claim segmentation

Most direct fix for list-style scoring.

Ideas:

- split `|`-separated catalog rows into separate claims when company-like names repeat
- split long answer blocks when repeated entity patterns appear
- add entity/list-aware segmentation for workbook-style outputs

Benefit:

- makes `context_recall` and claim-based `faithfulness` more faithful to actual item coverage

### B. Better context compaction

Most direct fix for full-set/network questions.

Ideas:

- raise `context_result_limit` dynamically for broad/network questions
- disable company dedup when question asks for locations, networks, or full supplier sets
- keep one structured summary plus several supporting rows instead of summary-only or arbitrary top rows

Benefit:

- the judge sees enough evidence to evaluate coverage fairly

### C. Better retrieval strategy

This helps before compaction even starts.

Ideas:

- retrieve by sub-query for each requested entity/group
- use graph-style expansion for `connected to each` / supplier-network questions
- preserve multiple rows per company when the question is multi-location or multi-relationship

Benefit:

- compaction starts from a better candidate pool
- fewer scoring failures caused by missing evidence upstream

### D. Prefer human golden answers whenever possible

Benefit:

- stabilizes `answer_accuracy`
- stabilizes `context_precision` and `context_recall`
- avoids circular evaluation against generated references

## Best next step

If the goal is more trustworthy scores on the human50-style workload, the best order is:

1. improve claim segmentation for workbook/list outputs
2. widen or specialize context compaction for network/full-set questions
3. tune retrieval for relationship-heavy questions

Why this order:

- scoring depends first on what claims exist
- then on what evidence survives compaction
- retrieval improvements matter most after those two layers stop throwing away structure
