# Phase 1F: description validation

This CPU-only audit complements generation-time checks with dataset-wide duplicate
checks and an explicit filtering policy. It never rewrites generated descriptions,
checkpoints, old validation statuses, or split assignments. It uses the selected
text field, not the saved valid/invalid label, for its checks.

## Run on rama

After reviewing/committing/pushing locally:

```bash
cd /home/jayanth/projects/nanovlm_new
git pull --ff-only
conda activate qwen-vl

python -m src.data.validate_descriptions \
  --input data/generated/shortdesc_qwen/shortdesc.jsonl \
  --field short_desc \
  --config configs/description_validation.json \
  --report data/processed/validation/shortdesc_report.json

python -m src.data.validate_descriptions \
  --input data/generated/longdesc_qwen_v2/longdesc.jsonl \
  --field long_desc \
  --config configs/description_validation.json \
  --report data/processed/validation/longdesc_report.json
```

Run commands individually and stop on operational errors. The printed summary
reports flag counts, word-count distribution, duplicate groups, and counts selected
by the configured policy. Exit zero means the report was produced, not that the
dataset passed every check. No GPU/model imports or remote services are needed.
Reports and subsets under `data/processed/` are ignored by Git.

## Checks and interpretation

- Word counts use whitespace-delimited tokens containing letters/digits, consistent
  with generation. The configurable review ranges default to 20–27 for ShortDesc
  and 60–70 for LongDesc. An out-of-range result is a flag, not automatic rejection.
- Empty/missing output, non-string output, punctuation-only output, obvious
  reasoning/markdown/list/heading artifacts and unusual control characters are
  flagged. Ordinary multiline prose is allowed. These are heuristic format checks,
  not grammar, language, hallucination or grounding checks.
- Every member of a duplicate-output group is flagged after Unicode NFKC
  normalization, case folding and whitespace collapsing. Punctuation remains
  significant. This detects matching text, not paraphrases or semantic similarity.
  Empty outputs do not create duplicate groups. Cross-assignment groups are
  identified separately; matching text alone is not proof of image/data leakage.
- Duplicate image IDs, invalid IDs/assignments, malformed JSON lines and non-object
  records are reported by source line number rather than silently skipped.
- A final saved attempt marked truncated is flagged when its text matches the
  selected description. EOS/truncation cannot be independently inferred from plain
  text, and no new token generation is performed.

Each report stores the input SHA-256, policy, every line's image ID/assignment,
word count, flags and policy decision, plus duplicate group members and summary.
Flags overlap, so their counts need not sum to the number of flagged records.
There is no source-caption semantic review, image decoding, or proof of complete
split coverage in this phase; those remain separate from description checks.

## Optional filtering: explicit, separate, reversible

The default `exclude_flags: []` selects every source line. Running with only
`--report` never creates a subset, regardless of the policy. To filter, create a
separate reviewed config and list exactly which flags to exclude for each field.
For example, a structural-only policy could list:

```json
["malformed_json", "malformed_record", "invalid_image_id", "invalid_assignment",
 "empty_output", "malformed_output", "truncated_output"]
```

Add `word_count_outside_range` only if you explicitly want a length filter.
Add `duplicate_output` only if you intend to exclude **all members** of each
normalized duplicate group; there is no arbitrary keep-first behavior.
`duplicate_image_id` is a separate available flag. Unknown flags fail clearly.

Then run, for example:

```bash
python -m src.data.validate_descriptions \
  --input data/generated/shortdesc_qwen/shortdesc.jsonl \
  --field short_desc \
  --config configs/description_validation.filtered.json \
  --report data/processed/validation/shortdesc_filtered_report.json \
  --filtered-output data/processed/validation/shortdesc_filtered.jsonl
```

The example config filename must be created first. Source lines selected by policy
are copied byte-for-byte, including generation provenance and original statuses.
Consequently, if malformed JSON lines are not explicitly excluded, they also remain
in the subset: no hidden cleanup or removal occurs. Independent short/long filtering
can produce different image sets; later consumers must align IDs explicitly rather
than zip the two files by row position. Filtering does not reshuffle or rebalance
splits, or replace the original manifests.

Reports and subsets cannot overwrite source/configuration files. Existing identical
outputs are reused; different outputs at the same path are refused. For a revised
policy, choose new report/subset paths so the previous audit stays available.
