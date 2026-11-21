# GUI Job Details Page — Ranking Ignores Criteria and Feedback (Only the Job-Ad Similarity Is Written Back)

## Summary

The Job Details (workspace) page now shows a rank and a total for every candidate, but the numbers are **not the job's rubric score**. The ranking is driven only by each CV's similarity to the job-ad text. Additional criteria and the like/feedback signal are completely ignored, and the total is put on a hard-coded `0..100` scale instead of the job's rubric currency.

This is a CLI/GUI misfit:

- the GUI starts the **`process_in_waves`** pipeline and renders the
  **`candidates` / `candidate_scores`** tables
  (`talentrankiq_gui/src/components/activities/workspace/SetupPane.tsx`,
  `talentrankiq_gui/src/components/activities/workspace/CandidatesPane.tsx`);
- `process_in_waves` computes similarities for the job ad **and every
  criterion** (`talentrankiq_cli/lib/libprocess-in-waves/src/lib.rs:52-66`),
  but its write-back only persists the **job-ad** row
  (`talentrankiq_cli/lib/libprocess-in-waves/src/db.rs:428-530`);
- the GUI's `list_candidates` reads `candidates.points` /
  `candidate_scores` (`talentrankiq_cli/lib/libtriqui-core/src/candidates.rs:277-315`),
  so criteria and feedback can never affect the rank or the total shown on the
  page.

The earlier bug — every row showing `—` — is fixed. This is the remaining
correctness bug: the page renders a score, but it is the wrong score.

## What the user sees

1. Open a job → **Job Details**.
2. Add a criterion (via the CLI/API; the criterion is listed in the left pane,
   e.g. `Kubernetes` with weight 5).
3. Import CVs and wait for ranking to finish.
4. The candidate table shows ranks and a `Total` such as `75.7`.
5. The candidate inspector shows only one rubric row, **Job fit**, and
   `Max Points 100.0`.
6. The job's rubric capacity is not 100: with the default weights it is
   `ad 50 + feedback-job 10 + feedback-profile 5 + criterion 5 = 70`.
7. Liking a candidate changes the star, but the ranking/total does not move.

Expected: the `Total` is the rubric total (`ad` + each criterion + feedback),
`Max Points` is the rubric capacity, and the rank order reflects all of those
inputs.

## Root cause

### 1. The write-back only mirrors the job-ad similarity

`write_back_candidate_scores` pivots only the job-ad similarity:

```rust
// talentrankiq_cli/lib/libprocess-in-waves/src/db.rs:474-480
SELECT c.id, c.file_name,
       MAX(CASE WHEN s.model_id = ?2 THEN s.similarity END) AS deep_sim,
       MAX(CASE WHEN s.model_id = ?3 THEN s.similarity END) AS quick_sim
  FROM candidates c
  JOIN file_state fs ON fs.path = c.source_file_path
  JOIN cv_similarity s ON s.cv_hash = fs.hash
                  AND s.target_hash = ?4          // <- only the job-ad hash
                  AND s.model_id IN (?2, ?3)
 WHERE c.job_id = ?1 AND c.extraction_status = 'Extracted'
 GROUP BY c.id, c.file_name
```

`cv_similarity` contains a row for every target the pipeline was given —
the job ad **and every criterion** — but the query filters on
`target_hash = jobad_hash` only. The criteria rows are computed and then
thrown away.

The insert is likewise a single ad row:

```rust
// talentrankiq_cli/lib/libprocess-in-waves/src/db.rs:501-509
DELETE FROM candidate_scores WHERE candidate_id = ?1 AND question_id = 'ad';
INSERT INTO candidate_scores
  (id, candidate_id, question_id, kind, label, points, max_points, sign)
VALUES (?1, ?2, 'ad', ?3, 'Job fit', ?4, ?5, 'Positive');
UPDATE candidates SET points = ?1, points_max = ?2 WHERE id = ?3;
```

There is no `Criterion`, `FeedbackJob`, `FeedbackProfile`, `Scout`, or
`AdJudge` row, and no criterion/feedback contribution to `candidates.points`.

### 2. The total is hard-coded to a 0..100 scale

```rust
// talentrankiq_cli/lib/libprocess-in-waves/src/db.rs:486
let points = (sim.clamp(0.0, 1.0) * 10000.0).round() / 100.0;  // sim * 100
...
ins.execute(rusqlite::params![..., points, 100.0])?;           // max_points = 100
upd.execute(rusqlite::params![points, 100.0, id])?;            // points_max = 100
```

The job's `ad_weight` (default 50), the feedback weights (10 / 5), and the
criterion weights are never read. The result is on a `0..100` scale while the
rest of the app (rubric dialog, job detail capacity, export) uses the job's
rubric currency. The GUI then renders:

```tsx
// talentrankiq_gui/src/components/activities/workspace/CandidatesPane.tsx:394-395
{candidate.points !== null && candidate.points !== undefined ? (
  <span className="font-mono">{candidate.points.toFixed(1)}</span>
) : (
  <span className="text-muted-foreground">—</span>
)}
```

and in the inspector:

```tsx
// talentrankiq_gui/src/components/activities/workspace/CandidatesPane.tsx:545
{candidate.points_max?.toFixed(1) ?? "—"}
```

so the page shows `75.7` / `Max Points 100.0` for a job whose rubric capacity
is `70`.

### 3. Criteria and feedback can never affect the rank

`list_candidates` builds each row's `scores` from `candidate_scores`
(`talentrankiq_cli/lib/libtriqui-core/src/candidates.rs:277-315`). Because the
waves write-back only inserted the `ad` row, every candidate's `scores` has
exactly one entry. The GUI's rubric breakdown therefore renders only
**Job fit**, and `grade_space` / `ad_source` are derived from that single row.

The old cascade's `rescore` (`talentrankiq_cli/lib/libtriqui-core/src/rescore.rs`)
is not a rescue: it also reads grades from `candidate_scores`
(`stage_scores_for_job`). With no criterion/feedback rows to read, the
criterion and feedback terms are simply absent from `sims`, so the total stays
ad-only. This is why clicking **Like** changes the star but not the ranking.

## Evidence (real `triq`, real models, one criterion)

Started `triq serve` with the real model cache, created a job, set a job ad,
added a criterion (`Kubernetes`, weight 5), pointed the CV directory at six
CVs, and waited for `process_in_waves` to converge.

`list_candidates` returned:

```
cv1.txt rank=1 points=75.66  scores=[('ad', 'Ad', 75.66)]
cv3.txt rank=2 points=75.40  scores=[('ad', 'Ad', 75.40)]
cv0.txt rank=3 points=75.28  scores=[('ad', 'Ad', 75.28)]
cv5.txt rank=4 points=75.24  scores=[('ad', 'Ad', 75.24)]
cv2.txt rank=5 points=75.12  scores=[('ad', 'Ad', 75.12)]
cv4.txt rank=6 points=75.08  scores=[('ad', 'Ad', 75.08)]
```

`get_job_detail` returned the criterion:

```
criteria count 1
crit crit_30211432-... Kubernetes (weight 5)
```

Yet the only score row for every candidate is `ad`. The `Kubernetes`
similarity was computed by the pipeline and stored in `cv_similarity`, but it
never reached `candidate_scores`, so it has zero effect on `points`, `rank`,
or the inspector's rubric breakdown.

## Impact

- Criteria added to a job are dead weight: they are displayed in the left pane
  but never change the order or the score.
- The **Total** column is not in the job's rubric currency; the inspector's
  **Max Points** is wrong.
- **Like / feedback** does not affect the ranking, because the feedback rows
  are never written and `rescore` has no feedback grades to read.
- The rubric breakdown in the candidate inspector is missing every row except
  Job fit.
- Any export or downstream consumer that trusts `candidates.points` /
  `candidate_scores` gets an ad-only score, not the job's rubric total.

## Reproduction

1. `cd talentrankiq_gui && npm run tauri dev`
2. Create a job and open **Job Details**.
3. Set a job-ad file.
4. Add a criterion through the CLI/API, e.g.:
   `add_criterion { job_id, text: "Kubernetes", weight: 5 }`
5. Set **CV directory** to a folder with a few CVs.
6. Wait for the candidate table to show ranks/totals.
7. Open a candidate inspector: the rubric breakdown contains only **Job fit**,
   and `Max Points` is `100.0` even though the job rubric capacity is
   `50 + 10 + 5 + 5 = 70`.
8. Click **Like**: the star changes, but the order and totals do not.

## Suggested fix direction

One source of truth is required. Either:

1. **Complete the waves write-back**: in
   `libprocess-in-waves/src/db.rs::write_back_candidate_scores`, pivot every
   target hash (job ad + each criterion), map each criterion's
   `hash_text(text)` back to its `criteria.id`, and insert a
   `candidate_scores` row per question with the criterion's weight as
   `max_points`. Add the feedback rows (or let `apply_feedback`/`rescore`
   write them) and set `candidates.points` / `points_max` to the rubric total
   and capacity — not a hard-coded 100. Or
2. **Read the waves tables in `list_candidates`**: make the candidate browser
   compute the total from `cv_similarity` + the job's rubric (ad, criteria,
   feedback) instead of reading `candidates.points` / `candidate_scores`.

The GUI side can stay thin either way; the important part is that the score
the page renders is the job's actual rubric score, with the correct
denominator.

## Related (separate) issue: quick model shows 0/N

The waves pipeline's quick/scout model is
`static-similarity-mrl-multilingual-v1`
(`talentrankiq_cli/lib/libconfigreader/src/config.rs:483`). When it is not on
disk, `ensure_embeddings` silently returns `Ok(0)`
(`talentrankiq_cli/lib/libprocess-in-waves/src/embed.rs:47-50`), so the
progress card can show `Quick embedded 0/N` forever. This is a distinct
problem from the score write-back: it only affects the quick counter and is
avoided by downloading the scout model from the Models page (Scout model
card). It is noted here because it produces a similar "the page is stuck"
impression.