//! Pure rubric scoring + feedback-learning math (no DB, no network).
//!
//! ## Rubric
//! Each "question" (job ad, each criterion, the per-job feedback vector, the cross-job
//! profile vector) is a graded item with a max-points weight. A candidate's points on a
//! question = `weight · cosine(cv_vector, question_vector)`. The candidate's **total =
//! Σ points**. Because the embedder returns unit vectors and cosine is scale-invariant,
//! this total is exactly the legacy single `match_score` (ranking is preserved); the
//! rubric simply decomposes it for display.
//!
//! ## Feedback learning
//! Two learned vectors — `ejobfeedback` (per job) and `eprofile` (cross-job) — start at
//! zero and are updated from reorders and likes via [`compute_feedback_delta`]. They
//! become two more rubric questions whose points contribute to the total.

use std::collections::HashMap;

use crate::embed::{add_weighted, centroid, cosine};
use crate::types::{QuestionKind, RubricScore};

/// A criterion's contribution inputs to the rubric.
#[derive(Clone)]
pub struct CriterionVec<'a> {
    pub id: String,
    pub text: String,
    pub weight: f32,
    pub vector: &'a [f32],
}

/// Build one rubric row for a weighted question.
fn row(
    question_id: &str,
    label: &str,
    kind: QuestionKind,
    ecv: &[f32],
    vec: &[f32],
    max_points: f32,
) -> RubricScore {
    RubricScore {
        question_id: question_id.to_string(),
        label: label.to_string(),
        kind,
        points: max_points * cosine(ecv, vec),
        max_points,
        chunk_explanations: vec![],
    }
}

/// Build a candidate's per-question rubric scores.
///
/// Order: job ad, then criteria (in input order), then the two feedback rows. The ad row
/// is omitted when `ead` is `None`; a feedback row is omitted when its vector is absent or
/// its weight is 0 (nothing learned / not yet enabled) so the breakdown stays clean.
pub fn rubric_scores(
    ecv: &[f32],
    ead: Option<(&[f32], f32)>,
    criteria: &[CriterionVec<'_>],
    ejobfeedback: Option<(&[f32], f32)>,
    eprofile: Option<(&[f32], f32)>,
) -> Vec<RubricScore> {
    let mut out = Vec::new();
    if let Some((vec, w)) = ead {
        out.push(row("ad", "Job ad", QuestionKind::Ad, ecv, vec, w));
    }
    for c in criteria {
        out.push(row(
            &c.id,
            &c.text,
            QuestionKind::Criterion,
            ecv,
            c.vector,
            c.weight,
        ));
    }
    if let Some((vec, w)) = ejobfeedback {
        if w > 0.0 {
            out.push(row(
                "fb_job",
                "Feedback (this job)",
                QuestionKind::FeedbackJob,
                ecv,
                vec,
                w,
            ));
        }
    }
    if let Some((vec, w)) = eprofile {
        if w > 0.0 {
            out.push(row(
                "fb_profile",
                "Feedback (all jobs)",
                QuestionKind::FeedbackProfile,
                ecv,
                vec,
                w,
            ));
        }
    }
    out
}

/// A candidate's total rubric points (sum of per-question points).
pub fn total_points(scores: &[RubricScore]) -> f32 {
    scores.iter().map(|s| s.points).sum()
}

/// The rubric's total capacity (sum of per-question max-points).
pub fn max_points(scores: &[RubricScore]) -> f32 {
    scores.iter().map(|s| s.max_points.abs()).sum()
}

// ---------------------------------------------------------------------------
// feedback learning
// ---------------------------------------------------------------------------

/// Feedback learning rates.
pub const LR_JOB: f32 = 0.5;
pub const LR_PROFILE: f32 = 0.2;
pub const LR_LIKE: f32 = 0.3;

/// Per-Apply updates to the two feedback vectors (added to the stored vectors by the
/// caller, which then normalizes them). Both live in embedding space.
pub struct FeedbackDelta {
    pub ejobfeedback: Vec<f32>,
    pub eprofile: Vec<f32>,
}

/// Compute the feedback-vector update from a staged reorder + like/unlike set.
///
/// `baseline` and `staged` are the triqui as `(id, vector)` pairs, in stored-rank
/// order and in the user's reordered order respectively (same membership). A candidate
/// that moved up (staged index < baseline index) is a *gainer*; moved down is a *loser*;
/// unchanged is *stable*. Since the reorderable surface is itself the top-K, "top-K
/// stables" collapse to all stables.
///
/// `edelta = mean(gainers ∪ stables) − mean(losers)`, defined only when there is at least
/// one loser (i.e. a reorder actually happened) — otherwise it is zero. Likes add
/// `LR_LIKE · v` per newly-liked CV and subtract per newly-unliked. The two vectors share
/// the same direction but scale by their respective reorder learning rate.
pub fn compute_feedback_delta(
    baseline: &[(&str, &[f32])],
    staged: &[(&str, &[f32])],
    added_likes: &[&[f32]],
    removed_likes: &[&[f32]],
    dim: usize,
) -> FeedbackDelta {
    let mut baseline_idx: HashMap<&str, usize> = HashMap::new();
    for (i, item) in baseline.iter().enumerate() {
        baseline_idx.insert(item.0, i);
    }

    let mut gainers: Vec<&[f32]> = Vec::new();
    let mut losers: Vec<&[f32]> = Vec::new();
    let mut stables: Vec<&[f32]> = Vec::new();
    for (staged_i, item) in staged.iter().enumerate() {
        let id = item.0;
        let vec = item.1;
        match baseline_idx.get(id).copied() {
            Some(base_i) if staged_i < base_i => gainers.push(vec),
            Some(base_i) if staged_i > base_i => losers.push(vec),
            _ => stables.push(vec),
        }
    }

    // mean1 = centroid(gainers ∪ stables); mean2 = centroid(losers).
    let mut anchor: Vec<&[f32]> = gainers;
    anchor.extend(stables.iter().copied());
    let mean1 = centroid(&anchor);
    let mean2 = centroid(&losers);

    let mut edelta = vec![0.0_f32; dim];
    if let (Some(m1), Some(m2)) = (&mean1, &mean2) {
        add_weighted(&mut edelta, m1.as_slice(), 1.0);
        add_weighted(&mut edelta, m2.as_slice(), -1.0);
    }

    let mut like = vec![0.0_f32; dim];
    for v in added_likes {
        add_weighted(&mut like, v, LR_LIKE);
    }
    for v in removed_likes {
        add_weighted(&mut like, v, -LR_LIKE);
    }

    let mut ejobfeedback = vec![0.0_f32; dim];
    add_weighted(&mut ejobfeedback, &edelta, LR_JOB);
    add_weighted(&mut ejobfeedback, &like, 1.0);

    let mut eprofile = vec![0.0_f32; dim];
    add_weighted(&mut eprofile, &edelta, LR_PROFILE);
    add_weighted(&mut eprofile, &like, 1.0);

    FeedbackDelta {
        ejobfeedback,
        eprofile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{cosine, Embedder, MockEmbedder};

    fn dim() -> usize {
        32
    }

    #[test]
    fn rubric_scores_ad_and_criteria_sum_to_total() {
        let e = MockEmbedder::new(dim());
        let ad = e.embed(&["backend engineer"]).unwrap()[0].clone();
        let cv = e.embed(&["backend engineer with aws"]).unwrap()[0].clone();
        let pos_vec = e.embed(&["aws cloud"]).unwrap()[0].clone();
        let crit = CriterionVec {
            id: "c1".into(),
            text: "aws".into(),
            weight: 5.0,
            vector: &pos_vec,
        };
        let scores = rubric_scores(&cv, Some((&ad, 20.0)), &[crit], None, None);
        assert_eq!(scores.len(), 2, "1 ad + 1 criterion, no feedback rows");
        assert!((scores[0].max_points - 20.0).abs() < 1e-4);
        assert!((scores[0].points - 20.0 * cosine(&cv, &ad)).abs() < 1e-4);
        assert!(
            (total_points(&scores) - scores.iter().map(|s| s.points).sum::<f32>()).abs() < 1e-4
        );
        assert!((max_points(&scores) - 25.0).abs() < 1e-4); // 20 ad + 5 crit
    }

    #[test]
    fn feedback_rows_omitted_until_learned() {
        let e = MockEmbedder::new(dim());
        let ad = e.embed(&["engineer"]).unwrap()[0].clone();
        let cv = e.embed(&["engineer"]).unwrap()[0].clone();
        // No feedback vectors -> only the ad row.
        assert_eq!(
            rubric_scores(&cv, Some((&ad, 1.0)), &[], None, None).len(),
            1
        );

        // A feedback vector present but weight 0 -> still omitted (no 0/0 noise).
        let fb = e.embed(&["liked trait"]).unwrap()[0].clone();
        assert_eq!(
            rubric_scores(&cv, Some((&ad, 1.0)), &[], Some((&fb, 0.0)), None).len(),
            1
        );

        // Learned + weighted -> row present.
        assert_eq!(
            rubric_scores(&cv, Some((&ad, 1.0)), &[], Some((&fb, 10.0)), None).len(),
            2
        );
    }

    #[test]
    fn tier1_total_orders_aligned_above_orthogonal() {
        // With feedback vectors absent, the rubric total is on the legacy match-score
        // scale: an aligned CV outscores an orthogonal one.
        let _dim = 4;
        let ad = vec![1.0, 0.0, 0.0, 0.0];
        let cv_aligned = vec![1.0, 0.0, 0.0, 0.0];
        let cv_orth = vec![0.0, 1.0, 0.0, 0.0];
        let s_a = total_points(&rubric_scores(
            &cv_aligned,
            Some((&ad, 1.0)),
            &[],
            None,
            None,
        ));
        let s_o = total_points(&rubric_scores(&cv_orth, Some((&ad, 1.0)), &[], None, None));
        assert!(s_a > s_o);
        assert!((s_a - 1.0).abs() < 1e-4);
        assert!(s_o.abs() < 1e-4);
    }

    #[test]
    fn delta_no_change_is_zero() {
        let e = MockEmbedder::new(dim());
        let a = e.embed(&["alpha"]).unwrap()[0].clone();
        let b = e.embed(&["beta"]).unwrap()[0].clone();
        let baseline: Vec<(&str, &[f32])> = vec![("x", &a), ("y", &b)];
        let staged: Vec<(&str, &[f32])> = vec![("x", &a), ("y", &b)];
        let d = compute_feedback_delta(&baseline, &staged, &[], &[], dim());
        assert!(d.ejobfeedback.iter().all(|v| v.abs() < 1e-6));
        assert!(d.eprofile.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn delta_reorder_pushes_toward_gainer() {
        let e = MockEmbedder::new(dim());
        let a = e.embed(&["alpha"]).unwrap()[0].clone();
        let b = e.embed(&["beta"]).unwrap()[0].clone();
        // baseline a,b -> staged b,a: b gains, a loses.
        let baseline: Vec<(&str, &[f32])> = vec![("a", &a), ("b", &b)];
        let staged: Vec<(&str, &[f32])> = vec![("b", &b), ("a", &a)];
        let d = compute_feedback_delta(&baseline, &staged, &[], &[], dim());
        assert!(
            cosine(&d.ejobfeedback, &b) > cosine(&d.ejobfeedback, &a),
            "feedback should lean toward the gainer"
        );
    }

    #[test]
    fn delta_like_aligns_with_liked_vector() {
        let e = MockEmbedder::new(dim());
        let a = e.embed(&["alpha"]).unwrap()[0].clone();
        let baseline: Vec<(&str, &[f32])> = vec![("a", &a)];
        let staged: Vec<(&str, &[f32])> = vec![("a", &a)];
        let liked = e.embed(&["liked trait"]).unwrap()[0].clone();
        let d = compute_feedback_delta(&baseline, &staged, &[&liked], &[], dim());
        assert!(
            cosine(&d.ejobfeedback, &liked) > 0.99,
            "like should push the feedback vector toward the liked CV"
        );
    }
}
