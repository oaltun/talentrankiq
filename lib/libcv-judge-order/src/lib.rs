use std::collections::{HashMap, HashSet};

use libjudge::tournament::GlickoState;

pub type CvId = String;
pub type PairKey = (CvId, CvId);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H2HStatus {
    Live,
    Frozen,
    Pending,
    Off,
}

#[derive(Debug, Clone)]
pub struct JudgeOrderState {
    pub glicko: HashMap<(CvId, u32), GlickoState>,
    pub comparisons: HashSet<PairKey>,
    pub provisional_sim_rank: Vec<(CvId, f64)>,
    pub job_version: u32,
}

pub fn first_opponent(
    new_cv: CvId,
    provisional_sim_rank: &[(CvId, f64)],
    already_compared: &HashSet<PairKey>,
) -> Option<CvId> {
    let idx = provisional_sim_rank.iter().position(|(cv, _)| cv == &new_cv);
    // Try nearest neighbors in sim-rank order (both directions), expanding outward.
    let order: Vec<usize> = match idx {
        Some(i) => {
            let mut v = Vec::new();
            let mut d = 1usize;
            while i + d < provisional_sim_rank.len() || i >= d {
                if i + d < provisional_sim_rank.len() {
                    v.push(i + d);
                }
                if i >= d {
                    v.push(i - d);
                }
                d += 1;
            }
            v
        }
        None => (0..provisional_sim_rank.len()).collect(),
    };
    for i in order {
        let cv = &provisional_sim_rank[i].0;
        if cv == &new_cv {
            continue;
        }
        let key = if new_cv <= *cv {
            (new_cv.clone(), cv.clone())
        } else {
            (cv.clone(), new_cv.clone())
        };
        if already_compared.contains(&key) {
            continue;
        }
        return Some(cv.clone());
    }
    None
}

pub fn exploration_opponents(
    cv: CvId,
    glicko_state: &HashMap<(CvId, u32), GlickoState>,
    job_version: u32,
    already_compared: &HashSet<PairKey>,
    budget: usize,
) -> Vec<CvId> {
    let player = glicko_state
        .get(&(cv.clone(), job_version))
        .copied()
        .unwrap_or_default();
    let mut others: Vec<(CvId, f64)> = glicko_state
        .iter()
        .filter(|((id, ver), _)| *ver == job_version && *id != cv)
        .map(|((id, _), st)| (id.clone(), (st.rating - player.rating).abs()))
        .collect();
    others.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::new();
    for (id, _) in others {
        if out.len() >= budget {
            break;
        }
        let key = if cv <= id {
            (cv.clone(), id.clone())
        } else {
            (id.clone(), cv.clone())
        };
        if already_compared.contains(&key) {
            continue;
        }
        out.push(id);
    }
    out
}

pub fn next_pair(state: &JudgeOrderState) -> Option<PairKey> {
    // Priority: new CVs needing first comparison -> exploration -> top-of-leaderboard.
    for (cv, _) in &state.provisional_sim_rank {
        let already: HashSet<PairKey> = state.comparisons.clone();
        if let Some(opp) = first_opponent(cv.clone(), &state.provisional_sim_rank, &already) {
            let key = if *cv <= opp {
                (cv.clone(), opp)
            } else {
                (opp, cv.clone())
            };
            if !state.comparisons.contains(&key) {
                return Some(key);
            }
        }
    }
    None
}

pub fn h2h_status(cv: CvId, state: &JudgeOrderState) -> H2HStatus {
    let has_comparison = state.comparisons.iter().any(|(a, b)| a == &cv || b == &cv);
    let entitled = true;
    if entitled && has_comparison {
        H2HStatus::Live
    } else if entitled {
        H2HStatus::Pending
    } else if has_comparison {
        H2HStatus::Frozen
    } else {
        H2HStatus::Off
    }
}