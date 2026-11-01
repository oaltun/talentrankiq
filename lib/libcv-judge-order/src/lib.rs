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
    // Priority (a): new CVs (no comparisons yet) needing first comparison.
    for (cv, _) in &state.provisional_sim_rank {
        let has = state.comparisons.iter().any(|(a, b)| a == cv || b == cv);
        if has {
            continue;
        }
        if let Some(opp) = first_opponent(cv.clone(), &state.provisional_sim_rank, &state.comparisons) {
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
    // Priority (b): CVs needing exploration opponents.
    for (cv, _) in &state.provisional_sim_rank {
        let opps = exploration_opponents(
            cv.clone(),
            &state.glicko,
            state.job_version,
            &state.comparisons,
            1,
        );
        if let Some(opp) = opps.into_iter().next() {
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
    // Priority (c): top-of-leaderboard pairs not yet compared.
    for i in 0..state.provisional_sim_rank.len() {
        for j in (i + 1)..state.provisional_sim_rank.len() {
            let a = state.provisional_sim_rank[i].0.clone();
            let b = state.provisional_sim_rank[j].0.clone();
            let key = if a <= b { (a, b) } else { (b, a) };
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_opponent_picks_sim_neighbor() {
        let rank = vec![("a".to_string(), 0.9), ("b".to_string(), 0.8), ("c".to_string(), 0.7)];
        let compared = HashSet::new();
        let opp = first_opponent("a".into(), &rank, &compared).unwrap();
        assert_eq!(opp, "b");
    }

    #[test]
    fn first_opponent_skips_compared() {
        let rank = vec![("a".to_string(), 0.9), ("b".to_string(), 0.8), ("c".to_string(), 0.7)];
        let mut compared = HashSet::new();
        compared.insert(("a".to_string(), "b".to_string()));
        let opp = first_opponent("a".into(), &rank, &compared).unwrap();
        assert_eq!(opp, "c");
    }

    #[test]
    fn exploration_opponents_picks_elo_neighbors() {
        let mut g = HashMap::new();
        g.insert(("a".to_string(), 1u32), GlickoState { rating: 1000.0, rd: 100.0 });
        g.insert(("b".to_string(), 1u32), GlickoState { rating: 1010.0, rd: 100.0 });
        g.insert(("c".to_string(), 1u32), GlickoState { rating: 1500.0, rd: 100.0 });
        let compared = HashSet::new();
        let opps = exploration_opponents("a".into(), &g, 1, &compared, 1);
        assert_eq!(opps, vec!["b".to_string()]);
    }

    #[test]
    fn next_pair_prioritizes_uncompared() {
        let state = JudgeOrderState {
            glicko: HashMap::new(),
            comparisons: HashSet::new(),
            provisional_sim_rank: vec![("a".to_string(), 0.9), ("b".to_string(), 0.8)],
            job_version: 1,
        };
        let p = next_pair(&state).unwrap();
        assert_eq!(p, ("a".to_string(), "b".to_string()));
    }

    #[test]
    fn h2h_status_variants() {
        let mut state = JudgeOrderState {
            glicko: HashMap::new(),
            comparisons: HashSet::new(),
            provisional_sim_rank: vec![],
            job_version: 1,
        };
        assert_eq!(h2h_status("a".into(), &state), H2HStatus::Pending);
        state.comparisons.insert(("a".to_string(), "b".to_string()));
        assert_eq!(h2h_status("a".into(), &state), H2HStatus::Live);
    }
}
