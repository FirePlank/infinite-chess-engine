//! The native bench workloads (nps_bench's search sweep, eval_bench's corpus)
//! exported to JS, so a wasm build can be timed and checked for identity: its
//! node counts and eval checksum must equal the native ones exactly.

use apeiron::Variant;
use apeiron::game::GameState;
use apeiron::search;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

const SEARCH_VARIANTS: [Variant; 15] = [
    Variant::CoaIPHO,
    Variant::CoaIPRO,
    Variant::CoaIPNO,
    Variant::ScatteredLeapers,
    Variant::Classical,
    Variant::CoaIP,
    Variant::Space,
    Variant::Palace,
    Variant::Knightline,
    Variant::Core,
    Variant::Standarch,
    Variant::Pawndard,
    Variant::ConfinedClassical,
    Variant::ClassicalPlus,
    Variant::Chess,
];

const EVAL_VARIANTS: [Variant; 22] = [
    Variant::Classical,
    Variant::CoaIP,
    Variant::CoaIPHO,
    Variant::CoaIPRO,
    Variant::CoaIPNO,
    Variant::Space,
    Variant::SpaceClassic,
    Variant::Palace,
    Variant::Knightline,
    Variant::Core,
    Variant::Standarch,
    Variant::Pawndard,
    Variant::ConfinedClassical,
    Variant::ClassicalPlus,
    Variant::ScatteredLeapers,
    Variant::Abundance,
    // The specialized evaluators and their nets.
    Variant::Chess,
    Variant::Obstocean,
    Variant::PawnHorde,
    Variant::DoubleKingClassical,
    Variant::TripleKingMaze,
    Variant::AllPiecesClassical,
];

fn setup(v: Variant) -> GameState {
    let mut g = GameState::new();
    g.setup_position_from_icn(v.starting_icn());
    g.variant = Some(v);
    g.game_rules.variant = Some(v);
    g
}

fn lcg(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Deterministic capture-leaning walk, identical to nps_bench/eval_bench.
fn walk(game: &mut GameState, plies: usize, seed0: u64, mut visit: impl FnMut(&GameState)) {
    apeiron::moves::set_slider_cache_bypass(true);
    let mut seed = seed0;
    for _ in 0..plies {
        visit(game);
        let moves = game.get_pseudo_legal_moves();
        if moves.is_empty() {
            break;
        }
        seed = lcg(seed);
        let caps: Vec<_> = moves
            .iter()
            .filter(|m| game.board.get_piece(m.to.x, m.to.y).is_some())
            .copied()
            .collect();
        let pool = if !caps.is_empty() && (seed >> 60).is_multiple_of(2) {
            &caps[..]
        } else {
            &moves[..]
        };
        let mv = pool[(seed >> 33) as usize % pool.len()];
        game.make_move(&mv);
    }
    apeiron::moves::set_slider_cache_bypass(false);
}

/// nps_bench's sweep: opening and walked middlegame per variant, fixed depth.
/// `only` filters variants by substring, as NPS_ONLY does. Returns total nodes.
#[wasm_bindgen]
pub fn search_nodes(depth: u32, only: &str) -> f64 {
    let vs: Vec<Variant> = SEARCH_VARIANTS
        .into_iter()
        .filter(|v| only.is_empty() || v.to_str().contains(only))
        .collect();
    let mut nodes = 0u64;
    for (i, v) in vs.into_iter().enumerate() {
        let mut g = setup(v);
        search::reset_search_state();
        if let Some((_, _, s)) = search::get_best_move(&mut g, depth as usize, 600_000, true, false) {
            nodes += s.nodes;
        }
        let mut g = setup(v);
        search::reset_search_state();
        walk(&mut g, 24, 0x9E3779B97F4A7C15 ^ (i as u64 + 1), |_| {});
        if let Some((_, _, s)) = search::get_best_move(&mut g, depth as usize, 600_000, true, false) {
            nodes += s.nodes;
        }
    }
    nodes as f64
}

thread_local! {
    static CORPUS: RefCell<Vec<(Variant, Vec<GameState>)>> = const { RefCell::new(Vec::new()) };
}

/// eval_bench's corpus (16 variants x a 40-ply walk) plus the specialized evaluators.
fn with_corpus<R>(f: impl FnOnce(&[(Variant, Vec<GameState>)]) -> R) -> R {
    CORPUS.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_empty() {
            for (vi, v) in EVAL_VARIANTS.into_iter().enumerate() {
                let mut g = setup(v);
                let mut states = Vec::with_capacity(40);
                walk(&mut g, 40, 0x9E3779B97F4A7C15 ^ (vi as u64 + 1), |s| states.push(s.clone()));
                c.push((v, states));
            }
        }
        f(&c)
    })
}

/// Sum of evaluate() over `passes` passes of the corpus. One pass is the identity
/// checksum; more passes are the eval timing loop. Wrapped to i32 so it stays exact
/// through f64.
#[wasm_bindgen]
pub fn eval_sum(passes: u32) -> f64 {
    with_corpus(|corpus| {
        let mut sum = 0i64;
        for (v, states) in corpus {
            // Each variant's bounds are process-global, so re-set them before its states.
            let g = setup(*v);
            drop(g);
            for _ in 0..passes {
                for s in states {
                    sum = sum.wrapping_add(apeiron::evaluation::evaluate(s) as i64);
                }
            }
        }
        sum as i32 as f64
    })
}

/// The net's SIMD dense layer against its scalar reference, inside this build.
#[wasm_bindgen]
pub fn net_selftest() -> bool {
    apeiron::eval_net::kernel_selftest()
}

/// Native reference for the same build features:
/// `cargo test --release -- --ignored --nocapture` (add `--features mt` for the MT build).
#[cfg(test)]
mod tests {
    #[test]
    #[ignore]
    fn native_reference() {
        assert!(super::net_selftest());
        println!("search_nodes(10)={} eval_sum(1)={}", super::search_nodes(10, ""), super::eval_sum(1));
    }
}
