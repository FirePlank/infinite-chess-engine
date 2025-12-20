use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

pub mod attacks;
pub mod board;
pub mod evaluation;
pub mod game;
pub mod moves;
pub mod search;
pub mod simd;
pub mod tiles;
mod utils;

// Initialize panic hook for better error messages in WASM
// This will show actual line numbers instead of just "unreachable"
#[cfg(feature = "debug")]
#[wasm_bindgen(start)]
fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

use crate::moves::{SpatialIndices, set_world_bounds};
use board::{Board, Coordinate, Piece, PieceType, PlayerColor};
use evaluation::calculate_initial_material;
use game::{EnPassantState, GameState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Classical,
    ConfinedClassical,
    ClassicalPlus,
    CoaIP,
    CoaIPHO,
    CoaIPRO,
    CoaIPNO,
    Palace,
    Pawndard,
    Core,
    Standarch,
    SpaceClassic,
    Space,
    Abundance,
    PawnHorde,
    Knightline,
    Obstocean,
    Chess,
}

impl Variant {
    pub fn from_str(s: &str) -> Self {
        match s {
            "Classical" => Variant::Classical,
            "Confined_Classical" => Variant::ConfinedClassical,
            "Classical_Plus" => Variant::ClassicalPlus,
            "CoaIP" => Variant::CoaIP,
            "CoaIP_HO" => Variant::CoaIPHO,
            "CoaIP_RO" => Variant::CoaIPRO,
            "CoaIP_NO" => Variant::CoaIPNO,
            "Palace" => Variant::Palace,
            "Pawndard" => Variant::Pawndard,
            "Core" => Variant::Core,
            "Standarch" => Variant::Standarch,
            "Space_Classic" => Variant::SpaceClassic,
            "Space" => Variant::Space,
            "Abundance" => Variant::Abundance,
            "Pawn_Horde" => Variant::PawnHorde,
            "Knightline" => Variant::Knightline,
            "Obstocean" => Variant::Obstocean,
            "Chess" => Variant::Chess,
            _ => Variant::Classical, // Default fallback
        }
    }
}

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

// ============================================================================
// Shared TT WASM Bindings (for Lazy SMP with SharedArrayBuffer)
// ============================================================================

/// Size of the shared TT in u64 words (32MB = 4M words at 8 bytes each)
#[cfg(feature = "multithreading")]
const SHARED_TT_SIZE_WORDS: usize = 32 * 1024 * 1024 / 8;

/// Size of work queue in u64 words (header + 256 moves * 6 words each)
#[cfg(feature = "multithreading")]
const WORK_QUEUE_SIZE_WORDS: usize = 6 + 256 * 6; // ~12KB

/// Static buffer for shared TT - lives in WASM linear memory
/// When WASM memory is backed by SharedArrayBuffer, all workers share this
#[cfg(feature = "multithreading")]
static mut SHARED_TT_BUFFER: [u64; SHARED_TT_SIZE_WORDS] = [0u64; SHARED_TT_SIZE_WORDS];

/// Static buffer for work queue - for root move splitting
#[cfg(feature = "multithreading")]
static mut SHARED_WORK_QUEUE: [u64; WORK_QUEUE_SIZE_WORDS] = [0u64; WORK_QUEUE_SIZE_WORDS];

/// Get the pointer to the shared TT buffer in WASM memory.
/// JavaScript can use this with the WASM memory buffer to share between workers.
#[cfg(feature = "multithreading")]
#[wasm_bindgen]
pub fn get_shared_tt_ptr() -> u32 {
    unsafe { SHARED_TT_BUFFER.as_ptr() as u32 }
}

/// Get the size of the shared TT buffer in u64 words.
#[cfg(feature = "multithreading")]
#[wasm_bindgen]
pub fn get_shared_tt_size() -> u32 {
    SHARED_TT_SIZE_WORDS as u32
}

/// Initialize the shared TT view in search module.
/// Call this after WASM is loaded to set up TT for search.
#[cfg(feature = "multithreading")]
#[wasm_bindgen]
pub fn init_shared_tt() {
    let ptr = unsafe { SHARED_TT_BUFFER.as_mut_ptr() };
    let len = SHARED_TT_SIZE_WORDS;

    // Store in the search module's thread-local state
    search::set_shared_tt_ptr(ptr, len);

    // Also initialize work queue
    let wq_ptr = unsafe { SHARED_WORK_QUEUE.as_mut_ptr() };
    let wq_len = WORK_QUEUE_SIZE_WORDS;
    search::set_shared_work_queue_ptr(wq_ptr, wq_len);

    log(&format!(
        "[WASM] Shared TT initialized: {} words ({} MB) at {:p}",
        len,
        (len * 8) / (1024 * 1024),
        ptr
    ));
}

#[derive(Serialize, Deserialize)]
pub struct JsMove {
    pub from: String, // "x,y"
    pub to: String,   // "x,y"
    pub promotion: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct JsMoveWithEval {
    pub from: String, // "x,y"
    pub to: String,   // "x,y"
    pub promotion: Option<String>,
    pub eval: i32,    // centipawn score from side-to-move's perspective
    pub depth: usize, // depth reached
}

/// A single PV line for MultiPV output
#[derive(Serialize, Deserialize)]
pub struct JsPVLine {
    pub from: String, // "x,y"
    pub to: String,   // "x,y"
    pub promotion: Option<String>,
    pub eval: i32,       // centipawn score from side-to-move's perspective
    pub depth: usize,    // depth searched
    pub pv: Vec<String>, // full PV as array of "x,y->x,y" strings
}

#[derive(Deserialize)]
struct JsFullGame {
    board: JsBoard,
    turn: String,
    /// All special rights - includes castling (kings/rooks) AND pawn double-move rights
    #[serde(default)]
    special_rights: Vec<String>,
    en_passant: Option<JsEnPassant>,
    halfmove_clock: u32,
    fullmove_number: u32,
    #[serde(default)]
    move_history: Vec<JsMoveHistory>,
    #[serde(default)]
    game_rules: Option<JsGameRules>,
    #[serde(default)]
    world_bounds: Option<JsWorldBounds>,
    #[serde(default)]
    clock: Option<JsClock>,
    #[serde(default)]
    variant: Option<String>,
    /// Optional strength hint from the UI/JS side (1=Relaxed, 2=Standard, 3=Maximum).
    #[serde(default)]
    strength_level: Option<u32>,
}

#[derive(Deserialize, Default)]
struct JsGameRules {
    #[serde(default)]
    promotion_ranks: Option<JsPromotionRanks>,
    #[serde(default)]
    promotions_allowed: Option<Vec<String>>,
    #[serde(default)]
    move_rule: Option<u32>,
}

#[derive(Deserialize)]
struct JsPromotionRanks {
    white: Vec<String>, // String because BigInt serializes as string
    black: Vec<String>,
}

#[derive(Deserialize)]
struct JsWorldBounds {
    left: String,
    right: String,
    bottom: String,
    top: String,
}

#[derive(Deserialize, Clone, Copy)]
struct JsClock {
    /// Remaining time for White in milliseconds
    wtime: u64,
    /// Remaining time for Black in milliseconds
    btime: u64,
    /// Increment for White in milliseconds
    winc: u64,
    /// Increment for Black in milliseconds
    binc: u64,
}

#[derive(Deserialize)]
struct JsMoveHistory {
    from: String, // "x,y"
    to: String,   // "x,y"
    #[serde(default)]
    promotion: Option<String>,
}

#[derive(Deserialize)]
struct JsBoard {
    pieces: Vec<JsPiece>,
}

#[derive(Deserialize)]
struct JsPiece {
    x: String,
    y: String,
    piece_type: String,
    player: String,
}

#[derive(Deserialize)]
struct JsEnPassant {
    square: String,      // "x,y"
    pawn_square: String, // "x,y"
}

#[cfg(feature = "eval_tuning")]
#[derive(Serialize)]
struct JsEvalWithFeatures {
    eval: i32,
    features: crate::evaluation::EvalFeatures,
}

#[wasm_bindgen]
pub struct Engine {
    game: GameState,
    clock: Option<JsClock>,
    strength_level: Option<u32>,
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(json_state: JsValue) -> Result<Engine, JsValue> {
        // Initialize magic bitboards for O(1) slider attacks
        // crate::tiles::magic::init();

        let js_game: JsFullGame = serde_wasm_bindgen::from_value(json_state)?;

        // If this looks like a fresh game, clear any persistent search/TT state.
        if js_game.move_history.is_empty() && js_game.fullmove_number <= 1 {
            crate::search::reset_search_state();
        }

        // Apply world bounds from playableRegion if provided
        if let Some(wb) = &js_game.world_bounds {
            let left = wb.left.parse::<i64>().unwrap_or(-1_000_000_000_000_000);
            let right = wb.right.parse::<i64>().unwrap_or(1_000_000_000_000_000);
            let bottom = wb.bottom.parse::<i64>().unwrap_or(-1_000_000_000_000_000);
            let top = wb.top.parse::<i64>().unwrap_or(1_000_000_000_000_000);
            set_world_bounds(left, right, bottom, top);
        }

        // Build starting GameState from JS board
        let mut board = Board::new();
        for p in &js_game.board.pieces {
            let x: i64 =
                p.x.parse()
                    .map_err(|_| JsValue::from_str("Invalid X coordinate"))?;
            let y: i64 =
                p.y.parse()
                    .map_err(|_| JsValue::from_str("Invalid Y coordinate"))?;

            let piece_type = PieceType::from_str(&p.piece_type).unwrap_or(PieceType::Pawn);

            let color = PlayerColor::from_str(&p.player).unwrap_or(PlayerColor::White);

            board.set_piece(x, y, Piece::new(piece_type, color));
        }

        // Starting side (color that moved first) as reported by JS. The engine
        // will reconstruct the current side-to-move by replaying move_history.
        let js_turn = PlayerColor::from_str(&js_game.turn).unwrap_or(PlayerColor::White);

        // Parse initial special rights (castling + pawn double-move)
        let mut special_rights = FxHashSet::default();
        for sr in js_game.special_rights {
            let parts: Vec<&str> = sr.split(',').collect();
            if parts.len() == 2 {
                if let (Ok(x), Ok(y)) = (parts[0].parse::<i64>(), parts[1].parse::<i64>()) {
                    special_rights.insert(Coordinate::new(x, y));
                }
            }
        }

        // Parse en passant directly as i64 (used only when there is no move history)
        let parsed_en_passant = if let Some(ep) = js_game.en_passant {
            let sq_parts: Vec<&str> = ep.square.split(',').collect();
            let pawn_parts: Vec<&str> = ep.pawn_square.split(',').collect();

            if sq_parts.len() == 2 && pawn_parts.len() == 2 {
                if let (Ok(sq_x), Ok(sq_y), Ok(pawn_x), Ok(pawn_y)) = (
                    sq_parts[0].parse::<i64>(),
                    sq_parts[1].parse::<i64>(),
                    pawn_parts[0].parse::<i64>(),
                    pawn_parts[1].parse::<i64>(),
                ) {
                    Some(EnPassantState {
                        square: Coordinate::new(sq_x, sq_y),
                        pawn_square: Coordinate::new(pawn_x, pawn_y),
                    })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Parse game rules from JS
        let game_rules = if let Some(js_rules) = js_game.game_rules {
            use game::{GameRules, PromotionRanks};

            let promotion_ranks = js_rules.promotion_ranks.map(|pr| PromotionRanks {
                white: pr
                    .white
                    .iter()
                    .filter_map(|s| s.parse::<i64>().ok())
                    .collect(),
                black: pr
                    .black
                    .iter()
                    .filter_map(|s| s.parse::<i64>().ok())
                    .collect(),
            });

            let mut rules = GameRules {
                promotion_ranks,
                promotion_types: None,
                promotions_allowed: js_rules.promotions_allowed,
                move_rule_limit: js_rules.move_rule,
            };
            rules.init_promotion_types();
            rules
        } else {
            game::GameRules::default()
        };

        // Precompute effective promotion ranks and dynamic back ranks once per
        // game from promotion_ranks. For standard chess this yields promo
        // ranks 8/1 and back ranks 1/8.
        let (white_promo_rank, black_promo_rank, white_back_rank, black_back_rank) =
            if let Some(ref ranks) = game_rules.promotion_ranks {
                let white_promo = ranks.white.iter().copied().max().unwrap_or(8);
                let black_promo = ranks.black.iter().copied().min().unwrap_or(1);

                // White's home side is near Black's promotion ranks, and vice versa.
                let wb = black_promo; // white back rank
                let bb = white_promo; // black back rank

                (white_promo, black_promo, wb, bb)
            } else {
                // Classical default: white promotes on 8, black on 1; back ranks 1/8.
                (8, 1, 1, 8)
            };

        // Initialize game with starting position; clocks and turn will be fixed below.
        let mut game = GameState {
            board,
            // Seed with the starting side; this ensures that replaying move history
            // produces the correct side-to-move even when Black (or another side)
            // moved first.
            turn: js_turn,
            special_rights,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            material_score: 0,
            game_rules,
            variant: js_game
                .variant
                .as_deref()
                .map(|s| crate::Variant::from_str(s)),
            hash: 0, // Will be computed below
            hash_stack: Vec::with_capacity(js_game.move_history.len().saturating_add(8)),
            null_moves: 0,
            white_piece_count: 0,
            black_piece_count: 0,
            white_pawn_count: 0,
            black_pawn_count: 0,
            starting_white_pieces: 0,
            starting_black_pieces: 0,
            white_pieces: Vec::new(),
            black_pieces: Vec::new(),
            spatial_indices: SpatialIndices::default(),
            starting_squares: FxHashSet::default(),
            white_back_rank,
            black_back_rank,
            white_promo_rank,
            black_promo_rank,
            white_king_pos: None,
            black_king_pos: None,
            pawn_hash: 0,
            nonpawn_hash: 0,
            material_hash: 0,
            repetition: 0,
            white_non_pawn_material: false,
            black_non_pawn_material: false,
        };

        game.material_score = calculate_initial_material(&game.board);
        game.recompute_piece_counts(); // Rebuild piece lists and counts
        game.init_starting_piece_counts(); // Cache starting non-pawn piece counts for phase detection
        // Initialize development starting squares from the initial board
        // before replaying move history.
        game.init_starting_squares();
        game.recompute_hash(); // Compute initial hash from position

        // Helper to parse "x,y" into (i64, i64)
        fn parse_coords(coord_str: &str) -> Option<(i64, i64)> {
            let parts: Vec<&str> = coord_str.split(',').collect();
            if parts.len() != 2 {
                return None;
            }
            let x = parts[0].parse::<i64>().ok()?;
            let y = parts[1].parse::<i64>().ok()?;
            Some((x, y))
        }

        if js_game.move_history.is_empty() {
            // No history: trust JS turn/en-passant for this position
            game.en_passant = parsed_en_passant;
            game.turn = js_turn;
        } else {
            // Replay the full move history from the start position.
            // Like UCI: just apply moves directly by coordinates, no legal move generation needed.
            for hist in &js_game.move_history {
                if let (Some((from_x, from_y)), Some((to_x, to_y))) =
                    (parse_coords(&hist.from), parse_coords(&hist.to))
                {
                    let promo = hist.promotion.as_ref().map(|s| s.as_str());
                    game.make_move_coords(from_x, from_y, to_x, to_y, promo);
                }
            }
            // After replay, GameState.turn, clocks, and en_passant have been
            // updated naturally by make_move_coords.
        }

        // Always use the clocks passed from JS, as they reflect the authoritative state
        // (e.g. edited counters in board editor, or simple synchronization).
        game.halfmove_clock = js_game.halfmove_clock;
        game.fullmove_number = if js_game.fullmove_number == 0 {
            1
        } else {
            js_game.fullmove_number
        };

        // Optional clock information (similar to UCI wtime/btime/winc/binc).
        let clock = js_game.clock;
        let strength_level = js_game.strength_level;

        Ok(Engine {
            game,
            clock,
            strength_level,
        })
    }

    pub fn get_best_move(&mut self) -> JsValue {
        if let Some((best_move, _eval, _stats)) =
            search::get_best_move(&mut self.game, 50, u128::MAX, false)
        {
            let js_move = JsMove {
                from: format!("{},{}", best_move.from.x, best_move.from.y),
                to: format!("{},{}", best_move.to.x, best_move.to.y),
                promotion: best_move.promotion.map(|p| p.to_str().to_string()),
            };
            serde_wasm_bindgen::to_value(&js_move).unwrap()
        } else {
            JsValue::NULL
        }
    }

    #[cfg(feature = "eval_tuning")]
    #[wasm_bindgen]
    pub fn evaluate_with_features(&mut self) -> JsValue {
        crate::evaluation::reset_eval_features();
        let eval = crate::evaluation::evaluate(&self.game);
        let features = crate::evaluation::snapshot_eval_features();
        serde_wasm_bindgen::to_value(&JsEvalWithFeatures { eval, features }).unwrap()
    }

    /// Set search parameters from a JSON string.
    /// Only available when the `search_tuning` feature is enabled.
    /// Returns true on success, false on parse failure.
    #[cfg(feature = "search_tuning")]
    #[wasm_bindgen]
    pub fn set_search_params(&self, json: &str) -> bool {
        crate::search::params::set_search_params_from_json(json)
    }

    /// Get current search parameters as a JSON string.
    /// Only available when the `search_tuning` feature is enabled.
    #[cfg(feature = "search_tuning")]
    #[wasm_bindgen]
    pub fn get_search_params(&self) -> String {
        crate::search::params::get_search_params_as_json()
    }

    /// Return the engine's static evaluation of the current position in centipawns,
    /// from the side-to-move's perspective (positive = advantage for side to move).
    pub fn evaluate_position(&mut self) -> i32 {
        evaluation::evaluate(&self.game)
    }

    /// Derive an effective time limit for this move from the current clock and
    /// game state. When a clock is present (timed game), we ignore the
    /// caller-provided fixed per-move limit and instead base the allocation on
    /// remaining time, increment, and a simple game-phase heuristic.
    ///
    /// When no clock is present (infinite/untimed), we fall back to the
    /// requested per-move limit.
    fn effective_time_limit_ms(&self, requested_limit_ms: u32) -> u128 {
        let Some(clock) = self.clock else {
            // No clock info: respect the fixed per-move limit.
            return requested_limit_ms as u128;
        };

        // Decide which side's clock to use.
        let (remaining_ms_raw, inc_ms_raw) = match self.game.turn {
            PlayerColor::White => (clock.wtime, clock.winc),
            PlayerColor::Black => (clock.btime, clock.binc),
            // Neutral side-to-move should not normally happen; fall back to
            // the requested limit in that case.
            PlayerColor::Neutral => return requested_limit_ms as u128,
        };

        // If there is no usable clock information, fall back to the
        // requested fixed limit.
        if remaining_ms_raw == 0 && inc_ms_raw == 0 {
            return requested_limit_ms as u128;
        }

        // Treat a zero remaining time but positive increment as a very short
        // remaining time budget based mostly on the increment.
        let remaining_ms = if remaining_ms_raw > 0 {
            remaining_ms_raw
        } else {
            // At least give ourselves a small buffer.
            inc_ms_raw.max(500)
        };

        let inc_ms = inc_ms_raw;

        // Crude game phase estimation based on total material count. This
        // does not need to be exact; it only guides relative time allocation.
        let total_pieces: u32 =
            (self.game.white_piece_count as u32).saturating_add(self.game.black_piece_count as u32);

        // Opening: many pieces on the board -> be conservative.
        // Middlegame: spend more.
        // Endgame: spend the most per move (within reason).
        let (moves_to_go, phase_factor): (u64, f64) = if total_pieces > 20 {
            (30, 0.7)
        } else if total_pieces > 10 {
            (20, 1.0)
        } else {
            (10, 1.2)
        };

        let moves_to_go = moves_to_go.max(5);
        let base_per_move = (remaining_ms / moves_to_go).max(10);
        let phase_scaled = (base_per_move as f64 * phase_factor) as u64;
        let inc_contrib = inc_ms / 2;

        let mut alloc = phase_scaled.saturating_add(inc_contrib);

        // Hard caps:
        //  - never spend more than half of the remaining time on a single move
        //  - global cap to keep engine thinking time reasonable in the browser
        let hard_cap_by_remaining = remaining_ms / 2;
        let global_cap_ms: u64 = 15_000; // 15 seconds
        let mut hard_cap = hard_cap_by_remaining.min(global_cap_ms);

        // Ensure the cap is not unreasonably tiny when we still have some time.
        if hard_cap < 250 {
            hard_cap = 250;
        }

        if alloc > hard_cap {
            alloc = hard_cap;
        }

        // Do not go below a tiny minimum, but also don't exceed the sum of
        // remaining time and one increment.
        let min_think_ms: u64 = 50;
        if alloc < min_think_ms {
            alloc = min_think_ms;
        }

        let max_reasonable = remaining_ms.saturating_add(inc_ms);
        if alloc > max_reasonable {
            alloc = max_reasonable.max(min_think_ms);
        }

        alloc as u128
    }

    /// Timed search. This also exposes the search evaluation as an `eval` field alongside the move,
    /// so callers can reuse the same search for adjudication.
    /// thread_id is used for Lazy SMP - helper threads (id > 0) skip the first move.
    #[wasm_bindgen]
    pub fn get_best_move_with_time(
        &mut self,
        time_limit_ms: u32,
        silent: Option<bool>,
        max_depth: Option<usize>,
        noise_amp: Option<i32>,
        thread_id: Option<u32>,
    ) -> JsValue {
        let effective_limit = if time_limit_ms == 0 && max_depth.is_some() {
            // If explicit depth is requested with 0 time, treat as infinite time (fixed depth search)
            u128::MAX
        } else {
            self.effective_time_limit_ms(time_limit_ms)
        };
        let silent = silent.unwrap_or(false);
        let depth = max_depth.unwrap_or(50).max(1).min(50);
        let strength = self.strength_level.unwrap_or(3).max(1).min(3);

        // Determine effective noise amplitude:
        // 1. If explicit noise_amp is provided, use it
        // 2. Otherwise, derive from strength level
        let effective_noise: i32 = if let Some(amp) = noise_amp {
            amp.max(0)
        } else {
            match strength {
                1 => 50,
                2 => 25,
                _ => 0, // strength 3 = no noise
            }
        };

        #[allow(unused_variables)]
        let pre_stats = crate::search::get_current_tt_stats();

        #[cfg(target_arch = "wasm32")]
        {
            if !silent {
                use crate::log;
                let variant = self
                    .game
                    .variant
                    .map_or("unknown".to_string(), |v| format!("{:?}", v));

                let tt_cap = pre_stats.tt_capacity;
                let tt_used = pre_stats.tt_used;
                let tt_fill = pre_stats.tt_fill_permille;

                if let Some(clock) = self.clock {
                    let side = match self.game.turn {
                        PlayerColor::White => "w",
                        PlayerColor::Black => "b",
                        PlayerColor::Neutral => "n",
                    };
                    log(&format!(
                        "info timealloc side {} wtime {} btime {} winc {} binc {} limit {} variant {} tt_cap {} tt_used {} tt_fill {}",
                        side,
                        clock.wtime,
                        clock.btime,
                        clock.winc,
                        clock.binc,
                        effective_limit,
                        variant,
                        tt_cap,
                        tt_used,
                        tt_fill,
                    ));
                } else {
                    log(&format!(
                        "info timealloc no_clock requested_limit {} effective_limit {} max_depth {:?} variant {} tt_cap {} tt_used {} tt_fill {}",
                        time_limit_ms,
                        effective_limit,
                        max_depth,
                        variant,
                        tt_cap,
                        tt_used,
                        tt_fill,
                    ));
                }
            }
        }

        // Choose search path based on effective noise.
        let tid = thread_id.unwrap_or(0) as usize;
        let (best_move, eval) = if effective_noise > 0 {
            // Use noisy search
            if let Some((bm, ev, _stats)) = search::get_best_move_with_noise(
                &mut self.game,
                depth,
                effective_limit,
                effective_noise,
                silent,
            ) {
                (bm, ev)
            } else {
                return JsValue::NULL;
            }
        } else {
            // Normal search with thread_id for Lazy SMP
            if let Some((bm, ev, _stats)) =
                search::get_best_move_threaded(&mut self.game, depth, effective_limit, silent, tid)
            {
                (bm, ev)
            } else {
                return JsValue::NULL;
            }
        };

        let js_move = JsMoveWithEval {
            from: format!("{},{}", best_move.from.x, best_move.from.y),
            to: format!("{},{}", best_move.to.x, best_move.to.y),
            promotion: best_move.promotion.map(|p| p.to_str().to_string()),
            eval,
            depth,
        };
        serde_wasm_bindgen::to_value(&js_move).unwrap()
    }

    /// MultiPV-enabled timed search. Returns an array of PV lines (best moves with their
    /// evaluations and full PVs).
    ///
    /// Parameters:
    /// - `time_limit_ms`: Maximum time to think in milliseconds
    /// - `multi_pv`: Number of best moves to return (default 1). Must be >= 1.
    /// - `silent`: If true, suppress info output during search
    ///
    /// When `multi_pv` is 1, this has zero overhead compared to `get_best_move_with_time`.
    /// For `multi_pv` > 1, subsequent PV lines are found by re-searching the position
    /// with previously found best moves excluded.
    #[wasm_bindgen]
    pub fn get_best_moves_multipv(
        &mut self,
        time_limit_ms: u32,
        multi_pv: Option<usize>,
        silent: Option<bool>,
    ) -> JsValue {
        let effective_limit = self.effective_time_limit_ms(time_limit_ms);
        let silent = silent.unwrap_or(false);
        let multi_pv = multi_pv.unwrap_or(1).max(1);

        let result =
            search::get_best_moves_multipv(&mut self.game, 50, effective_limit, multi_pv, silent);

        // Convert to JS-friendly format
        let js_lines: Vec<JsPVLine> = result
            .lines
            .iter()
            .map(|line| {
                // Format PV as array of "x,y->x,y" strings
                let pv_strings: Vec<String> = line
                    .pv
                    .iter()
                    .map(|m| format!("{},{}->{},{}", m.from.x, m.from.y, m.to.x, m.to.y))
                    .collect();

                JsPVLine {
                    from: format!("{},{}", line.mv.from.x, line.mv.from.y),
                    to: format!("{},{}", line.mv.to.x, line.mv.to.y),
                    promotion: line.mv.promotion.map(|p| p.to_str().to_string()),
                    eval: line.score,
                    depth: line.depth,
                    pv: pv_strings,
                }
            })
            .collect();

        serde_wasm_bindgen::to_value(&js_lines).unwrap_or(JsValue::NULL)
    }

    pub fn perft(&mut self, depth: usize) -> u64 {
        self.game.perft(depth)
    }

    pub fn setup_position(&mut self, position_icn: &str) {
        self.game.setup_position_from_icn(position_icn);
    }

    /// Returns all legal moves as a JS array of {from: "x,y", to: "x,y", promotion: string|null}
    pub fn get_legal_moves_js(&mut self) -> JsValue {
        let pseudo_legal = self.game.get_legal_moves();
        let mut legal_moves: Vec<JsMove> = Vec::new();

        for m in pseudo_legal {
            let undo = self.game.make_move(&m);
            let illegal = self.game.is_move_illegal();
            self.game.undo_move(&m, undo);

            if !illegal {
                legal_moves.push(JsMove {
                    from: format!("{},{}", m.from.x, m.from.y),
                    to: format!("{},{}", m.to.x, m.to.y),
                    promotion: m.promotion.map(|p| p.to_str().to_string()),
                });
            }
        }

        serde_wasm_bindgen::to_value(&legal_moves).unwrap_or(JsValue::NULL)
    }
}
