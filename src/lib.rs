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
    pub fn parse(s: &str) -> Self {
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

impl std::str::FromStr for Variant {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Variant::parse(s))
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

#[derive(Clone, Serialize, Deserialize)]
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
    #[serde(default)]
    win_conditions: Option<JsWinConditions>,
}

/// Win conditions per side, as received from JavaScript.
/// Each side has an array of conditions; we only use the first.
#[derive(Deserialize, Default)]
struct JsWinConditions {
    #[serde(default)]
    white: Vec<String>,
    #[serde(default)]
    black: Vec<String>,
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
        } else {
            // Always reset to infinite defaults to prevent world borders from
            // leaking from previous games in the same worker (e.g. SPRT).
            set_world_bounds(
                -1_000_000_000_000_000,
                1_000_000_000_000_000,
                -1_000_000_000_000_000,
                1_000_000_000_000_000,
            );
        }

        // Build starting GameState from JS board
        let mut board = Board::new();
        let mut white_has_royal = false;
        let mut black_has_royal = false;
        for p in &js_game.board.pieces {
            let x: i64 =
                p.x.parse()
                    .map_err(|_| JsValue::from_str("Invalid X coordinate"))?;
            let y: i64 =
                p.y.parse()
                    .map_err(|_| JsValue::from_str("Invalid Y coordinate"))?;

            let piece_type = p.piece_type.parse::<PieceType>().unwrap_or(PieceType::Pawn);

            let color = p
                .player
                .parse::<PlayerColor>()
                .unwrap_or(PlayerColor::White);

            if piece_type.is_royal() {
                match color {
                    PlayerColor::White => white_has_royal = true,
                    PlayerColor::Black => black_has_royal = true,
                    _ => {}
                }
            }

            board.set_piece(x, y, Piece::new(piece_type, color));
        }

        // Starting side (color that moved first) as reported by JS. The engine
        // will reconstruct the current side-to-move by replaying move_history.
        let js_turn = js_game
            .turn
            .parse::<PlayerColor>()
            .unwrap_or(PlayerColor::White);

        // Parse initial special rights (castling + pawn double-move)
        let mut special_rights = FxHashSet::default();
        for sr in js_game.special_rights {
            let parts: Vec<&str> = sr.split(',').collect();
            if parts.len() == 2
                && let (Ok(x), Ok(y)) = (parts[0].parse::<i64>(), parts[1].parse::<i64>())
            {
                special_rights.insert(Coordinate::new(x, y));
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
        let mut game_rules = if let Some(js_rules) = js_game.game_rules {
            use game::{GameRules, PromotionRanks, WinCondition};

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

            // Parse win conditions - use the first condition from each side's array.
            // white_win_condition = what White must do to beat Black (how White wins)
            // black_win_condition = what Black must do to beat White (how Black wins)
            let (white_win_condition, black_win_condition) =
                if let Some(wc) = js_rules.win_conditions {
                    let white_wc = wc
                        .white
                        .first()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(WinCondition::Checkmate);
                    let black_wc = wc
                        .black
                        .first()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(WinCondition::Checkmate);
                    (white_wc, black_wc)
                } else {
                    (WinCondition::Checkmate, WinCondition::Checkmate)
                };

            let mut rules = GameRules {
                promotion_ranks,
                promotion_types: None,
                promotions_allowed: js_rules.promotions_allowed,
                move_rule_limit: js_rules.move_rule,
                white_win_condition,
                black_win_condition,
            };
            rules.init_promotion_types();
            rules
        } else {
            game::GameRules::default()
        };

        // If a side has no royal pieces, the OPPONENT can't checkmate them, so the
        // opponent's win condition must be AllPiecesCaptured instead of Checkmate.
        // - If White has no royal: Black beats White via AllPiecesCaptured → set black_win_condition
        // - If Black has no royal: White beats Black via AllPiecesCaptured → set white_win_condition
        if !white_has_royal {
            game_rules.black_win_condition = game::WinCondition::AllPiecesCaptured;
        }
        if !black_has_royal {
            game_rules.white_win_condition = game::WinCondition::AllPiecesCaptured;
        }

        // Precompute effective promotion ranks and dynamic back ranks once per
        // game from promotion_ranks. For standard chess this yields promo
        // ranks 8/1 and back ranks 1/8.
        let (white_promo_rank, black_promo_rank, white_back_rank, black_back_rank) =
            if let Some(ref ranks) = game_rules.promotion_ranks {
                let white_promo = ranks
                    .white
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(2_000_000_000_000_000);
                let black_promo = ranks
                    .black
                    .iter()
                    .copied()
                    .min()
                    .unwrap_or(-2_000_000_000_000_000);

                // White's home side is near Black's promotion ranks, and vice versa.
                let wb = if black_promo == -2_000_000_000_000_000 {
                    1
                } else {
                    black_promo
                }; // white back rank
                let bb = if white_promo == 2_000_000_000_000_000 {
                    8
                } else {
                    white_promo
                }; // black back rank

                (white_promo, black_promo, wb, bb)
            } else {
                // Classical default: NO promotion unless explicitly provided.
                // For simplicity use unreachable ranks.
                (2_000_000_000_000_000, -2_000_000_000_000_000, 1, 8)
            };

        // Initialize game with starting position; clocks and turn will be fixed below.
        // crate::tiles::magic::init();

        let mut game = GameState {
            board,
            // Seed with the starting side; this ensures that replaying move history
            // produces the correct side-to-move even when Black (or another side)
            // moved first.
            turn: js_turn,
            special_rights,
            en_passant: None,
            halfmove_clock: js_game.halfmove_clock,
            fullmove_number: 1,
            material_score: 0,
            game_rules,
            variant: js_game.variant.as_deref().map(crate::Variant::parse),
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
            check_squares_white: FxHashSet::default(),
            check_squares_black: FxHashSet::default(),
            slider_rays_white: [None; 8],
            slider_rays_black: [None; 8],
            discovered_check_squares_white: FxHashSet::default(),
            discovered_check_squares_black: FxHashSet::default(),
            pawn_hash: 0,
            nonpawn_hash: 0,
            material_hash: 0,
            repetition: 0,
            white_non_pawn_material: false,
            black_non_pawn_material: false,
            effective_castling_rights: 0,
            castling_partner_counts: [0; 4],
            pinned_white: rustc_hash::FxHashMap::default(),
            pinned_black: rustc_hash::FxHashMap::default(),
            checkers_count_white: 0,
            checkers_count_black: 0,
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
                    let promo = hist.promotion.as_deref();
                    game.make_move_coords(from_x, from_y, to_x, to_y, promo);
                }
            }
        }

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
            search::get_best_move(&mut self.game, 50, u128::MAX, false, true)
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
    /// game state. Returns `(time_ms, is_soft_limit)`.
    ///
    /// **is_soft_limit** indicates whether the returned time is a suggestion
    /// (where exceeding it slightly is acceptable) vs a hard constraint.
    /// - Soft limit: untimed/infinite games with a suggested per-move limit
    /// - Hard limit: timed games where exceeding the budget risks flagging
    ///
    /// Uses logarithmic time formulas for proper scaling across
    /// different time controls (blitz, rapid, classical).
    fn effective_time_limit_ms(&self, requested_limit_ms: u32) -> (u128, u128, bool) {
        let Some(clock) = self.clock else {
            // No clock info: use the fixed per-move limit as a soft limit.
            // The search can use up to this time freely without flagging risk.
            let limit = requested_limit_ms as u128;
            return (limit, limit, true);
        };

        // Decide which side's clock to use.
        let (remaining_ms_raw, inc_ms_raw) = match self.game.turn {
            PlayerColor::White => (clock.wtime, clock.winc),
            PlayerColor::Black => (clock.btime, clock.binc),
            // Neutral side-to-move should not happen; fall back to
            // the requested limit as soft.
            PlayerColor::Neutral => {
                let limit = requested_limit_ms as u128;
                return (limit, limit, true);
            }
        };

        // If there is no usable clock information, use requested limit as soft.
        if remaining_ms_raw == 0 && inc_ms_raw == 0 {
            let limit = requested_limit_ms as u128;
            return (limit, limit, true);
        }

        // Handle zero remaining time with positive increment (increment-only)
        let remaining_ms = if remaining_ms_raw > 0 {
            remaining_ms_raw
        } else {
            // At least give ourselves a small buffer based on increment.
            inc_ms_raw.max(500)
        };

        let inc_ms = inc_ms_raw;

        // =========================================================================
        // Dynamic Time Allocation
        // Calculates optimal and maximum thinking time based on remaining budget.
        // =========================================================================

        // Move overhead for communication latency
        let move_overhead: u64 = 50;

        // Calculate scaled time (remaining time minus one move overhead)
        let scaled_time = remaining_ms.saturating_sub(move_overhead);

        // Maximum move horizon (centiMTG = moves to go * 100)
        // For games without movestogo, assume ~50 moves remaining
        // If less than 1 second, gradually reduce moves to go estimate
        let centi_mtg: i64 = if scaled_time >= 1000 {
            5051 // 50.51 moves * 100
        } else {
            ((scaled_time as f64) * 5.051) as i64
        };
        let centi_mtg = centi_mtg.max(100); // At least 1 move expected

        // timeLeft: total time we can use considering increment and overhead
        // Formula: remaining + inc * (MTG - 1) - overhead * (2 + MTG)
        // Division by 100 because centiMTG is in centimoves
        let time_left = (remaining_ms as i64
            + (inc_ms as i64 * (centi_mtg - 100) - move_overhead as i64 * (200 + centi_mtg)) / 100)
            .max(1) as f64;

        // original_time_adjust: logarithmic scaling factor based on available time
        // This factor prevents overspending at longer time controls by adjusting the
        // base allocation according to the total time budget.
        // For 10s games: log10(10000) = 4, adjust = 0.3128*4 - 0.4354 = 0.816
        // For 40m games: log10(2400000) = 6.38, adjust = 1.56
        let original_time_adjust = (0.3128 * time_left.max(1.0).log10() - 0.4354).max(0.1);

        // Log-based time constants for adaptive scaling
        let log_time_sec = (scaled_time as f64 / 1000.0).max(0.001).log10();

        // Optimum and maximum time constants
        let opt_constant = (0.0032116 + 0.000321123 * log_time_sec).min(0.00508017);
        let max_constant = (3.3977 + 3.0395 * log_time_sec).max(2.94761);

        // Game ply estimation based on fullmove number
        // In chess, typically ply = (fullmove_number - 1) * 2 + (is_black ? 1 : 0)
        let ply = self
            .game
            .fullmove_number
            .saturating_sub(1)
            .saturating_mul(2) as f64
            + if self.game.turn == PlayerColor::Black {
                1.0
            } else {
                0.0
            };

        // optScale: percentage of timeLeft to use for this move
        // Multiply by originalTimeAdjust to prevent overspending
        let opt_scale = ((0.0121431 + (ply + 2.94693_f64).powf(0.461073) * opt_constant)
            .min(0.213035 * remaining_ms as f64 / time_left))
            * original_time_adjust;

        // maxScale: multiplier from optimum to maximum
        let max_scale = max_constant.min(6.67704) + ply / 11.9847;

        // Calculate optimum and maximum allocations
        let optimum = (opt_scale * time_left) as u64;
        let maximum = ((max_scale * optimum as f64)
            .min(0.825179 * remaining_ms as f64 - move_overhead as f64)
            - 10.0)
            .max(0.0) as u64;

        // Apply sensible bounds
        let min_think_ms: u64 = 10;
        let optimum = optimum.max(min_think_ms);
        let maximum = maximum.max(optimum); // Maximum should never be less than optimum

        // Final safety cap: never exceed 82.5% of remaining time
        let absolute_cap = ((remaining_ms as f64) * 0.825 - move_overhead as f64) as u64;
        let optimum = optimum.min(absolute_cap.max(min_think_ms));
        let maximum = maximum.min(absolute_cap.max(min_think_ms));

        // Timed games are hard limits (risk of flagging).
        (optimum as u128, maximum as u128, false)
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
        // let legal_moves = self.game.get_legal_moves();
        // web_sys::console::log_1(&format!("Legal moves: {:?}", legal_moves).into());

        let (opt_time, max_time, is_soft_limit) = if time_limit_ms == 0 && max_depth.is_some() {
            // If explicit depth is requested with 0 time, treat as infinite time (fixed depth search)
            (u128::MAX, u128::MAX, true)
        } else {
            self.effective_time_limit_ms(time_limit_ms)
        };
        let silent = silent.unwrap_or(false);
        let depth = max_depth.unwrap_or(50).clamp(1, 50);
        let strength = self.strength_level.unwrap_or(3).clamp(1, 3);

        // Determine effective noise amplitude:
        // 1. If explicit noise_amp is provided, use it
        // 2. Otherwise, derive from strength level
        let effective_noise: i32 = if let Some(amp) = noise_amp {
            amp.max(0)
        } else {
            match strength {
                1 => 800,
                2 => 400,
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
                        "info timealloc side {} wtime {} btime {} winc {} binc {} limit {} soft {} variant {} tt_cap {} tt_used {} tt_fill {}",
                        side,
                        clock.wtime,
                        clock.btime,
                        clock.winc,
                        clock.binc,
                        opt_time,
                        is_soft_limit,
                        variant,
                        tt_cap,
                        tt_used,
                        tt_fill,
                    ));
                } else {
                    log(&format!(
                        "info timealloc no_clock requested_limit {} effective_limit {} max_depth {:?} variant {} tt_cap {} tt_used {} tt_fill {}",
                        time_limit_ms, opt_time, max_depth, variant, tt_cap, tt_used, tt_fill,
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
                opt_time,
                max_time,
                effective_noise,
                silent,
                is_soft_limit,
            ) {
                (bm, ev)
            } else {
                return JsValue::NULL;
            }
        } else {
            // Normal search with thread_id for Lazy SMP
            if let Some((bm, ev, _stats)) = search::get_best_move_threaded(
                &mut self.game,
                depth,
                opt_time,
                max_time,
                silent,
                tid,
                is_soft_limit,
            ) {
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
        let (opt_time, max_time, is_soft_limit) = self.effective_time_limit_ms(time_limit_ms);
        let silent = silent.unwrap_or(false);
        let multi_pv = multi_pv.unwrap_or(1).max(1);

        let result = search::get_best_moves_multipv(
            &mut self.game,
            50,
            opt_time,
            max_time,
            multi_pv,
            silent,
            is_soft_limit,
        );

        // Convert to JS-friendly format
        let js_lines: Vec<JsPVLine> = result
            .lines
            .iter()
            .map(|line| {
                // Format PV as array of "x,y->x,y" strings
                let pv_strings: Vec<String> = line
                    .pv
                    .iter()
                    .map(|m| {
                        format!(
                            "{},{}->{},{}{}",
                            m.from.x,
                            m.from.y,
                            m.to.x,
                            m.to.y,
                            m.promotion.map_or("", |p| p.to_site_code())
                        )
                    })
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

    /// Returns true if the current side to move is in check.
    pub fn is_in_check(&self) -> bool {
        self.game.is_in_check()
    }

    /// Returns true if either side has sufficient material to force checkmate.
    /// Returns false if the position is a dead draw due to insufficient material.
    /// This can be used by the SPRT harness to detect insufficient material draws.
    pub fn is_sufficient_material(&self) -> bool {
        // Use the evaluate_insufficient_material function
        // None = sufficient, Some(0) = dead draw, Some(n) = drawish
        match evaluation::insufficient_material::evaluate_insufficient_material(&self.game) {
            None => true,     // Sufficient material
            Some(0) => false, // Dead draw (insufficient)
            Some(_) => true,  // Drawish but not dead draw
        }
    }
}
