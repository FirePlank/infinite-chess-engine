use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use wasm_bindgen::prelude::*;

pub mod board;
pub mod evaluation;
pub mod game;
pub mod moves;
pub mod search;
mod utils;

use crate::moves::{set_world_bounds, SpatialIndices};
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

// #[wasm_bindgen]
// pub fn init_panic_hook() {
//     utils::set_panic_hook();
// }

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
    pub eval: i32, // centipawn score from side-to-move's perspective
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
}

#[derive(Deserialize, Default)]
struct JsGameRules {
    #[serde(default)]
    promotion_ranks: Option<JsPromotionRanks>,
    #[serde(default)]
    promotions_allowed: Option<Vec<String>>,
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
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(json_state: JsValue) -> Result<Engine, JsValue> {
        let js_game: JsFullGame = serde_wasm_bindgen::from_value(json_state)?;

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
        let mut special_rights = HashSet::new();
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
            };
            rules.init_promotion_types();
            rules
        } else {
            game::GameRules::default()
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
            white_pieces: Vec::new(),
            black_pieces: Vec::new(),
            spatial_indices: SpatialIndices::default(),
        };

        game.material_score = calculate_initial_material(&game.board);
        game.recompute_piece_counts(); // Rebuild piece lists and counts
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
            // No history: trust JS clocks/turn/en-passant for this position
            game.en_passant = parsed_en_passant;
            game.turn = js_turn;
            game.halfmove_clock = js_game.halfmove_clock;
            game.fullmove_number = if js_game.fullmove_number == 0 {
                1
            } else {
                js_game.fullmove_number
            };
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

        // Optional clock information (similar to UCI wtime/btime/winc/binc).
        let clock = js_game.clock;

        Ok(Engine { game, clock })
    }

    pub fn get_best_move(&mut self) -> JsValue {
        // console log all legal moves in the position
        let moves = self.game.get_legal_moves();
        for m in &moves {
            let piece_code = m.piece.piece_type.to_str();
            let color_code = m.piece.color.to_str();
            let promo_part = match m.promotion {
                Some(p) => format!(" promo={}", p.to_str()),
                None => String::new(),
            };
            let line = format!(
                "{}{}: ({},{}) -> ({},{}){}",
                color_code, piece_code, m.from.x, m.from.y, m.to.x, m.to.y, promo_part,
            );
            web_sys::console::debug_1(&JsValue::from(line));
        }
        if let Some(best_move) = search::get_best_move(&mut self.game, 50) {
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
    pub fn get_best_move_with_time(&mut self, time_limit_ms: u32) -> JsValue {
        let effective_limit = self.effective_time_limit_ms(time_limit_ms);
        #[cfg(target_arch = "wasm32")]
        {
            use crate::log;
            let variant = self
                .game
                .variant
                .map_or("unknown".to_string(), |v| format!("{:?}", v));
            if let Some(clock) = self.clock {
                let side = match self.game.turn {
                    PlayerColor::White => "w",
                    PlayerColor::Black => "b",
                    PlayerColor::Neutral => "n",
                };
                log(&format!(
                    "info timealloc side {} wtime {} btime {} winc {} binc {} limit {} variant {}",
                    side,
                    clock.wtime,
                    clock.btime,
                    clock.winc,
                    clock.binc,
                    effective_limit,
                    variant
                ));
            } else {
                log(&format!(
                    "info timealloc no_clock requested_limit {} effective_limit {} variant {}",
                    time_limit_ms, effective_limit, variant
                ));
            }
        }
        if let Some((best_move, eval)) =
            search::get_best_move_timed_with_eval(&mut self.game, 50, effective_limit, false)
        {
            let js_move = JsMoveWithEval {
                from: format!("{},{}", best_move.from.x, best_move.from.y),
                to: format!("{},{}", best_move.to.x, best_move.to.y),
                promotion: best_move.promotion.map(|p| p.to_str().to_string()),
                eval,
            };
            serde_wasm_bindgen::to_value(&js_move).unwrap()
        } else {
            JsValue::NULL
        }
    }

    pub fn perft(&mut self, depth: usize) -> u64 {
        self.game.perft(depth)
    }

    pub fn setup_standard_chess(&mut self) {
        self.game.setup_standard_chess();
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
