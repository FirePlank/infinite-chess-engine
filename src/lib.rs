use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub mod board;
pub mod game;
pub mod moves;
mod utils;
pub mod evaluation;
pub mod search;

use board::{Board, Piece, PieceType, PlayerColor, Coordinate};
use game::{GameState, EnPassantState};
use evaluation::calculate_initial_material;
use crate::moves::{set_world_bounds, SpatialIndices};

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
    pub eval: i32,    // centipawn score from side-to-move's perspective
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
    white: Vec<String>,  // String because BigInt serializes as string
    black: Vec<String>,
}

#[derive(Deserialize)]
struct JsWorldBounds {
    left: String,
    right: String,
    bottom: String,
    top: String,
}

#[derive(Deserialize)]
struct JsMoveHistory {
    from: String,       // "x,y"
    to: String,         // "x,y"
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

#[wasm_bindgen]
pub struct Engine {
    game: GameState,
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
            let x: i64 = p.x.parse().map_err(|_| JsValue::from_str("Invalid X coordinate"))?;
            let y: i64 = p.y.parse().map_err(|_| JsValue::from_str("Invalid Y coordinate"))?;

            let piece_type = PieceType::from_str(&p.piece_type)
                .unwrap_or(PieceType::Pawn);

            let color = PlayerColor::from_str(&p.player)
                .unwrap_or(PlayerColor::White);

            board.set_piece(x, y, Piece::new(piece_type, color));
        }

        // Starting side (color that moved first) as reported by JS. The engine
        // will reconstruct the current side-to-move by replaying move_history.
        let js_turn = PlayerColor::from_str(&js_game.turn)
            .unwrap_or(PlayerColor::White);

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
            
            let promotion_ranks = js_rules.promotion_ranks.map(|pr| {
                PromotionRanks {
                    white: pr.white.iter().filter_map(|s| s.parse::<i64>().ok()).collect(),
                    black: pr.black.iter().filter_map(|s| s.parse::<i64>().ok()).collect(),
                }
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
            if parts.len() != 2 { return None; }
            let x = parts[0].parse::<i64>().ok()?;
            let y = parts[1].parse::<i64>().ok()?;
            Some((x, y))
        }

        if js_game.move_history.is_empty() {
            // No history: trust JS clocks/turn/en-passant for this position
            game.en_passant = parsed_en_passant;
            game.turn = js_turn;
            game.halfmove_clock = js_game.halfmove_clock;
            game.fullmove_number = if js_game.fullmove_number == 0 { 1 } else { js_game.fullmove_number };
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

        Ok(Engine { game })
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
                color_code,
                piece_code,
                m.from.x,
                m.from.y,
                m.to.x,
                m.to.y,
                promo_part,
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

    /// Return the engine's static evaluation of the current position in centipawns,
    /// from the side-to-move's perspective (positive = advantage for side to move).
    pub fn evaluate_position(&mut self) -> i32 {
        evaluation::evaluate(&self.game)
    }

    /// Timed search. This also exposes the search evaluation as an `eval` field alongside the move,
    /// so callers can reuse the same search for adjudication.
    pub fn get_best_move_with_time(&mut self, time_limit_ms: u32) -> JsValue {
        if let Some((best_move, eval)) = search::get_best_move_timed_with_eval(
            &mut self.game,
            50,
            time_limit_ms as u128,
            true,
        ) {
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
