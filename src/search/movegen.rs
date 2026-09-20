//! Staged move generation. Each context (main search, evasions, ProbCut, qsearch)
//! walks its own stage sequence in [`MoveStage`] order, so the search can take a
//! cutoff before later stages generate anything.

use super::params::{DEFAULT_SORT_QUIET, sort_countermove, sort_killer1, sort_killer2};
use super::{
    LOW_PLY_HISTORY_MASK, LOW_PLY_HISTORY_SIZE, PAWN_HISTORY_MASK, Searcher, hash_coord_16,
    hash_move_dest,
};
use crate::board::{PieceType, PlayerColor};
use crate::game::GameState;
use crate::moves::{Move, MoveGenContext, MoveList, get_quiescence_captures, get_quiet_moves_into};

/// Good quiet threshold
const GOOD_QUIET_THRESHOLD: i32 = -14000;

/// Depth-staged tight generation: at remaining depth <= this, quiet slider
/// candidates are capped per ray (short-range + king-aligned always survive).
const TIGHT_GEN_DEPTH: i32 = 3;
const TIGHT_GEN_RAY_CAP: usize = 3;

/// Stages of move generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveStage {
    // Main search
    MainTT,
    CaptureInit,
    GoodCapture,
    Killer1,
    Killer2,
    QuietInit,
    GoodQuiet,
    BadCapture,
    BadQuiet,

    // Evasion (when in check)
    EvasionTT,
    EvasionInit,
    Evasion,

    // ProbCut
    ProbCutTT,
    ProbCutInit,
    ProbCut,

    // QSearch
    QSearchTT,
    QCaptureInit,
    QCapture,

    Done,
}

/// Move with score for sorting
#[derive(Clone, Copy)]
struct ScoredMove {
    m: Move,
    score: i32,
    /// Set only where quiet scoring already computed it; negamax reuses it
    /// instead of running a second `move_gives_check_fast` on the same move.
    gives_check: Option<bool>,
}

/// (idx, prev_cap, prev_ic, prev_piece, prev_to_h)
type ContHistoryIndex = (usize, usize, usize, usize, usize);

// Scored-move buffers returned on drop: the picker is built once per interior
// node, and a fresh Vec there is a malloc/free per node.
thread_local! {
    static PICKER_VECS: std::cell::RefCell<Vec<Vec<ScoredMove>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

impl Drop for StagedMoveGen {
    fn drop(&mut self) {
        let mut v = std::mem::take(&mut self.moves);
        if v.capacity() > 0 {
            v.clear();
            PICKER_VECS.with(|p| {
                let mut p = p.borrow_mut();
                if p.len() < 64 {
                    p.push(v);
                }
            });
        }
    }
}

/// Staged move generator
pub struct StagedMoveGen {
    stage: MoveStage,
    tt_move: Option<Move>,

    // Move buffer
    moves: Vec<ScoredMove>,
    cur: usize,
    end_bad_captures: usize,
    end_bad_quiets: usize,
    cached_gives_check: Option<bool>,
    end_captures: usize,
    end_generated: usize,

    // Search parameters
    ply: usize,
    depth: i32,
    threshold: i32, // For ProbCut SEE threshold

    // Previous move info for countermove lookup
    prev_from_hash: usize,
    prev_to_hash: usize,

    // Killers (scored in score_quiet, not separate stages)
    /// Last (square, attacked) answers for this node. Generation groups moves by
    /// piece, so one queen's 30 quiets asked the same question 30 times.
    memo_from: std::cell::Cell<Option<(i64, i64, bool)>>,
    memo_victim: std::cell::Cell<Option<(i64, i64, bool)>>,
    killer1: Option<Move>,
    killer2: Option<Move>,

    // Flags
    skip_quiets: bool,
    excluded_move: Option<Move>,

    // Pre-calculated continuation history pointers for the current ply.
    pub(crate) cont_history_indices: smallvec::SmallVec<[ContHistoryIndex; 3]>,

    // Side-to-move pin map, computed once per node and shared by the capture
    // and quiet stages.
    pins_cache: Option<rustc_hash::FxHashMap<crate::board::Coordinate, (i64, i64)>>,
}

/// Sorts moves with score >= limit to the front in descending order.
/// Uses slice::sort_unstable_by for O(N log N) performance.
#[inline]
fn partial_insertion_sort(moves: &mut [ScoredMove], limit: i32) {
    if limit == i32::MIN {
        // Sort everything
        moves.sort_unstable_by(|a, b| b.score.cmp(&a.score));
    } else {
        // Partition moves >= limit to the front
        let mut left = 0;
        for i in 0..moves.len() {
            if moves[i].score >= limit {
                moves.swap(i, left);
                left += 1;
            }
        }
        // Sort only the high-scoring segment
        moves[0..left].sort_unstable_by(|a, b| b.score.cmp(&a.score));
    }
}

impl StagedMoveGen {
    /// Create MovePicker for main search or quiescence search.
    /// Primary constructor.
    pub fn new(
        tt_move: Option<Move>,
        ply: usize,
        depth: i32,
        searcher: &Searcher,
        game: &GameState,
    ) -> Self {
        Self::new_with_check(tt_move, ply, depth, searcher, game, game.is_in_check())
    }

    /// Same as [`new`], for callers that already know whether the side to move is
    /// attacked: recomputing it here costs a full attack scan per node.
    pub fn new_with_check(
        tt_move: Option<Move>,
        ply: usize,
        depth: i32,
        searcher: &Searcher,
        game: &GameState,
        in_check: bool,
    ) -> Self {
        let is_in_check = in_check && game.must_escape_check();
        let tt_move = tt_move.map(|m| Self::reconstruct_castling_partner(game, m));
        let tt_valid = tt_move.is_some() && Self::is_pseudo_legal(game, &tt_move.unwrap());
        // An invalid TT move must not linger: the later stages filter "the TT
        // move" out of killers and quiets, so keeping it erases a real move.
        let tt_move = if tt_valid { tt_move } else { None };

        // Set initial stage based on TT move availability
        // If TT move is valid, start at TT stage; otherwise skip to Init stage
        let start_stage = if is_in_check {
            if tt_valid {
                MoveStage::EvasionTT
            } else {
                MoveStage::EvasionInit
            }
        } else if depth > 0 {
            if tt_valid {
                MoveStage::MainTT
            } else {
                MoveStage::CaptureInit
            }
        } else {
            // QSearch
            if tt_valid {
                MoveStage::QSearchTT
            } else {
                MoveStage::QCaptureInit
            }
        };

        Self::init(tt_move, ply, depth, 0, searcher, start_stage)
    }

    /// Create MovePicker for ProbCut - captures with SEE >= threshold.
    /// Secondary constructor.
    pub fn new_probcut(
        tt_move: Option<Move>,
        threshold: i32,
        searcher: &Searcher,
        game: &GameState,
    ) -> Self {
        debug_assert!(!Self::is_in_check(game), "ProbCut not used when in check");

        // TT move valid only if it's a capture and pseudo-legal
        let tt_move = tt_move.map(|m| Self::reconstruct_castling_partner(game, m));
        let tt_valid = tt_move.is_some()
            && Self::is_capture(game, &tt_move.unwrap())
            && Self::is_pseudo_legal(game, &tt_move.unwrap());

        let start_stage = if tt_valid {
            MoveStage::ProbCutTT
        } else {
            MoveStage::ProbCutInit
        };

        Self::init(tt_move, 0, 0, threshold, searcher, start_stage)
    }

    fn init(
        tt_move: Option<Move>,
        ply: usize,
        depth: i32,
        threshold: i32,
        searcher: &Searcher,
        stage: MoveStage,
    ) -> Self {
        let (prev_from_hash, prev_to_hash) = if ply > 0 {
            searcher.prev_move_stack[ply - 1]
        } else {
            (0, 0)
        };

        let killer1 = if ply < searcher.killers.len() {
            searcher.killers[ply][0]
        } else {
            None
        };
        let killer2 = if ply < searcher.killers.len() {
            searcher.killers[ply][1]
        } else {
            None
        };

        Self {
            stage,
            tt_move,
            moves: PICKER_VECS
                .with(|p| p.borrow_mut().pop())
                .unwrap_or_else(|| Vec::with_capacity(64)),
            cur: 0,
            end_bad_captures: 0,
            end_bad_quiets: 0,
            cached_gives_check: None,
            end_captures: 0,
            end_generated: 0,
            ply,
            depth,
            threshold,
            prev_from_hash,
            prev_to_hash,
            memo_from: std::cell::Cell::new(None),
            memo_victim: std::cell::Cell::new(None),
            killer1,
            killer2,
            skip_quiets: false,
            excluded_move: None,
            cont_history_indices: smallvec::SmallVec::new(),
            pins_cache: None,
        }
    }

    /// Create with exclusion for singular extension
    pub fn with_exclusion(
        tt_move: Option<Move>,
        ply: usize,
        depth: i32,
        searcher: &Searcher,
        game: &GameState,
        excluded: Move,
    ) -> Self {
        let mut generator = Self::new(tt_move, ply, depth, searcher, game);
        generator.excluded_move = Some(excluded);
        generator
    }

    /// Signal to skip quiet moves (LMP)
    #[inline]
    pub fn skip_quiet_moves(&mut self) {
        self.skip_quiets = true;
    }

    #[inline]
    fn is_in_check(game: &GameState) -> bool {
        game.is_in_check() && game.must_escape_check()
    }

    #[inline]
    fn moves_match(a: &Move, b: &Option<Move>) -> bool {
        match b {
            Some(bm) => a.from == bm.from && a.to == bm.to && a.promotion == bm.promotion,
            None => false,
        }
    }

    #[inline]
    fn is_excluded(&self, m: &Move) -> bool {
        Self::moves_match(m, &self.excluded_move)
    }

    /// TT and killer moves decode without their castling rook partner, so a castling
    /// move fails pseudo-legality and is then skipped as already-tried when generation
    /// produces it. Rebuild the partner by the same nearest-eligible rule.
    fn reconstruct_castling_partner(game: &GameState, mut m: Move) -> Move {
        if m.partner_x != crate::moves::NO_PARTNER || !m.piece.piece_type().is_royal() {
            return m;
        }
        let dx = m.to.x - m.from.x;
        if dx.abs() != 2 {
            return m;
        }
        let dir = if dx > 0 { 1i64 } else { -1i64 };
        if let Some(row) = game.spatial_indices.rows.get(&m.from.y)
            && let Some((partner_x, packed)) = row.find_nearest(m.to.x, dir)
        {
            let partner = crate::board::Piece::from_packed(packed);
            let partner_coord = crate::board::Coordinate::new(partner_x, m.from.y);
            if partner.color() == m.piece.color()
                && partner.piece_type() != PieceType::Pawn
                && !partner.piece_type().is_royal()
                && game.special_rights.contains(&partner_coord)
            {
                m.partner_x = partner_coord.x;
            }
        }
        m
    }

    #[inline]
    fn is_tt_move(&self, m: &Move) -> bool {
        Self::moves_match(m, &self.tt_move)
    }

    #[inline]
    fn is_capture(game: &GameState, m: &Move) -> bool {
        // En passant lands on an empty square, so an occupancy test alone files it
        // as a quiet — wrong for evasion scoring and for ProbCut's TT move.
        game.board.is_occupied(m.to.x, m.to.y) || game.is_en_passant(m)
    }

    /// Pseudo-legal check - verifies piece exists, correct color/type, and path validation
    #[inline]
    fn is_pseudo_legal(game: &GameState, m: &Move) -> bool {
        use crate::tiles::{local_index, tile_coords};

        // 1. Fast Tile Access
        let (cx, cy) = tile_coords(m.from.x, m.from.y);
        let from_idx = local_index(m.from.x, m.from.y);

        let tile = match game.board.tiles.get_tile(cx, cy) {
            Some(t) => t,
            None => return false,
        };

        // 2. Identity Check
        let packed = tile.piece[from_idx];
        if packed == 0 {
            return false;
        }

        let piece = crate::board::Piece::from_packed(packed);

        if piece.color() != game.turn || piece.piece_type() != m.piece.piece_type() {
            return false;
        }

        // One tile fetch yields both the packed piece and the occupancy bit. A neutral
        // Void packs to 0 yet occupies, so pawn pushes must test occupancy rather than
        // packed == 0.
        let (tx, ty) = tile_coords(m.to.x, m.to.y);
        let to_idx = local_index(m.to.x, m.to.y);
        let (target_packed, target_occupied) = if tx == cx && ty == cy {
            (tile.piece[to_idx], (tile.occ_all >> to_idx) & 1 != 0)
        } else {
            match game.board.tiles.get_tile(tx, ty) {
                Some(t) => (t.piece[to_idx], (t.occ_all >> to_idx) & 1 != 0),
                None => (0, false),
            }
        };

        if target_packed != 0 {
            let target = crate::board::Piece::from_packed(target_packed);
            if target.color() == game.turn {
                return false;
            }
        }

        // Castling belongs to any royal with rights paired with any non-pawn that
        // has them, not to the king alone.
        if piece.piece_type().is_royal() {
            let dx = m.to.x - m.from.x;
            if m.to.y == m.from.y && dx.abs() > 1 {
                if m.partner_x == crate::moves::NO_PARTNER {
                    return false;
                }
                let partner = crate::board::Coordinate::new(m.partner_x, m.from.y);
                if !game
                    .board
                    .is_occupied_by_color(partner.x, partner.y, game.turn)
                {
                    return false;
                }
                // Both ends need their rights, and the pair must stand at least three
                // apart, or the king would land on or past its partner.
                if !game.special_rights.contains(&m.from)
                    || !game.special_rights.contains(&partner)
                    || partner.y != m.from.y
                    || (partner.x - m.from.x).abs() < 3
                    || (partner.x - m.from.x).signum() != (m.to.x - m.from.x).signum()
                {
                    return false;
                }
                // The partner must be the nearest piece along the row: anything in
                // between blocks the castle, however far apart the two ends are.
                let dir = if dx > 0 { 1 } else { -1 };
                if let Some(row_pieces) = game.spatial_indices.rows.get(&m.from.y)
                    && let Some((nearest_x, _)) = row_pieces.find_nearest(m.from.x, dir)
                    && ((dir > 0 && nearest_x < partner.x) || (dir < 0 && nearest_x > partner.x))
                {
                    return false;
                }
                return true;
            }
        }

        // 4. Piece-Specific Logic
        match piece.piece_type() {
            PieceType::Pawn => {
                let dx = (m.to.x - m.from.x).abs();
                let dy = m.to.y - m.from.y;
                let dir = if game.turn == PlayerColor::White {
                    1
                } else {
                    -1
                };

                if dx == 0 {
                    // Push. Emptiness is judged by occupancy (Void occupies but
                    // packs to 0); the target bit was read with the tile above.
                    if dy == dir {
                        !target_occupied
                    } else if dy == 2 * dir {
                        // 2 steps. The right is per-square and a killer outlives the
                        // pawn that earned it, so a different pawn now standing there
                        // must not inherit its double step.
                        if target_occupied || !game.special_rights.contains(&m.from) {
                            return false;
                        }
                        // Intermediate square: one direct occ_all read (reuse
                        // the from-tile when the mid square is in it).
                        let mid_y = m.from.y + dir;
                        let mid_idx = local_index(m.from.x, mid_y);
                        let (mx, my) = tile_coords(m.from.x, mid_y);
                        let mid_occupied = if mx == cx && my == cy {
                            (tile.occ_all >> mid_idx) & 1 != 0
                        } else {
                            match game.board.tiles.get_tile(mx, my) {
                                Some(t) => (t.occ_all >> mid_idx) & 1 != 0,
                                None => false,
                            }
                        };
                        !mid_occupied
                    } else {
                        false
                    }
                } else {
                    // Capture
                    if dx == 1 && dy == dir {
                        if target_packed != 0 {
                            return true;
                        }
                        if let Some(ep) = &game.en_passant
                            && ep.square.x == m.to.x
                            && ep.square.y == m.to.y
                        {
                            return true;
                        }
                        return false;
                    }
                    false
                }
            }
            PieceType::Knight => {
                let dx = (m.to.x - m.from.x).abs();
                let dy = (m.to.y - m.from.y).abs();
                (dx == 1 && dy == 2) || (dx == 2 && dy == 1)
            }
            PieceType::King => {
                let dx = (m.to.x - m.from.x).abs();
                let dy = (m.to.y - m.from.y).abs();
                dx <= 1 && dy <= 1
            }
            PieceType::Rook | PieceType::Bishop | PieceType::Queen => {
                let dx = m.to.x - m.from.x;
                let dy = m.to.y - m.from.y;

                let is_ortho = dx == 0 || dy == 0;
                let is_diag = dx.abs() == dy.abs();

                let valid_type = match piece.piece_type() {
                    PieceType::Rook => is_ortho,
                    PieceType::Bishop => is_diag,
                    PieceType::Queen => is_ortho || is_diag,
                    _ => false,
                };
                if !valid_type {
                    return false;
                }

                // Same-Tile Fast Path
                let step_x = dx.signum();
                let step_y = dy.signum();
                if let Some(is_clear) =
                    crate::moves::is_path_clear_locally(&game.board, &m.from, &m.to, step_x, step_y)
                {
                    return is_clear;
                }

                crate::moves::is_piece_attacking_square(
                    &game.board,
                    &piece,
                    &m.from,
                    &m.to,
                    &game.spatial_indices,
                    &game.game_rules,
                )
            }
            _ => crate::moves::is_piece_attacking_square(
                &game.board,
                &piece,
                &m.from,
                &m.to,
                &game.spatial_indices,
                &game.game_rules,
            ),
        }
    }

    /// Score capture: 10 * VictimValue - AttackerValue + CaptureHistory + StatScore
    fn score_capture(game: &GameState, searcher: &Searcher, m: &Move) -> i32 {
        if let Some(target) = game.board.get_piece(m.to.x, m.to.y) {
            let victim_val = game.get_piece_value(target.piece_type(), target.color());
            let attacker_val = game.get_piece_value(m.piece.piece_type(), m.piece.color());
            // Include promotion gain so capture-promotions sort by their true value.
            let promo_gain = m.promotion.map_or(0, |pt| {
                game.get_piece_value(pt, m.piece.color()) - attacker_val
            });

            let pt_idx = m.piece.piece_type() as usize;
            let target_idx = target.piece_type() as usize;

            let cap_hist = searcher
                .capture_history
                .get(pt_idx)
                .and_then(|row| row.get(target_idx))
                .copied()
                .unwrap_or(0);

            let hist_idx = hash_move_dest(m);
            let history_score =
                searcher.history[crate::search::hist_color(m.piece.color())][pt_idx][hist_idx];

            10 * (victim_val + promo_gain) - attacker_val + (cap_hist / 8) + (history_score / 8)
        } else if game.is_en_passant(m) {
            // The victim sits beside m.to, not on it, so the branch above scores a
            // real pawn capture as 0 and sorts it below every other capture.
            let attacker_val = game.get_piece_value(PieceType::Pawn, m.piece.color());
            let victim_val = game.get_piece_value(PieceType::Pawn, m.piece.color().opponent());
            let pt_idx = m.piece.piece_type() as usize;
            let cap_hist = searcher
                .capture_history
                .get(pt_idx)
                .and_then(|row| row.get(PieceType::Pawn as usize))
                .copied()
                .unwrap_or(0);
            10 * victim_val - attacker_val + (cap_hist / 8)
        } else if let Some(pt) = m.promotion {
            // A quiet promotion has no victim, so the branch above never sees it
            // and it sorted at 0, below every capture; price it by what it wins.
            let attacker_val = game.get_piece_value(m.piece.piece_type(), m.piece.color());
            let promo_gain = game.get_piece_value(pt, m.piece.color()) - attacker_val;
            let hist_idx = hash_move_dest(m);
            let history_score = searcher.history[crate::search::hist_color(m.piece.color())]
                [m.piece.piece_type() as usize][hist_idx];
            10 * promo_gain - attacker_val + (history_score / 8)
        } else {
            0
        }
    }

    /// Score quiet move using history heuristics (includes killer/countermove bonuses)
    fn score_quiet(&self, game: &GameState, searcher: &Searcher, m: &Move) -> (i32, Option<bool>) {
        let mut score: i32 = DEFAULT_SORT_QUIET;

        // Killer bonus (integrated into scoring, not separate stages)
        // Check killers via exact match (cheap)
        if let Some(k1) = self.killer1
            && m.from == k1.from
            && m.to == k1.to
            && m.promotion == k1.promotion
        {
            return (sort_killer1(), None);
        }
        if let Some(k2) = self.killer2
            && m.from == k2.from
            && m.to == k2.to
            && m.promotion == k2.promotion
        {
            return (sort_killer2(), None);
        }

        // A run to the far shell is a last resort, so it belongs behind every other
        // quiet: full-width searching it costs ~20% nodes, and returning here also
        // skips the attack scan below for a move that almost never attacks anything.
        if crate::moves::is_far_escape_move(m) {
            return (GOOD_QUIET_THRESHOLD - 1, None);
        }

        // Countermove bonus
        if self.ply > 0 && self.prev_from_hash < 256 && self.prev_to_hash < 256 {
            let entry = unsafe {
                searcher
                    .countermoves
                    .get_unchecked(self.prev_from_hash)
                    .get_unchecked(self.prev_to_hash)
            };
            if entry.0 != 0
                && entry.0 == m.piece.piece_type() as u8
                && entry.1 == m.to.x as i32
                && entry.2 == m.to.y as i32
            {
                score += sort_countermove();
            }
        }

        // Main history: 2 * mainHistory[us][move]
        let idx = hash_move_dest(m);
        let pt_idx = m.piece.piece_type() as usize; // Safe cast

        unsafe {
            if pt_idx < 32 {
                // Bounds check for safety, though piece type should be valid
                let side = crate::search::hist_color(m.piece.color());
                let val = *searcher
                    .history
                    .get_unchecked(side)
                    .get_unchecked(pt_idx)
                    .get_unchecked(idx);
                score += 2 * val;
            }
        }

        // Pawn history: 2 * pawnHistory[pawn_hash % SIZE][piece][to]
        let ph_idx = (game.pawn_hash & PAWN_HISTORY_MASK) as usize;
        score += 2 * searcher.pawn_hist(ph_idx, pt_idx, idx);

        // Continuation history - Optimized using pre-calculated indices
        let cur_from_hash = hash_coord_16(m.from.x, m.from.y);
        let cur_to_hash = hash_coord_16(m.to.x, m.to.y);

        const CONT_WEIGHTS: [i32; 3] = [1024, 712, 410];
        for &(idx, prev_cap, prev_ic, prev_piece, prev_to_h) in &self.cont_history_indices {
            // Access: cont_history[idx][prev_cap][prev_ic][prev_piece][prev_to_h][cur_from_hash][cur_to_hash]
            let val = searcher.cont_history[idx][prev_cap][prev_ic][prev_piece][prev_to_h]
                [cur_from_hash][cur_to_hash] as i32;
            score += (val * CONT_WEIGHTS[idx]) / 1024;
        }

        let gives_check = Self::move_gives_check_fast(game, m);
        if gives_check && super::see_ge(game, m, -75) {
            score += 16384;
        }

        if self.ply < LOW_PLY_HISTORY_SIZE {
            let move_hash = idx & LOW_PLY_HISTORY_MASK;
            unsafe {
                if let Some(row) = searcher.low_ply_history.get(self.ply) {
                    let val = *row.get_unchecked(move_hash);
                    score += 8 * val / (1 + self.ply as i32);
                }
            }
        }

        // A quiet attacking an enemy piece is about twice as likely to be played, and
        // unlike dest-hashed history this signal does not alias at deep nodes. Gated to
        // depth >= 4, since the shallow nodes are the bulk of the tree.
        if self.depth >= 4 {
            let mover = m
                .promotion
                .map(|pt| crate::board::Piece::new(pt, m.piece.color()))
                .unwrap_or(m.piece);
            let value_of = |pt: PieceType, c: PlayerColor| game.get_piece_value(pt, c);
            let (best_vv, best_sq) = match crate::moves::best_capture_victim(
                &game.board,
                &mover,
                &m.to,
                &game.spatial_indices,
                &value_of,
            ) {
                Some(found) => found,
                None => {
                    // Knightrider, rose and huygen rays still need the generator.
                    let empty_pins = rustc_hash::FxHashMap::default();
                    let ctx = crate::moves::MoveGenContext {
                        special_rights: &game.special_rights,
                        en_passant: &game.en_passant,
                        game_rules: &game.game_rules,
                        indices: &game.spatial_indices,
                        enemy_king_pos: game.enemy_king_pos(),
                        pinned: &empty_pins,
                    };
                    let mut caps = MoveList::new();
                    crate::moves::generate_captures_for_piece(
                        &game.board,
                        &mover,
                        &m.to,
                        &ctx,
                        &mut caps,
                    );
                    let mut bv = 0;
                    let mut bs = None;
                    for c in caps.iter() {
                        if let Some(vic) = game.board.get_piece(c.to.x, c.to.y)
                            && vic.color() != m.piece.color()
                            && vic.color() != PlayerColor::Neutral
                            && !vic.piece_type().is_uncapturable()
                        {
                            let vv = game.get_piece_value(vic.piece_type(), vic.color());
                            if vv > bv {
                                bv = vv;
                                bs = Some(c.to);
                            }
                        }
                    }
                    (bv, bs)
                }
            };
            if best_vv > 0 {
                let mut q = best_vv * 6;
                // Undefended victim: no piece of the victim's color covers its square.
                if let Some(sq) = best_sq
                    && !self.square_attacked_memo(game, &sq, &self.memo_victim, m.piece.color())
                {
                    q *= 2;
                }
                score += q.min(12000);
            }

            // Moving a valuable piece off an attacked square saves material, so order
            // it early or LMR and LMP bury the defensive resource.
            let mover_val = game.get_piece_value(m.piece.piece_type(), m.piece.color());
            if mover_val >= 250
                && self.square_attacked_memo(game, &m.from, &self.memo_from, m.piece.color())
                && !crate::moves::is_square_attacked(
                    &game.board,
                    &m.to,
                    m.piece.color().opponent(),
                    &game.spatial_indices,
                )
            {
                score += (mover_val * 5).min(10000);
            }
        }

        (score, Some(gives_check))
    }

    /// `is_square_attacked` against a one-entry memo. The picker scores a single
    /// position, so a repeated square always has the same answer.
    #[inline]
    fn square_attacked_memo(
        &self,
        game: &GameState,
        sq: &crate::board::Coordinate,
        memo: &std::cell::Cell<Option<(i64, i64, bool)>>,
        mover: PlayerColor,
    ) -> bool {
        if let Some((x, y, v)) = memo.get()
            && x == sq.x
            && y == sq.y
        {
            return v;
        }
        let v = crate::moves::is_square_attacked(
            &game.board,
            sq,
            mover.opponent(),
            &game.spatial_indices,
        );
        memo.set(Some((sq.x, sq.y, v)));
        v
    }

    /// Score evasion move
    fn score_evasion(
        &self,
        game: &GameState,
        searcher: &Searcher,
        m: &Move,
    ) -> (i32, Option<bool>) {
        if Self::is_capture(game, m) {
            // Capture: PieceValue + (1 << 28)
            let captured_val = game
                .board
                .get_piece(m.to.x, m.to.y)
                .map(|p| game.get_piece_value(p.piece_type(), p.color()))
                .unwrap_or(0);
            (captured_val + (1 << 28), None)
        } else {
            // Quiet: use history
            self.score_quiet(game, searcher, m)
        }
    }

    /// Is the ray between two aligned squares clear once `vacated` is emptied?
    /// Uses the index-based clear test, splitting the segment when the mover's
    /// origin lies on it, so no ray is ever walked square by square.
    #[inline]
    fn ray_clear_after_move(
        game: &GameState,
        from: &crate::board::Coordinate,
        to: &crate::board::Coordinate,
        vacated: &crate::board::Coordinate,
    ) -> bool {
        use crate::evaluation::base::is_clear_line_between_fast;
        let idx = &game.spatial_indices;
        let on_segment = {
            let (dx, dy) = (to.x - from.x, to.y - from.y);
            let (vx, vy) = (vacated.x - from.x, vacated.y - from.y);
            dx * vy == dy * vx
                && vx.signum() == dx.signum()
                && vy.signum() == dy.signum()
                && vx.abs() <= dx.abs()
                && vy.abs() <= dy.abs()
                && (vx != 0 || vy != 0)
                && (vx != dx || vy != dy)
        };
        if on_segment {
            is_clear_line_between_fast(idx, from, vacated)
                && is_clear_line_between_fast(idx, vacated, to)
        } else {
            is_clear_line_between_fast(idx, from, to)
        }
    }

    /// Fast check detection
    #[inline(always)]
    pub fn move_gives_check_fast(game: &GameState, m: &Move) -> bool {
        // What stands on the destination after the move, which is the promoted piece
        // for a promotion; testing those with the pawn's geometry misses the check.
        let pt = m.promotion.unwrap_or(m.piece.piece_type());
        let color = m.piece.color();
        let tx = m.to.x;
        let ty = m.to.y;

        // Knights and pawns: test the live royal positions directly. This arithmetic
        // beats a hash probe, and a set built at setup goes stale once a royal moves.
        if pt == PieceType::Knight || pt == PieceType::Pawn {
            let royals = if color == PlayerColor::White {
                &game.black_royals
            } else {
                &game.white_royals
            };
            let fwd = if color == PlayerColor::White { 1 } else { -1 };
            for r in royals {
                let adx = (r.x - tx).abs();
                let dy = r.y - ty;
                if pt == PieceType::Knight {
                    if (adx == 1 && dy.abs() == 2) || (adx == 2 && dy.abs() == 1) {
                        return true;
                    }
                } else if adx == 1 && dy == fwd {
                    return true;
                }
            }
            return false;
        }

        // Get enemy royal positions
        let royals = if color == PlayerColor::White {
            &game.black_royals
        } else {
            &game.white_royals
        };

        if royals.is_empty() {
            return false;
        }

        use crate::attacks::{
            CAMEL_MASK, DIAG_MASK, GIRAFFE_MASK, HAWK_MASK, KING_MASK, KNIGHT_MASK, ORTHO_MASK,
            ZEBRA_MASK,
        };
        let pt_bit = 1u32 << (pt as u8);
        // One test keeps non-fairy variants off the per-royal offset checks below.
        const FAIRY_LEAP_MASK: u32 =
            KING_MASK | CAMEL_MASK | GIRAFFE_MASK | ZEBRA_MASK | HAWK_MASK;
        let is_fairy_leaper = (pt_bit & FAIRY_LEAP_MASK) != 0;

        for king_pos in royals {
            let dx = tx - king_pos.x;
            let dy = ty - king_pos.y;
            let adx = dx.abs();
            let ady = dy.abs();

            // Knight-like check
            if (pt_bit & KNIGHT_MASK) != 0 && ((adx == 1 && ady == 2) || (adx == 2 && ady == 1)) {
                return true;
            }

            // These all jump, so a matching offset IS the check -- no ray to clear.
            // Missing them let real checks be futility/history-pruned and dropped from qsearch.
            if is_fairy_leaper {
                let hits = if (pt_bit & CAMEL_MASK) != 0 {
                    (adx == 1 && ady == 3) || (adx == 3 && ady == 1)
                } else if (pt_bit & GIRAFFE_MASK) != 0 {
                    (adx == 1 && ady == 4) || (adx == 4 && ady == 1)
                } else if (pt_bit & ZEBRA_MASK) != 0 {
                    (adx == 2 && ady == 3) || (adx == 3 && ady == 2)
                } else if (pt_bit & HAWK_MASK) != 0 {
                    let step = adx.max(ady);
                    (step == 2 || step == 3) && (adx == 0 || ady == 0 || adx == ady)
                } else {
                    // KING_MASK: King/Guard/Centaur/RoyalCentaur one-square step.
                    adx <= 1 && ady <= 1 && (adx | ady) != 0
                };
                if hits {
                    return true;
                }
            }

            // A sliding check needs a clear ray from destination to royal; alignment
            // alone counts blocked rays as checks, exempting them from pruning. The
            // mover's origin cannot block, as this move vacates it.
            let ortho = (pt_bit & ORTHO_MASK) != 0 && (dx == 0 || dy == 0) && (adx + ady) > 0;
            let diag = (pt_bit & DIAG_MASK) != 0 && adx == ady && adx > 0;
            if ortho || diag {
                let to_sq = crate::board::Coordinate::new(tx, ty);
                if Self::ray_clear_after_move(game, &to_sq, king_pos, &m.from) {
                    return true;
                }
            }
        }

        false
    }

    /// Continuation-history slots for this ply. Built before the killer stages as
    /// well as the quiet one, or a killer searched late reads an empty list and gets
    /// none of the reduction relief every other quiet move gets.
    fn ensure_cont_history_indices(&mut self, searcher: &Searcher) {
        if !self.cont_history_indices.is_empty() {
            return;
        }
        let ply = self.ply;
        let offsets = [1usize, 2, 4];
        for (idx, &plies_ago) in offsets.iter().enumerate() {
            if let Some(prev_idx) = ply.checked_sub(plies_ago)
                && let Some(Some(prev_move)) = searcher.move_history.get(prev_idx)
                && let Some(&prev_piece) = searcher.moved_piece_history.get(prev_idx)
            {
                let prev_piece = prev_piece as usize;
                if prev_piece < 32 {
                    let prev_to_h = hash_coord_16(prev_move.to.x, prev_move.to.y);
                    let prev_ic = searcher.in_check_history[prev_idx] as usize;
                    let prev_cap = searcher.capture_history_stack[prev_idx] as usize;
                    self.cont_history_indices
                        .push((idx, prev_cap, prev_ic, prev_piece, prev_to_h));
                }
            }
        }
    }

    /// Compute and cache the side-to-move pin map, shared by the capture and
    /// quiet stages.
    fn ensure_pins(&mut self, game: &GameState) {
        if self.pins_cache.is_none() {
            let king_pos = if game.turn == PlayerColor::White {
                game.white_royals.first().copied()
            } else {
                game.black_royals.first().copied()
            };
            self.pins_cache = Some(match king_pos {
                Some(kp) => game.compute_pins(&kp, game.turn),
                None => rustc_hash::FxHashMap::default(),
            });
        }
    }

    fn generate_captures(&mut self, game: &GameState, searcher: &Searcher) {
        let mut captures = MoveList::new();

        // Only quiet slider generation reads the pin map, so the capture stage
        // pays nothing for one.
        let no_pins = rustc_hash::FxHashMap::default();
        {
            let ctx = MoveGenContext {
                pinned: &no_pins,
                special_rights: &game.special_rights,
                en_passant: &game.en_passant,
                game_rules: &game.game_rules,
                indices: &game.spatial_indices,
                enemy_king_pos: game.enemy_king_pos(),
            };
            // Routed on the position-derived eval_kind, not the [Variant] tag, so an
            // untagged obstacle board gets the same treatment a tagged one does.
            if game.eval_kind == crate::evaluation::eval_kind::EvalKind::Obstocean {
                crate::evaluation::variants::obstocean_search::get_quiescence_captures(
                    &game.board,
                    game.turn,
                    &ctx,
                    &mut captures,
                );
            } else {
                get_quiescence_captures(&game.board, game.turn, &ctx, &mut captures);
            }
        }

        for &m in captures.iter() {
            if self.is_tt_move(&m) || self.is_excluded(&m) {
                continue;
            }
            let score = Self::score_capture(game, searcher, &m);
            self.moves.push(ScoredMove {
                m,
                score,
                gives_check: None,
            });
        }
    }

    fn generate_quiets(&mut self, game: &GameState, searcher: &Searcher) {
        let mut quiets = MoveList::new();

        self.ensure_pins(game);
        {
            let ctx = MoveGenContext {
                pinned: self.pins_cache.as_ref().unwrap(),
                special_rights: &game.special_rights,
                en_passant: &game.en_passant,
                game_rules: &game.game_rules,
                indices: &game.spatial_indices,
                enemy_king_pos: game.enemy_king_pos(),
            };
            // Depth-staged tight generation: shallow subtrees can't use the
            // deep quiet-slider tail (played-move audit: rank<=3 covers 67%),
            // so cap candidates per ray there; full width at deep nodes.
            let tight = self.depth <= TIGHT_GEN_DEPTH;
            if tight {
                crate::moves::set_quiet_ray_cap(TIGHT_GEN_RAY_CAP);
            }
            get_quiet_moves_into(&game.board, game.turn, &ctx, &mut quiets);
            if tight {
                crate::moves::set_quiet_ray_cap(0);
            }
        }

        for &m in quiets.iter() {
            if self.is_tt_move(&m)
                || self.is_excluded(&m)
                || Self::moves_match(&m, &self.killer1)
                || Self::moves_match(&m, &self.killer2)
            {
                continue;
            }
            let (score, gives_check) = self.score_quiet(game, searcher, &m);
            self.moves.push(ScoredMove {
                m,
                score,
                gives_check,
            });
        }
    }

    fn generate_evasions(&mut self, game: &GameState, searcher: &Searcher) {
        let mut evasions = MoveList::new();
        game.get_evasion_moves_into(&mut evasions);

        for &m in evasions.iter() {
            if self.is_tt_move(&m) || self.is_excluded(&m) {
                continue;
            }
            let (score, gives_check) = self.score_evasion(game, searcher, &m);
            self.moves.push(ScoredMove {
                m,
                score,
                gives_check,
            });
        }
    }

    /// Get next move using multi-stage generation
    /// Gives-check bit for the move `next` just returned, when quiet scoring
    /// already computed it. Valid until the next `next` call.
    #[inline]
    pub fn cached_gives_check(&self) -> Option<bool> {
        self.cached_gives_check
    }

    pub fn next(&mut self, game: &GameState, searcher: &Searcher) -> Option<Move> {
        self.cached_gives_check = None;
        loop {
            match self.stage {
                MoveStage::MainTT
                | MoveStage::EvasionTT
                | MoveStage::QSearchTT
                | MoveStage::ProbCutTT => {
                    // Advance to next stage
                    self.stage = match self.stage {
                        MoveStage::MainTT => MoveStage::CaptureInit,
                        MoveStage::EvasionTT => MoveStage::EvasionInit,
                        MoveStage::QSearchTT => MoveStage::QCaptureInit,
                        MoveStage::ProbCutTT => MoveStage::ProbCutInit,
                        _ => unreachable!(),
                    };

                    // Return TT move (already validated in constructor)
                    if let Some(tt_m) = self.tt_move
                        && !self.is_excluded(&tt_m)
                    {
                        return Some(tt_m);
                    }
                }

                MoveStage::CaptureInit | MoveStage::QCaptureInit | MoveStage::ProbCutInit => {
                    self.generate_captures(game, searcher);

                    self.cur = 0;
                    self.end_bad_captures = 0;
                    self.end_captures = self.moves.len();

                    // Sort all captures (limit = MIN to include all)
                    partial_insertion_sort(&mut self.moves, i32::MIN);

                    self.stage = match self.stage {
                        MoveStage::CaptureInit => MoveStage::GoodCapture,
                        MoveStage::QCaptureInit => MoveStage::QCapture,
                        MoveStage::ProbCutInit => MoveStage::ProbCut,
                        _ => unreachable!(),
                    };
                }

                MoveStage::GoodCapture => {
                    while self.cur < self.end_captures {
                        let sm = self.moves[self.cur];

                        if super::see_ge(game, &sm.m, -18) {
                            self.cur += 1;
                            return Some(sm.m);
                        } else {
                            // Bad capture - swap to front for later
                            self.moves.swap(self.end_bad_captures, self.cur);
                            self.end_bad_captures += 1;
                        }
                        self.cur += 1;
                    }

                    self.stage = MoveStage::Killer1;
                }

                MoveStage::Killer1 => {
                    self.stage = MoveStage::Killer2;
                    // Lazy Killer 1
                    if self.skip_quiets {
                        continue;
                    }
                    self.ensure_cont_history_indices(searcher);

                    if let Some(m) = self.killer1
                        && !self.is_tt_move(&m)
                        && !self.is_excluded(&m)
                        && !Self::is_capture(game, &m)
                        && !game.is_en_passant(&m)
                        && Self::is_pseudo_legal(game, &m)
                    {
                        return Some(m);
                    }
                }

                MoveStage::Killer2 => {
                    self.stage = MoveStage::QuietInit;
                    // Lazy Killer 2
                    if self.skip_quiets {
                        continue;
                    }

                    if let Some(m) = self.killer2
                        && !self.is_tt_move(&m)
                        && !self.is_excluded(&m)
                        && !Self::moves_match(&m, &self.killer1)
                        && !Self::is_capture(game, &m)
                        && !game.is_en_passant(&m)
                        && Self::is_pseudo_legal(game, &m)
                    {
                        return Some(m);
                    }
                }

                MoveStage::QuietInit => {
                    if self.skip_quiets {
                        // Prepare for bad captures
                        self.cur = 0;
                        self.stage = MoveStage::BadCapture;
                        continue;
                    }

                    let quiet_start = self.moves.len();

                    self.ensure_cont_history_indices(searcher);

                    self.generate_quiets(game, searcher);
                    self.end_generated = self.moves.len();

                    // Partial sort with depth-based limit
                    let limit = -3560 * self.depth;
                    partial_insertion_sort(&mut self.moves[quiet_start..], limit);

                    self.cur = quiet_start;
                    self.end_bad_quiets = quiet_start;
                    self.stage = MoveStage::GoodQuiet;
                }

                MoveStage::GoodQuiet => {
                    if self.skip_quiets {
                        self.cur = 0;
                        self.stage = MoveStage::BadCapture;
                        continue;
                    }

                    while self.cur < self.end_generated {
                        if self.moves[self.cur].score > GOOD_QUIET_THRESHOLD {
                            let sm = self.moves[self.cur];
                            self.cur += 1;
                            self.cached_gives_check = sm.gives_check;
                            return Some(sm.m);
                        }
                        // Bad quiet - swap to the front of the quiet span for later
                        self.moves.swap(self.end_bad_quiets, self.cur);
                        self.end_bad_quiets += 1;
                        self.cur += 1;
                    }

                    // Prepare for bad captures
                    self.cur = 0;
                    self.stage = MoveStage::BadCapture;
                }

                MoveStage::BadCapture => {
                    if self.cur < self.end_bad_captures {
                        let m = self.moves[self.cur].m;
                        self.cur += 1;
                        return Some(m);
                    }

                    // Prepare for bad quiets
                    self.cur = self.end_captures;
                    self.stage = MoveStage::BadQuiet;
                }

                MoveStage::BadQuiet => {
                    if self.skip_quiets {
                        self.stage = MoveStage::Done;
                        return None;
                    }

                    if self.cur < self.end_bad_quiets {
                        let sm = self.moves[self.cur];
                        self.cur += 1;
                        self.cached_gives_check = sm.gives_check;
                        return Some(sm.m);
                    }

                    self.stage = MoveStage::Done;
                }

                MoveStage::EvasionInit => {
                    self.generate_evasions(game, searcher);
                    self.end_generated = self.moves.len();
                    self.cur = 0;

                    partial_insertion_sort(&mut self.moves, i32::MIN);

                    self.stage = MoveStage::Evasion;
                }

                MoveStage::Evasion | MoveStage::QCapture => {
                    if self.cur < self.end_generated.max(self.end_captures) {
                        let sm = self.moves[self.cur];
                        self.cur += 1;
                        self.cached_gives_check = sm.gives_check;
                        return Some(sm.m);
                    }
                    self.stage = MoveStage::Done;
                }

                MoveStage::ProbCut => {
                    while self.cur < self.end_captures {
                        let sm = self.moves[self.cur];
                        self.cur += 1;

                        if super::see_ge(game, &sm.m, self.threshold) {
                            return Some(sm.m);
                        }
                    }
                    self.stage = MoveStage::Done;
                }

                MoveStage::Done => {
                    return None;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Coordinate, Piece, PieceType};
    use crate::search::Searcher;

    /// A castling partner may stand any distance away on the row, so validation must
    /// use the same rules the generator does: rights on both ends, at least three
    /// apart, and nothing in between. It previously demanded the square three files
    /// out be empty, which is the partner itself at the minimum legal distance.
    #[test]
    fn pseudo_legal_accepts_castling_with_a_partner_three_squares_away() {
        let game = game_from_icn("w 0/100 1 K5,1+|R2,1+|k5,8+");
        let mut moves = crate::moves::MoveList::new();
        game.get_pseudo_legal_moves_into(&mut moves);
        let castle = moves
            .iter()
            .find(|m| m.from == Coordinate::new(5, 1) && m.to == Coordinate::new(3, 1))
            .copied()
            .expect("generator emits the castle with a partner three away");
        assert!(
            StagedMoveGen::is_pseudo_legal(&game, &castle),
            "a legal castle was rejected, so a killer naming it would be dropped"
        );

        // A piece between the two ends blocks it, however far apart they are.
        let blocked = game_from_icn("w 0/100 1 K5,1+|R1,1+|N3,1|k5,8+");
        assert!(
            !StagedMoveGen::is_pseudo_legal(&blocked, &castle),
            "castling must be rejected when a piece stands between king and partner"
        );
    }

    fn game_from_icn(icn: &str) -> GameState {
        let mut game = GameState::new();
        game.setup_position_from_icn(icn);
        game
    }

    fn find_move(game: &GameState, from: (i64, i64), to: (i64, i64)) -> Move {
        game.get_pseudo_legal_moves()
            .into_iter()
            .find(|m| m.from.x == from.0 && m.from.y == from.1 && m.to.x == to.0 && m.to.y == to.1)
            .unwrap()
    }

    #[test]
    fn quiet_queen_promotion_orders_above_a_pawn_capture() {
        // A quiet promotion has no victim on its target square; it must still
        // sort by what it wins, not fall through to the zero score.
        let game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|k5,8|P4,7|P1,4|p2,5");
        let searcher = Searcher::new(1_000);
        let promo = find_move(&game, (4, 7), (4, 8));
        assert!(promo.promotion.is_some());
        let pxp = find_move(&game, (1, 4), (2, 5));
        let promo_score = StagedMoveGen::score_capture(&game, &searcher, &promo);
        let pxp_score = StagedMoveGen::score_capture(&game, &searcher, &pxp);
        assert!(
            promo_score > pxp_score,
            "quiet promotion {promo_score} must outrank PxP {pxp_score}"
        );
    }

    #[test]
    fn pseudo_legal_accepts_pawn_push_capture_and_en_passant() {
        let push_game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|k5,8|P4,2");
        let push = find_move(&push_game, (4, 2), (4, 3));
        assert!(StagedMoveGen::is_pseudo_legal(&push_game, &push));

        let capture_game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|k5,8|P4,4|p5,5");
        let capture = find_move(&capture_game, (4, 4), (5, 5));
        assert!(StagedMoveGen::is_pseudo_legal(&capture_game, &capture));

        let ep_game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|k5,8|P5,5|p6,5 6,6");
        let ep = find_move(&ep_game, (5, 5), (6, 6));
        assert!(StagedMoveGen::is_pseudo_legal(&ep_game, &ep));
    }

    #[test]
    fn pseudo_legal_rejects_wrong_color_and_blocked_slider() {
        let game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|k5,8|r4,4|P4,5");

        let wrong_color = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(4, 1),
            Piece::new(PieceType::Rook, PlayerColor::Black),
        );
        assert!(!StagedMoveGen::is_pseudo_legal(&game, &wrong_color));

        let blocked_rook = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(4, 8),
            Piece::new(PieceType::Rook, PlayerColor::Black),
        );
        assert!(!StagedMoveGen::is_pseudo_legal(&game, &blocked_rook));
    }

    #[test]
    fn pseudo_legal_handles_knight_and_castling_rules() {
        let knight_game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|k5,8|N4,4");
        let knight = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(5, 6),
            Piece::new(PieceType::Knight, PlayerColor::White),
        );
        assert!(StagedMoveGen::is_pseudo_legal(&knight_game, &knight));

        let castle_game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1+|R8,1+|k5,8");
        let castle = castle_game
            .get_pseudo_legal_moves()
            .into_iter()
            .find(|m| m.piece.piece_type() == PieceType::King && (m.to.x - m.from.x).abs() > 1)
            .unwrap();
        assert!(StagedMoveGen::is_pseudo_legal(&castle_game, &castle));
    }

    #[test]
    fn move_gives_check_fast_detects_knight_slider_and_empty_royals() {
        let knight_game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|k5,8|N2,5");
        let knight = Move::new(
            Coordinate::new(2, 5),
            Coordinate::new(4, 6),
            Piece::new(PieceType::Knight, PlayerColor::White),
        );
        assert!(StagedMoveGen::move_gives_check_fast(&knight_game, &knight));

        let rook_game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|k5,8|R1,8");
        let rook = Move::new(
            Coordinate::new(1, 8),
            Coordinate::new(4, 8),
            Piece::new(PieceType::Rook, PlayerColor::White),
        );
        assert!(StagedMoveGen::move_gives_check_fast(&rook_game, &rook));

        let no_enemy_royal = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|Q4,4");
        let queen = Move::new(
            Coordinate::new(4, 4),
            Coordinate::new(4, 5),
            Piece::new(PieceType::Queen, PlayerColor::White),
        );
        assert!(!StagedMoveGen::move_gives_check_fast(
            &no_enemy_royal,
            &queen
        ));
    }

    #[test]
    fn move_gives_check_fast_detects_fairy_leapers() {
        // Each lands on its own leap offset from k5,8; a miss here means the check
        // is scored as a quiet and can be futility/history-pruned or cut in qsearch.
        type Case = (&'static str, PieceType, (i64, i64), (i64, i64));
        let cases: [Case; 5] = [
            ("w 0/100 1 (8;q|1;q) K5,1|k5,8|CA1,1", PieceType::Camel, (1, 1), (6, 5)),
            ("w 0/100 1 (8;q|1;q) K5,1|k5,8|GI1,1", PieceType::Giraffe, (1, 1), (6, 4)),
            ("w 0/100 1 (8;q|1;q) K5,1|k5,8|ZE1,1", PieceType::Zebra, (1, 1), (7, 5)),
            ("w 0/100 1 (8;q|1;q) K5,1|k5,8|HA1,1", PieceType::Hawk, (1, 1), (5, 5)),
            ("w 0/100 1 (8;q|1;q) K5,1|k5,8|GU1,1", PieceType::Guard, (1, 1), (5, 7)),
        ];
        for (icn, pt, from, to) in cases {
            let game = game_from_icn(icn);
            let m = Move::new(
                Coordinate::new(from.0, from.1),
                Coordinate::new(to.0, to.1),
                Piece::new(pt, PlayerColor::White),
            );
            assert!(
                StagedMoveGen::move_gives_check_fast(&game, &m),
                "{pt:?} moving to {to:?} attacks k5,8 but was not detected"
            );
        }
    }

    #[test]
    fn tt_castling_move_survives_missing_partner_coord() {
        // King e1 and rook h1 both retain rights; kingside castling is legal.
        let game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1+|R8,1+|k5,8");

        let castle = game
            .get_pseudo_legal_moves()
            .into_iter()
            .find(|m| m.piece.piece_type() == PieceType::King && (m.to.x - m.from.x).abs() == 2)
            .expect("kingside castling should be legal");
        assert!(
            castle.partner_x != crate::moves::NO_PARTNER,
            "generated castling has a partner"
        );

        // Simulate the TT/killer round-trip, which drops the rook partner.
        let tt_decoded = Move {
            partner_x: crate::moves::NO_PARTNER,
            ..castle
        };

        let searcher = Searcher::new(1000);
        let mut picker = StagedMoveGen::new(Some(tt_decoded), 0, 2, &searcher, &game);

        let mut found = false;
        while let Some(m) = picker.next(&game, &searcher) {
            if m.from == castle.from && m.to == castle.to {
                found = true;
                assert!(
                    m.partner_x != crate::moves::NO_PARTNER,
                    "emitted castling move lost its rook partner"
                );
            }
        }
        assert!(found, "castling move was dropped when it was the TT move");
    }

    #[test]
    fn pseudo_legal_rejects_pawn_moves_onto_or_through_void() {
        // A neutral Void packs to byte 0 but occupies its square. A packed-only
        // emptiness test would let a (stale TT/killer) pawn push land on or pass
        // through it; occupancy-based checks must reject that.
        let game = game_from_icn("w 0/100 1 (8;q|1;q) K1,1|P4,4|k8,8|VO4,5");
        assert!(
            game.board.is_occupied(4, 5),
            "void should occupy its square"
        );

        let pawn = Piece::new(PieceType::Pawn, PlayerColor::White);
        let push = Move::new(Coordinate::new(4, 4), Coordinate::new(4, 5), pawn);
        let double = Move::new(Coordinate::new(4, 4), Coordinate::new(4, 6), pawn);
        assert!(
            !StagedMoveGen::is_pseudo_legal(&game, &push),
            "pawn must not push onto a void"
        );
        assert!(
            !StagedMoveGen::is_pseudo_legal(&game, &double),
            "pawn must not push through a void"
        );

        // Control: with the square clear the single push is pseudo-legal, and the
        // double push is too once the pawn actually holds its double-step right.
        let clear = game_from_icn("w 0/100 1 (8;q|1;q) K1,1|P4,4+|k8,8");
        assert!(StagedMoveGen::is_pseudo_legal(&clear, &push));
        assert!(StagedMoveGen::is_pseudo_legal(&clear, &double));

        // Without that right the generator never emits the double step, so a stale
        // killer naming it must not be replayed either.
        let no_right = game_from_icn("w 0/100 1 (8;q|1;q) K1,1|P4,4|k8,8");
        assert!(StagedMoveGen::is_pseudo_legal(&no_right, &push));
        assert!(
            !StagedMoveGen::is_pseudo_legal(&no_right, &double),
            "a pawn with no double-step right must not double-push"
        );
    }

    #[test]
    fn staged_movegen_can_skip_quiets_and_respect_exclusion() {
        let game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|Q4,4|k5,8|p4,7");
        let searcher = Searcher::new(1000);
        let capture = find_move(&game, (4, 4), (4, 7));

        let mut skip = StagedMoveGen::new(None, 0, 2, &searcher, &game);
        skip.skip_quiet_moves();
        let first = skip.next(&game, &searcher).unwrap();
        assert_eq!(first.to, capture.to);
        assert!(skip.next(&game, &searcher).is_none());

        let mut excluded = StagedMoveGen::with_exclusion(None, 0, 2, &searcher, &game, capture);
        let generated: Vec<_> = std::iter::from_fn(|| excluded.next(&game, &searcher)).collect();
        assert!(generated.iter().all(|m| m.to != capture.to));
    }

    #[test]
    fn staged_movegen_prefers_tt_move_and_probcut_filters_bad_see() {
        let game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|Q4,4|k5,8|p4,7");
        let searcher = Searcher::new(1000);
        let tt_move = find_move(&game, (4, 4), (4, 7));

        let mut picker = StagedMoveGen::new(Some(tt_move), 0, 2, &searcher, &game);
        assert_eq!(picker.next(&game, &searcher).unwrap().to, tt_move.to);

        let mut probcut = StagedMoveGen::new_probcut(None, 10_000, &searcher, &game);
        assert!(probcut.next(&game, &searcher).is_none());
    }

    #[test]
    fn staged_movegen_uses_evasion_stages_when_in_check() {
        let game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|k5,8|r5,4");
        assert!(game.is_in_check());
        assert!(game.must_escape_check());

        let searcher = Searcher::new(1000);
        let mut picker = StagedMoveGen::new(None, 0, 2, &searcher, &game);
        assert_eq!(picker.stage, MoveStage::EvasionInit);

        let first = picker.next(&game, &searcher).unwrap();
        let mut check_game = game.clone();
        let undo = check_game.make_move(&first);
        assert!(!check_game.is_move_illegal());
        check_game.undo_move(&first, undo);
    }

    #[test]
    fn staged_movegen_scores_countermove_and_killers_as_quiets() {
        let game = game_from_icn("w 0/100 1 (8;q|1;q) K5,1|R1,1|k5,8");
        let quiet = find_move(&game, (1, 1), (1, 2));
        let mut searcher = Searcher::new(1000);

        searcher.killers[0][0] = Some(quiet);
        let picker = StagedMoveGen::new(None, 0, 2, &searcher, &game);
        assert_eq!(picker.score_quiet(&game, &searcher, &quiet).0, sort_killer1());

        let mut searcher = Searcher::new(1000);
        searcher.prev_move_stack[0] = (3, 9);
        searcher.countermoves[3][9] = (
            quiet.piece.piece_type() as u8,
            quiet.to.x as i32,
            quiet.to.y as i32,
        );
        let picker = StagedMoveGen::new(None, 1, 2, &searcher, &game);
        assert!(picker.score_quiet(&game, &searcher, &quiet).0 >= sort_countermove());
    }
}
