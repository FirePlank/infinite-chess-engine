use crate::board::{Coordinate, Piece, PieceType, PlayerColor};
use crate::game::GameState;
use crate::moves::Move;

use super::{INFINITY, MATE_SCORE, MATE_VALUE};

/// Value adjustment for storage:
/// Adjusts a mate score from "plies to mate from the root" to
/// "plies to mate from the current position" for storage in TT.
/// Standard scores are unchanged.
#[inline]
pub fn value_to_tt(value: i32, ply: usize) -> i32 {
    // is_win: value > MATE_SCORE (positive mate score)
    if value > MATE_SCORE {
        value + ply as i32
    }
    // is_loss: value < -MATE_SCORE (negative mate score, being mated)
    else if value < -MATE_SCORE {
        value - ply as i32
    } else {
        value
    }
}

pub struct TTProbeParams {
    pub hash: u64,
    pub alpha: i32,
    pub beta: i32,
    pub depth: usize,
    pub ply: usize,
    pub rule50_count: u32,
    pub rule_limit: i32,
}

pub struct TTStoreParams {
    pub hash: u64,
    pub depth: usize,
    pub flag: TTFlag,
    pub score: i32,
    pub best_move: Option<Move>,
    pub ply: usize,
}

/// Value adjustment for retrieval:
/// Inverse of value_to_tt: adjusts TT score back to root-relative.
/// Downgrades mate scores that are unreachable due to the 50-move rule.
///
/// Parameters:
/// - value: The score stored in TT
/// - ply: Current search ply
/// - rule50_count: Current halfmove clock (NOT remaining moves)
/// - rule_limit: Current move rule limit (e.g. 100 for 50-move rule)
#[inline]
pub fn value_from_tt(value: i32, ply: usize, rule50_count: u32, rule_limit: i32) -> i32 {
    // Handle winning mate scores (we are giving mate)
    if value > MATE_SCORE {
        // mate_distance = how many plies until mate from the stored position
        let mate_distance = MATE_VALUE - value;

        // Or: mate_distance + rule50_count > rule_limit
        if mate_distance + rule50_count as i32 > rule_limit {
            // Downgrade to non-mate winning score (just below mate threshold)
            return MATE_SCORE - 1;
        }

        // Adjust back to root-relative
        return value - ply as i32;
    }

    // Handle losing mate scores (we are being mated)
    if value < -MATE_SCORE {
        let mate_distance = MATE_VALUE + value;

        if mate_distance + rule50_count as i32 > rule_limit {
            // Downgrade to non-mate losing score
            return -MATE_SCORE + 1;
        }

        return value + ply as i32;
    }

    value
}

// ============================================================================
// Constants
// ============================================================================

/// Number of entries per bucket (cluster). 3 entries × 64 bytes = 192 bytes.
/// We use 3 entries to allow for larger 64-byte entries (storing full 64-bit hash).
const ENTRIES_PER_BUCKET: usize = 3;

/// Sentinel value indicating no move is stored
const NO_MOVE_SENTINEL: i64 = i64::MIN;

// ============================================================================
// TT Entry Flags (packed into 2 bits)
// ============================================================================

/// TT bound type (2 bits)
#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
pub enum TTFlag {
    None = 0,       // Invalid/empty entry
    Exact = 1,      // Exact score (PV node)
    LowerBound = 2, // Score >= beta (cut node, failed high)
    UpperBound = 3, // Score <= alpha (all node, failed low)
}

impl TTFlag {
    #[inline]
    fn from_u8(v: u8) -> Self {
        unsafe { std::mem::transmute(v & 0b11) }
    }
}

// ============================================================================
// Compact Move Representation
// ============================================================================

/// Compact move for TT storage. Uses i64 coordinates (required for infinite chess)
/// but removes Option overhead by using sentinel values.
///
/// Size: 40 bytes (from: 16 + to: 16 + piece_type: 1 + piece_color: 1 + promotion: 1 + padding: 5)
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TTMove {
    from_x: i64,
    from_y: i64,
    to_x: i64,
    to_y: i64,
    piece_type: u8,
    piece_color: u8,
    promotion: u8, // 0 = none, else PieceType as u8
}

impl TTMove {
    /// Create a "no move" sentinel
    #[inline]
    pub const fn none() -> Self {
        TTMove {
            from_x: NO_MOVE_SENTINEL,
            from_y: 0,
            to_x: 0,
            to_y: 0,
            piece_type: 0,
            piece_color: 0,
            promotion: 0,
        }
    }

    /// Check if this represents a valid move
    #[inline]
    pub fn is_some(&self) -> bool {
        self.from_x != NO_MOVE_SENTINEL
    }

    /// Create from a Move
    #[inline]
    pub fn from_move(m: &Move) -> Self {
        TTMove {
            from_x: m.from.x,
            from_y: m.from.y,
            to_x: m.to.x,
            to_y: m.to.y,
            piece_type: m.piece.piece_type() as u8,
            piece_color: m.piece.color() as u8,
            promotion: m.promotion.map_or(0, |p| p as u8),
        }
    }

    /// Convert back to Option<Move> for use in search
    #[inline]
    pub fn to_move(self) -> Option<Move> {
        if !self.is_some() {
            return None;
        }

        // Reconstruct piece type and color using safe conversion
        let piece_type = PieceType::from_u8(self.piece_type);
        let color = PlayerColor::from_u8(self.piece_color);
        let piece = Piece::new(piece_type, color);

        // Reconstruct promotion
        let promotion = if self.promotion == 0 {
            None
        } else {
            Some(unsafe { std::mem::transmute::<u8, PieceType>(self.promotion) })
        };

        Some(Move {
            from: Coordinate {
                x: self.from_x,
                y: self.from_y,
            },
            to: Coordinate {
                x: self.to_x,
                y: self.to_y,
            },
            piece,
            promotion,
            rook_coord: None, // TT doesn't store rook_coord; is_pseudo_legal validates
        })
    }

    /// Check if this TTMove matches a given Move (for move ordering)
    #[inline]
    #[allow(dead_code)]
    pub fn matches(&self, m: &Move) -> bool {
        self.is_some()
            && self.from_x == m.from.x
            && self.from_y == m.from.y
            && self.to_x == m.to.x
            && self.to_y == m.to.y
    }
}

// ============================================================================
// TT Entry (Single Entry in a Bucket)
// ============================================================================

/// Transposition Table entry - optimized for memory efficiency.
///
/// Layout (64 bytes total):
/// - key: u64         - Full hash key (8 bytes)
/// - score: i32       - Evaluation score (4 bytes)
/// - depth: u8        - Search depth (1 byte)
/// - gen_bound: u8    - Generation (6 bits) + Bound type (2 bits) (1 byte)
/// - padding: [u8; 10]- Padding to align to 64 bytes/cache line
/// - tt_move: TTMove  - Best move (40 bytes)
#[derive(Clone, Copy, Debug)]
#[repr(C)] // Ensure C layout for reliable size
pub struct TTEntry {
    /// Full 64-bit hash key for verification
    key: u64,
    /// Score from the search (with mate score adjustment for storage)
    score: i32,
    /// Search depth that produced this result
    depth: u8,
    /// Packed: generation (upper 6 bits) + bound type (lower 2 bits)
    gen_bound: u8,
    /// Padding to reach 64 bytes (8+4+1+1+10+40 = 64)
    /// Also ensures alignment if needed.
    _padding: [u8; 10],
    /// Best move found (or sentinel for none)
    tt_move: TTMove,
}

impl TTEntry {
    /// Create an empty/invalid entry
    #[inline]
    pub const fn empty() -> Self {
        TTEntry {
            key: 0,
            score: 0,
            depth: 0,
            gen_bound: 0,
            _padding: [0; 10],
            tt_move: TTMove::none(),
        }
    }

    /// Check if entry is empty (never written to)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.gen_bound == 0 && self.key == 0
    }

    /// Extract the bound type from gen_bound
    #[inline]
    pub fn flag(&self) -> TTFlag {
        TTFlag::from_u8(self.gen_bound)
    }

    /// Extract the generation from gen_bound
    #[inline]
    pub fn generation(&self) -> u8 {
        self.gen_bound >> 2
    }

    /// Create packed gen_bound from generation and flag
    #[inline]
    fn pack_gen_bound(generation: u8, flag: TTFlag) -> u8 {
        (generation << 2) | (flag as u8)
    }

    /// Get the best move as Option<Move>
    #[inline]
    pub fn best_move(&self) -> Option<Move> {
        self.tt_move.to_move()
    }
}

// ============================================================================
// TT Bucket (Cluster of Entries)
// ============================================================================

/// A bucket/cluster containing multiple TT entries.
/// This improves collision handling - when storing, we pick the least valuable
/// entry in the bucket to replace.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct TTBucket {
    entries: [TTEntry; ENTRIES_PER_BUCKET],
}

impl TTBucket {
    #[inline]
    pub const fn empty() -> Self {
        TTBucket {
            entries: [TTEntry::empty(); ENTRIES_PER_BUCKET],
        }
    }
}

// ============================================================================
// Transposition Table
// ============================================================================

/// Transposition Table with bucket-based collision handling.
///
/// Key optimizations:
/// - Bucket system: 3 entries per index reduces effective collision rate
/// - Align entries to cache lines (64 bytes)
/// - Store FULL 64-bit hash key to prevent collisions
/// - Power-of-two sizing for fast index calculation
pub struct TranspositionTable {
    buckets: Vec<TTBucket>,
    /// Bitmask for indexing (capacity is always power of two)
    mask: usize,
    /// Current generation (incremented each search)
    generation: u8,
    /// Number of entries currently stored (for fill percentage)
    used: usize,
}

impl TranspositionTable {
    /// Create a new TT with approximately `size_mb` megabytes of storage.
    /// For WASM builds, the size is capped at 64MB to avoid browser memory limits.
    pub fn new(size_mb: usize) -> Self {
        // Cap size for WASM to stay within browser memory constraints
        #[cfg(target_arch = "wasm32")]
        let size_mb = size_mb.min(64);

        let bytes = size_mb.max(1) * 1024 * 1024;
        let bucket_size = std::mem::size_of::<TTBucket>();
        let num_buckets = (bytes / bucket_size).max(1);

        // Round down to power of two for efficient masking
        let mut cap_pow2 = 1usize;
        while cap_pow2 * 2 <= num_buckets {
            cap_pow2 *= 2;
        }

        TranspositionTable {
            buckets: vec![TTBucket::empty(); cap_pow2],
            mask: cap_pow2 - 1,
            generation: 1, // Start at 1 so 0 indicates empty
            used: 0,
        }
    }

    /// Get the hash for the current position
    #[inline]
    pub fn generate_hash(game: &GameState) -> u64 {
        game.hash
    }

    /// Number of buckets in the table
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buckets.len() * ENTRIES_PER_BUCKET
    }

    /// Number of entries currently stored
    #[inline]
    pub fn used_entries(&self) -> usize {
        self.used
    }

    /// Fill percentage in permille (0-1000)
    #[inline]
    pub fn fill_permille(&self) -> u32 {
        let capacity = self.capacity();
        if capacity == 0 {
            return 0;
        }
        ((self.used as u64 * 1000) / capacity as u64) as u32
    }

    /// Calculate bucket index from hash
    #[inline]
    fn bucket_index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    /// Prefetch the TT bucket for the given hash into L1 cache.
    /// This should be called BEFORE making a move, so the TT entry
    /// for the child position is already in cache when we probe it.
    ///
    /// On x86_64: uses _mm_prefetch with _MM_HINT_T0 (L1 cache)
    /// On other: no-op
    #[inline]
    #[cfg(all(target_arch = "x86_64", not(target_arch = "wasm32")))]
    pub fn prefetch_entry(&self, hash: u64) {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        let idx = self.bucket_index(hash);
        let ptr = self.buckets.as_ptr().wrapping_add(idx) as *const i8;
        // SAFETY: ptr points into a valid, allocated slice
        unsafe { _mm_prefetch(ptr, _MM_HINT_T0) };
    }

    /// Probe the TT for a position.
    ///
    /// Returns `Some((score, best_move))` where:
    /// - If `score` is usable for cutoff (not `INFINITY + 1`), use it directly.
    /// - If `score == INFINITY + 1`, only the move is usable (for ordering).
    pub fn probe(&self, params: &TTProbeParams) -> Option<(i32, Option<Move>)> {
        let hash = params.hash;
        let alpha = params.alpha;
        let beta = params.beta;
        let depth = params.depth;
        let ply = params.ply;
        let rule50_count = params.rule50_count;
        let rule_limit = params.rule_limit;

        let idx = self.bucket_index(hash);
        let bucket = &self.buckets[idx];

        // Search all entries in the bucket for a match
        for entry in &bucket.entries {
            // Check full 64-bit key or if empty
            if entry.key != hash || entry.is_empty() {
                continue;
            }

            // Found a matching entry
            let best_move = entry.best_move();

            // Only use score if depth is sufficient
            if entry.depth as usize >= depth {
                // Adjust score from TT to search value, handling 50-move rule
                let score = value_from_tt(entry.score, ply, rule50_count, rule_limit);

                // Check if we can use this score for a cutoff
                let usable_score = match entry.flag() {
                    TTFlag::Exact => Some(score),
                    TTFlag::LowerBound if score >= beta => Some(score),
                    TTFlag::UpperBound if score <= alpha => Some(score),
                    _ => None,
                };

                if let Some(s) = usable_score {
                    return Some((s, best_move));
                }
            }

            // Depth insufficient or bounds don't allow cutoff, but move is still useful
            return Some((INFINITY + 1, best_move));
        }

        None
    }

    /// Probe the TT for Singular Extension data.
    /// Returns raw entry data without applying cutoff logic.
    /// Returns `Some((flag, depth, score, move))` if a matching entry exists.
    pub fn probe_for_singular(
        &self,
        hash: u64,
        ply: usize,
    ) -> Option<(TTFlag, u8, i32, Option<Move>)> {
        let idx = self.bucket_index(hash);
        let bucket = &self.buckets[idx];

        for entry in &bucket.entries {
            if entry.key != hash || entry.is_empty() {
                continue;
            }

            // Adjust mate scores for current ply
            let mut score = entry.score;
            if score > MATE_SCORE {
                score -= ply as i32;
            } else if score < -MATE_SCORE {
                score += ply as i32;
            }

            return Some((entry.flag(), entry.depth, score, entry.best_move()));
        }

        None
    }

    /// Store an entry in the TT.
    ///
    /// Uses a smart replacement strategy within the bucket:
    /// 1. If we find our position, always update it
    /// 2. Otherwise, find the least valuable
    pub fn store(&mut self, params: &TTStoreParams) {
        let hash = params.hash;
        let depth = params.depth;
        let flag = params.flag;
        let score = params.score;
        let best_move = params.best_move;
        let ply = params.ply;

        // Adjust mate scores for storage
        let adjusted_score = value_to_tt(score, ply);

        let idx = self.bucket_index(hash);
        let generation = self.generation;
        let bucket = &mut self.buckets[idx];

        // Find the best slot to use (search for existing entry or worst victim)
        let mut replace_idx = 0;
        let mut worst_score = i32::MAX;

        for (i, entry) in bucket.entries.iter_mut().enumerate() {
            // If we find our own position
            if entry.key == hash {
                // Preserve the existing move if no new move is provided:
                let move_to_store = if best_move.is_some() {
                    best_move.as_ref().map_or(TTMove::none(), TTMove::from_move)
                } else {
                    // Keep the old move
                    entry.tt_move
                };

                // Replacement condition for existing entries:
                // PV bonus = +2 for exact bounds, threshold = old_depth - 4
                let pv_bonus = if flag == TTFlag::Exact { 2 } else { 0 };
                let new_adjusted_depth = depth as i32 + pv_bonus;
                let old_threshold = entry.depth as i32 - 4;

                // Replace if: exact bound, or new depth high enough, or entry is aged
                let relative_age = (generation.wrapping_sub(entry.generation())) & 0x3F;
                if flag == TTFlag::Exact || new_adjusted_depth > old_threshold || relative_age != 0
                {
                    if entry.is_empty() {
                        self.used += 1;
                    }
                    *entry = TTEntry {
                        key: hash,
                        depth: depth as u8,
                        gen_bound: TTEntry::pack_gen_bound(generation, flag),
                        score: adjusted_score,
                        _padding: [0; 10],
                        tt_move: move_to_store,
                    };
                } else if entry.depth >= 5 && entry.flag() != TTFlag::Exact {
                    // Apply soft aging to deep non-exact entries:
                    // Makes them easier to replace in future iterations
                    entry.depth = entry.depth.saturating_sub(1);
                }
                return;
            }

            // Calculate replacement priority score (lower = more replaceable)
            let entry_score = Self::calculate_replacement_score(entry, generation);

            if entry_score < worst_score {
                worst_score = entry_score;
                replace_idx = i;
            }
        }

        // Prepare the new entry
        let new_entry = TTEntry {
            key: hash,
            depth: depth as u8,
            gen_bound: TTEntry::pack_gen_bound(generation, flag),
            score: adjusted_score,
            _padding: [0; 10],
            tt_move: best_move.as_ref().map_or(TTMove::none(), TTMove::from_move),
        };

        // Calculate value of the new entry
        let new_score = Self::calculate_replacement_score(&new_entry, generation);

        // Replace if new entry is more valuable OR victim is empty/aged
        if new_score >= worst_score {
            if bucket.entries[replace_idx].is_empty() {
                self.used += 1;
            }
            bucket.entries[replace_idx] = new_entry;
        }
    }

    /// Calculate a score for replacement priority (higher = more valuable, less replaceable).
    ///
    /// Factors considered:
    /// - Empty entries are always replaceable (score i32::MIN)
    /// - Deeper entries are more valuable (Base value = depth)
    /// - Older entries are less valuable (Penalty = 2 * age_diff)
    /// - Exact/PV nodes get a small bonus
    #[inline]
    fn calculate_replacement_score(entry: &TTEntry, current_generation: u8) -> i32 {
        if entry.is_empty() {
            return i32::MIN;
        }

        let mut score = entry.depth as i32;

        // Age penalty: penalize 2 points per generation old
        // Use 6-bit generation difference (wrapping)
        let age_diff = (current_generation.wrapping_sub(entry.generation())) & 0x3F;
        score -= (age_diff as i32) * 2;

        // Bound type bonus: slightly favor exact/PV nodes
        if entry.flag() == TTFlag::Exact {
            score += 2;
        }

        score
    }

    /// Increment the generation counter (call at the start of each search from root)
    pub fn increment_age(&mut self) {
        // Wrap at 63 (6 bits)
        self.generation = (self.generation + 1) & 0x3F;
        if self.generation == 0 {
            self.generation = 1; // Keep 0 reserved for empty
        }
    }

    /// Clear the entire table
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            *bucket = TTBucket::empty();
        }
        self.generation = 1;
        self.used = 0;
    }
}

// ============================================================================
// Re-export old types for API compatibility
// ============================================================================

// TTFlag is already defined above with the same variants (compatible)

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        // Verify our compact structs are the expected sizes
        assert_eq!(
            std::mem::size_of::<TTMove>(),
            40,
            "TTMove should be 40 bytes"
        );
        assert_eq!(
            std::mem::size_of::<TTEntry>(),
            64,
            "TTEntry should be 64 bytes"
        );
        assert_eq!(
            std::mem::size_of::<TTBucket>(),
            64 * ENTRIES_PER_BUCKET,
            "TTBucket should be 192 bytes (3 x 64)"
        );
    }

    #[test]
    fn test_tt_basic_operations() {
        let mut tt = TranspositionTable::new(1); // 1 MB table

        // Store and probe
        let hash = 0x123456789ABCDEF0u64;
        tt.store(&TTStoreParams {
            hash,
            depth: 5,
            flag: TTFlag::Exact,
            score: 100,
            best_move: None,
            ply: 0,
        });

        let result = tt.probe(&TTProbeParams {
            hash,
            alpha: -1000,
            beta: 1000,
            depth: 5,
            ply: 0,
            rule50_count: 0,
            rule_limit: 100,
        });
        assert!(result.is_some());
        let (score, _) = result.unwrap();
        assert_eq!(score, 100);
    }

    #[test]
    fn test_tt_gen_bound_packing() {
        // Test that generation and bound are packed correctly
        for r#gen in [0u8, 1, 31, 63] {
            for flag in [
                TTFlag::None,
                TTFlag::Exact,
                TTFlag::LowerBound,
                TTFlag::UpperBound,
            ] {
                let packed = TTEntry::pack_gen_bound(r#gen, flag);
                let entry = TTEntry {
                    key: 0,
                    score: 0,
                    depth: 0,
                    gen_bound: packed,
                    _padding: [0; 10],
                    tt_move: TTMove::none(),
                };
                assert_eq!(entry.generation(), r#gen & 0x3F);
                assert_eq!(entry.flag(), flag);
            }
        }
    }

    #[test]
    fn test_ttmove_sentinel() {
        let none = TTMove::none();
        assert!(!none.is_some());
        assert!(none.to_move().is_none());

        // A real move should be_some
        let real = TTMove {
            from_x: 0,
            from_y: 0,
            to_x: 1,
            to_y: 1,
            piece_type: 0,
            piece_color: 0,
            promotion: 0,
        };
        assert!(real.is_some());
    }
}
