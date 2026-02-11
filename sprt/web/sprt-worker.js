// Map internal engine piece letters to infinitechess.org two-letter codes
function engineLetterToSiteCode(letter) {
    const map = {
        'k': 'K', 'q': 'Q', 'r': 'R', 'b': 'B', 'n': 'N', 'p': 'P',
        'm': 'AM', 'c': 'CH', 'a': 'AR', 'h': 'HA', 'g': 'GU',
        'l': 'CA', 'i': 'GI', 'z': 'ZE', 'e': 'CE', 'y': 'RQ',
        'd': 'RC', 's': 'NR', 'u': 'HU', 'o': 'RO', 'x': 'OB', 'v': 'VO'
    };
    return map[letter] || letter.toUpperCase();
}

import initOld, * as wasmOld from './pkg-old/hydrochess_wasm.js';
const EngineOld = wasmOld.Engine;
import initNew, * as wasmNew from './pkg-new/hydrochess_wasm.js';
const EngineNew = wasmNew.Engine;
const initThreadPool = wasmNew.initThreadPool;
import { getVariantData, getAllVariants, generateSetupICN } from './variants.js';

let wasmReady = false;
let threadPoolInitialized = false;

// async function tryInitThreadPool(count) {
//     if (threadPoolInitialized) return;
//     if (typeof initThreadPool !== 'function') return;

//     try {
//         console.log(`[sprt-worker] Initializing thread pool with ${count} threads...`);
//         await initThreadPool(count);
//         threadPoolInitialized = true;
//         console.log(`[sprt-worker] Thread pool initialized.`);
//     } catch (e) {
//         console.warn(`[sprt-worker] Failed to initialize thread pool:`, e);
//     }
// }

function getVariantPosition(variantName, clock = null) {
    const variantData = getVariantData(variantName);
    const pieces = [];
    const special_rights = []; // Build dynamically from '+' suffix

    // Parse ICN position string into pieces array
    for (const pieceStr of variantData.position.split('|')) {
        if (!pieceStr) continue;

        const parts = pieceStr.split(',');
        if (parts.length !== 2) continue;

        const pieceInfo = parts[0];
        const yStr = parts[1];

        if (!pieceInfo) continue;

        // Split pieceInfo into a variable-length piece code and numeric x coordinate.
        // Examples:
        //   "P1"    -> pieceCode="P",  xPart="1"
        //   "AM3"   -> pieceCode="AM", xPart="3"
        //   "AR4+"  -> pieceCode="AR", xPart="4+"
        let splitIndex = 0;
        while (splitIndex < pieceInfo.length) {
            const ch = pieceInfo[splitIndex];
            if ((ch >= '0' && ch <= '9') || ch === '-') {
                break;
            }
            splitIndex++;
        }

        const pieceCode = pieceInfo.slice(0, splitIndex);
        const xRaw = pieceInfo.slice(splitIndex);

        if (!pieceCode || !xRaw) continue;

        // Side to move comes from piece code casing (first char of code)
        const isWhite = pieceCode[0] === pieceCode[0].toUpperCase();
        let player = isWhite ? 'w' : 'b';

        // Handle special rights (+ suffix) - check both x and y for safety
        const hasSpecialRights = xRaw.endsWith('+') || yStr.endsWith('+');
        const x = xRaw.endsWith('+') ? xRaw.slice(0, -1) : xRaw;
        const y = yStr.endsWith('+') ? yStr.slice(0, -1) : yStr;

        // Validate coordinates are valid numbers (allow negative and multi-digit)
        if (isNaN(parseInt(x, 10)) || isNaN(parseInt(y, 10))) {
            console.warn(`Invalid coordinates in ICN: ${pieceStr} -> x:${x}, y:${y}`);
            continue;
        }

        // Add to special_rights array if this piece has special rights
        if (hasSpecialRights) {
            special_rights.push(`${x},${y}`);
        }

        // Map piece types (including multi-letter raw ICN codes) to engine codes
        // Engine single-letter codes from README.md:
        //  p Pawn, n Knight, b Bishop, r Rook, q Queen, k King
        //  m Amazon, c Chancellor, a Archbishop, e Centaur, d Royal Centaur,
        //  h Hawk, g Guard, s Knightrider, l Camel, i Giraffe, z Zebra, y Royal Queen
        const codeLower = pieceCode.toLowerCase();
        let piece_type;
        switch (codeLower) {
            // Standard pieces
            case 'k': piece_type = 'k'; break;
            case 'q': piece_type = 'q'; break;
            case 'r': piece_type = 'r'; break;
            case 'b': piece_type = 'b'; break;
            case 'n': piece_type = 'n'; break;
            case 'p': piece_type = 'p'; break;

            // Amazon (raw ICN 'am') -> engine 'm'
            case 'am': piece_type = 'm'; break;

            // Chancellor (raw ICN 'ch') -> engine 'c'
            case 'ch': piece_type = 'c'; break;

            // Archbishop (raw ICN 'ar') -> engine 'a'
            case 'ar': piece_type = 'a'; break;

            // Hawk (raw ICN 'ha') -> engine 'h'
            case 'ha': piece_type = 'h'; break;

            // Guard (raw ICN 'gu') -> engine 'g'
            case 'gu': piece_type = 'g'; break;

            // Camel (raw ICN 'ca') -> engine 'l'
            case 'ca': piece_type = 'l'; break;

            // Giraffe (raw ICN 'gi') -> engine 'i'
            case 'gi': piece_type = 'i'; break;

            // Zebra (raw ICN 'ze') -> engine 'z'
            case 'ze': piece_type = 'z'; break;

            // Centaur (raw ICN 'ce') -> engine 'e'
            case 'ce': piece_type = 'e'; break;

            // Royal Queen (raw ICN 'rq') -> engine 'y'
            case 'rq': piece_type = 'y'; break;

            // Royal Centaur (raw ICN 'rc') -> engine 'd'
            case 'rc': piece_type = 'd'; break;

            // Knightrider (raw ICN 'nr') -> engine 's'
            case 'nr': piece_type = 's'; break;

            // Huygen (raw ICN 'hu') -> engine 'u'
            case 'hu': piece_type = 'u'; break;

            // Rose (raw ICN 'ro') -> engine 'o'; break;
            case 'ro': piece_type = 'o'; break;

            // Neutrals / other engine-local codes
            case 'ob':
                piece_type = 'x';
                player = 'n';
                break; // Obstacle (neutral blocker)
            case 'vo':
                piece_type = 'v';
                player = 'n';
                break; // Void (neutral)

            default:
                continue; // Skip unknown pieces
        }

        pieces.push({ x, y, piece_type, player });
    }

    // Variant-specific game rules (promotion ranks and allowed promotions) are
    // encoded alongside positions in sprt/web/variants.js so that this worker
    // stays in sync with the main site configuration. Pass them through
    // unchanged to the WASM side.
    const game_rules = variantData.game_rules || null;

    // World bounds: approximate the site's playableRegion by taking the
    // min/max piece coordinates and expanding them by worldBorder (if
    // present). This mirrors how variant.ts uses worldBorder as padding
    // between the furthest piece and the world border.
    let world_bounds = null;
    if (typeof variantData.worldBorder === 'number' && pieces.length > 0) {
        const pad = variantData.worldBorder;
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const p of pieces) {
            const xi = parseInt(p.x, 10);
            const yi = parseInt(p.y, 10);
            if (Number.isNaN(xi) || Number.isNaN(yi)) continue;
            if (xi < minX) minX = xi;
            if (xi > maxX) maxX = xi;
            if (yi < minY) minY = yi;
            if (yi > maxY) maxY = yi;
        }
        if (minX !== Infinity) {
            world_bounds = {
                left: String(minX - pad),
                right: String(maxX + pad),
                bottom: String(minY - pad),
                top: String(maxY + pad),
            };
        }
    }

    return {
        board: { pieces },
        turn: 'w',
        castling_rights: [], // Legacy for old EngineOld builds
        special_rights, // Use dynamically built array from '+' suffix
        en_passant: null,
        halfmove_clock: 0,
        fullmove_number: 1,
        move_history: [],
        game_rules,
        world_bounds,
        clock,
        variant: variantName, // Add variant name for custom evaluation
    };
}

function getStandardPosition(clock = null) {
    // Fallback to Classical variant for backward compatibility
    return getVariantPosition('Classical', clock);
}

function applyMove(position, move) {
    const pieces = position.board.pieces;
    const [fromX, fromY] = move.from.split(',');
    const [toX, toY] = move.to.split(',');

    const capturedIdx = pieces.findIndex(p => p.x === toX && p.y === toY);
    if (capturedIdx !== -1) {
        // Remove special rights for the captured piece if it had any
        if (position.special_rights) {
            const toKey = toX + ',' + toY;
            const sIdx = position.special_rights.indexOf(toKey);
            if (sIdx !== -1) {
                position.special_rights.splice(sIdx, 1);
            }
        }
        pieces.splice(capturedIdx, 1);
    }

    const movingPiece = pieces.find(p => p.x === fromX && p.y === fromY);
    if (!movingPiece) {
        throw new Error('No piece at ' + move.from);
    }

    // Enforce side-to-move: do not allow engines to move the opponent's pieces.
    if (position.turn === 'w' && movingPiece.player !== 'w') {
        throw new Error('Illegal move: white to move but piece at ' + move.from + ' is not white');
    }
    if (position.turn === 'b' && movingPiece.player !== 'b') {
        throw new Error('Illegal move: black to move but piece at ' + move.from + ' is not black');
    }

    const fromXi = parseInt(fromX, 10);
    const toXi = parseInt(toX, 10);
    const fromYi = parseInt(fromY, 10);
    const toYi = parseInt(toY, 10);
    const dx = toXi - fromXi;
    const dy = toYi - fromYi;
    const isKing = movingPiece.piece_type === 'k';
    const isPawn = movingPiece.piece_type === 'p';

    // === SPECIAL RIGHTS MANAGEMENT ===
    // Remove special rights for the piece that moved
    const fromKey = fromX + ',' + fromY;
    if (position.special_rights) {
        const idx = position.special_rights.indexOf(fromKey);
        if (idx !== -1) {
            position.special_rights.splice(idx, 1);
        }

        // If a king moves, remove special rights for all friendly castling partners on the same rank
        // Castling partners are any non-pawn, non-king piece on the same rank
        if (isKing) {
            const kingRank = fromY; // Use original position for rank check
            for (const p of pieces) {
                if (p.player !== movingPiece.player) continue;
                if (p.piece_type === 'p' || p.piece_type === 'k') continue;
                if (p.y !== kingRank) continue; // Must be on same rank as king was

                const partnerKey = p.x + ',' + p.y;
                const partnerIdx = position.special_rights.indexOf(partnerKey);
                if (partnerIdx !== -1) {
                    position.special_rights.splice(partnerIdx, 1);
                }
            }
        }

        // If a rook/guard moves, check if king should lose its right
        // (King loses castling right only when it moves, which is already handled above)
        // But we track partner movement by removing partner's special right
    }

    // === EN PASSANT HANDLING ===
    // Handle en passant capture: if pawn moves diagonally to en_passant square, capture the pawn
    if (isPawn && position.en_passant) {
        const epSquare = position.en_passant.square;
        const epPawnSquare = position.en_passant.pawn_square;
        if (toX + ',' + toY === epSquare) {
            // Remove the pawn at pawn_square
            const epIdx = pieces.findIndex(p => p.x + ',' + p.y === epPawnSquare);
            if (epIdx !== -1) {
                pieces.splice(epIdx, 1);
            }
        }
    }

    // Reset en passant - will be set below if this is a double pawn push
    position.en_passant = null;

    // Track en passant for double pawn push
    if (isPawn && Math.abs(dy) === 2) {
        const dir = movingPiece.player === 'w' ? 1 : -1;
        const epY = fromYi + dir; // The en passant capture square is between from and to
        position.en_passant = {
            square: fromX + ',' + epY,
            pawn_square: toX + ',' + toY
        };
    }

    // Handle castling in the worker's local board representation. The engine
    // implements castling by moving the king more than 1 square horizontally
    // and then relocating the rook on the same rank beyond the king's
    // destination. We mimic that here so our local board stays in sync.
    if (isKing && dy === 0 && Math.abs(dx) > 1) {
        const castleDir = dx > 0 ? 1 : -1;
        let partnerXi = toXi + castleDir; // search beyond king's destination
        // Look for any friendly piece (not just rooks) - engine supports castling with Guards etc.
        while (Math.abs(partnerXi - toXi) <= 16) {
            const partnerXStr = String(partnerXi);
            const pieceAt = pieces.find(p => p.x === partnerXStr && p.y === fromY);
            if (pieceAt) {
                // Accept any friendly piece as the castling partner (rook, guard, etc.)
                if (pieceAt.player === movingPiece.player) {
                    // Remove special rights for the castling partner
                    if (position.special_rights) {
                        const partnerKey = pieceAt.x + ',' + pieceAt.y;
                        const partnerIdx = position.special_rights.indexOf(partnerKey);
                        if (partnerIdx !== -1) {
                            position.special_rights.splice(partnerIdx, 1);
                        }
                    }
                    // Move the castling partner to the square the king jumped over
                    const partnerToXi = toXi - castleDir;
                    pieceAt.x = String(partnerToXi);
                    pieceAt.y = fromY;
                }
                break;
            }
            partnerXi += castleDir;
        }
    }

    movingPiece.x = toX;
    movingPiece.y = toY;

    if (move.promotion) {
        movingPiece.piece_type = move.promotion.toLowerCase();
    }

    position.turn = position.turn === 'w' ? 'b' : 'w';
    return position;
}

function clonePosition(position) {
    // Simple deep clone for our small position objects
    return JSON.parse(JSON.stringify(position));
}

function makePositionKey(position) {
    // Include all pieces with their positions
    const pieces = position.board.pieces.map((p) => p.player + p.piece_type + p.x + ',' + p.y);
    pieces.sort();

    // Compute EFFECTIVE castling rights for threefold repetition.
    // A side can castle if and only if:
    // 1. The king has special rights
    // 2. At least one castling partner (rook/guard) on the same side has special rights
    // If either condition fails, the other piece's rights become irrelevant.
    let castlingRights = '';

    if (position.special_rights && position.special_rights.length > 0) {
        const rights = new Set(position.special_rights);
        const boardPieces = position.board.pieces;

        // Find kings and their castling partners with rights
        for (const color of ['w', 'b']) {
            const king = boardPieces.find(p => p.player === color && p.piece_type === 'k');
            if (!king) continue;

            const kingKey = king.x + ',' + king.y;
            const kingHasRights = rights.has(kingKey);

            if (!kingHasRights) {
                // King has moved - this side cannot castle, rook rights are irrelevant
                continue;
            }

            if (kingHasRights) {
                // King has rights - check which castling partners (left/right) have rights
                // Castling partners are any friendly non-pawn, non-king piece on the same rank
                const kingY = parseInt(king.y, 10);
                const kingX = parseInt(king.x, 10);
                let leftPartner = false;
                let rightPartner = false;

                for (const p of boardPieces) {
                    if (p.player !== color) continue;
                    if (p.piece_type === 'p' || p.piece_type === 'k') continue;

                    const partnerY = parseInt(p.y, 10);
                    if (partnerY !== kingY) continue;

                    const partnerKey = p.x + ',' + p.y;
                    if (rights.has(partnerKey)) {
                        const partnerX = parseInt(p.x, 10);
                        if (partnerX < kingX) {
                            leftPartner = true;
                        } else {
                            rightPartner = true;
                        }
                    }
                }

                if (leftPartner) castlingRights += color + 'L';
                if (rightPartner) castlingRights += color + 'R';
            }
        }
    }

    // Compute individual PAWN special rights (double-push rights)
    let pawnRights = '';
    if (position.special_rights && position.special_rights.length > 0) {
        const rights = new Set(position.special_rights);
        const boardPieces = position.board.pieces;
        const pawnRightsCoords = [];

        for (const p of boardPieces) {
            if (p.piece_type === 'p') {
                const key = p.x + ',' + p.y;
                if (rights.has(key)) {
                    pawnRightsCoords.push(key);
                }
            }
        }

        if (pawnRightsCoords.length > 0) {
            pawnRightsCoords.sort();
            pawnRights = pawnRightsCoords.join(';');
        }
    }

    // Include en passant square if present
    // For threefold repetition, positions must have same en passant possibility
    let ep = '';
    if (position.en_passant && position.en_passant.square) {
        ep = position.en_passant.square;
    }

    return position.turn + '|' + pieces.join(';') + '|' + castlingRights + '|' + pawnRights + '|' + ep;
}

function nowMs() {
    if (typeof performance !== 'undefined' && performance && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
}

function formatClock(ms) {
    const totalSeconds = Math.floor(Math.max(0, ms) / 1000);
    const h = Math.floor(totalSeconds / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60);
    const s = Math.floor(totalSeconds % 60);
    const dec = Math.floor((Math.max(0, ms) % 1000) / 100);
    const hStr = h.toString();
    const mStr = m.toString().padStart(2, '0');
    const sStr = s.toString().padStart(2, '0');
    return `${hStr}:${mStr}:${sStr}.${dec}`;
}

async function ensureInit() {
    if (!wasmReady) {
        await initOld();
        await initNew();

        // Detect thread support and initialize pool if available
        // await tryInitThreadPool(2);

        wasmReady = true;
    }
}

async function playSingleGame(timePerMove, maxMoves, newPlaysWhite, materialThreshold, baseTimeMs, incrementMs, timeControl, variantName = 'Classical', maxDepth, searchNoise, seed, oldStrength) {
    if (typeof wasmNew.reset_engine_state === 'function') {
        wasmNew.reset_engine_state();
    }
    if (typeof wasmOld.reset_engine_state === 'function') {
        wasmOld.reset_engine_state();
    }
    const startPosition = getVariantPosition(variantName);
    let position = clonePosition(startPosition);
    const newColor = newPlaysWhite ? 'w' : 'b';
    const moveLines = [];
    const moveHistory = [];
    const texelSamples = [];

    const initialBase = typeof baseTimeMs === 'number' && baseTimeMs > 0 ? baseTimeMs : 0;
    const increment = typeof incrementMs === 'number' && incrementMs > 0 ? incrementMs : 0;
    let whiteClock = initialBase;
    let blackClock = initialBase;
    const haveClocks = initialBase > 0;
    const repetitionCounts = new Map();
    // Track 50-move rule and fullmove counter in JS, as the engine expects authoritative values
    let halfmoveClock = startPosition.halfmove_clock || 0;
    let fullmoveNumber = startPosition.fullmove_number || 1;

    // Track last known search evaluation (in cp from White's perspective)
    // for each engine, based on the eval returned alongside its normal
    // timed search for a move. If either engine does not expose eval, we
    // simply never adjudicate.
    let lastEvalNew = null;
    let lastEvalOld = null;

    function recordRepetition() {
        const key = makePositionKey(position);
        const prev = repetitionCounts.get(key) || 0;
        const next = prev + 1;
        repetitionCounts.set(key, next);
        return next;
    }

    // Helper for definitive terminal check (mate/stalemate) to prioritize over adjudication
    const getTerminalResult = (context = "") => {
        try {
            const icnString = generateSetupICN(
                variantName,
                startPosition.turn,
                halfmoveClock,
                fullmoveNumber,
                moveHistory
            );
            const checker = wasmNew.Engine.from_icn(icnString, {});
            const legal = typeof checker.get_legal_moves_js === 'function' ? checker.get_legal_moves_js() : [];
            const inCheck = typeof checker.is_in_check === 'function' && checker.is_in_check();
            checker.free();

            if (Array.isArray(legal) && legal.length === 0) {
                const winningColor = position.turn === 'w' ? 'b' : 'w';
                if (inCheck) {
                    const result = winningColor === newColor ? 'win' : 'loss';
                    const result_token = winningColor === 'w' ? '1-0' : '0-1';
                    for (const s of texelSamples) s.result_token = result_token;
                    moveLines.push('# Checkmate' + context);
                    return { result, log: moveLines.join('\n'), reason: 'checkmate', samples: texelSamples };
                } else {
                    for (const s of texelSamples) s.result_token = '1/2-1/2';
                    moveLines.push('# Stalemate' + context);
                    return { result: 'draw', log: moveLines.join('\n'), reason: 'stalemate', samples: texelSamples };
                }
            }
        } catch (e) { }
        return null;
    };

    // Initial position before any moves
    recordRepetition();

    try {
        for (let i = 0; i < maxMoves; i++) {
            const sideToMove = position.turn;
            const isWhiteTurn = sideToMove === 'w';

            // Sample positions for Texel-style tuning. We record a subset of
            // midgame positions (by ply index) together with the current
            // move_history and side to move. Final game result is attached
            // when the game finishes.
            const ply = moveHistory.length; // number of moves already played
            const pieceCount = position.board.pieces.length;
            if (ply >= 12 && ply <= 120 && ply % 4 === 0 && pieceCount > 4 && texelSamples.length < 32) {
                texelSamples.push({
                    move_history: moveHistory.slice(),
                    side_to_move: sideToMove,
                    ply_index: ply,
                    piece_count: pieceCount,
                    // Capture the full board state at this ply so that downstream
                    // tooling can reconstruct the exact position for inspection.
                    position: clonePosition(position),
                });
            }

            // Perform terminal check at start of ply (covers results of previous move)
            const earlyTerminal = getTerminalResult(' (terminal state detected at start of ply)');
            if (earlyTerminal) return earlyTerminal;

            // Let the appropriate engine choose a move
            const EngineClass = isWhiteTurn
                ? (newPlaysWhite ? EngineNew : EngineOld)
                : (newPlaysWhite ? EngineOld : EngineNew);
            const engineName = isWhiteTurn
                ? (newPlaysWhite ? 'new' : 'old')
                : (newPlaysWhite ? 'old' : 'new');

            let searchTimeMs = timePerMove;

            // Build ICN string from initial position and move history using the shared helper.
            // This ensures dynamic promotion ranks, world bounds, and move rule limits.
            const icnString = generateSetupICN(
                variantName,
                startPosition.turn,
                halfmoveClock,
                fullmoveNumber,
                moveHistory
            );

            const engineConfig = {
                strength_level: (engineName === 'old' && oldStrength && oldStrength < 3) ? oldStrength : 3,
                wtime: Math.floor(whiteClock),
                btime: Math.floor(blackClock),
                winc: Math.floor(increment),
                binc: Math.floor(increment)
            };

            let engine;
            if (typeof EngineClass.from_icn === 'function') {
                engine = EngineClass.from_icn(icnString, engineConfig);
            } else {
                // Backward compatibility for versions that don't support from_icn
                const engineInput = clonePosition(startPosition);
                engineInput.move_history = moveHistory.slice();
                engineInput.halfmove_clock = halfmoveClock;
                engineInput.fullmove_number = fullmoveNumber;
                if (oldStrength && oldStrength < 3) {
                    engineInput.strength_level = oldStrength;
                }
                if (haveClocks) {
                    engineInput.clock = {
                        wtime: Math.floor(whiteClock),
                        btime: Math.floor(blackClock),
                        winc: Math.floor(increment),
                        binc: Math.floor(increment),
                    };
                }
                engine = new EngineClass(engineInput);
            }

            const startMs = nowMs();

            // Safety check: if clock time is already zero or negative, flag timeout immediately
            if (haveClocks) {
                const currentClock = isWhiteTurn ? whiteClock : blackClock;
                if (currentClock <= 0) {
                    flaggedOnTime = true;
                    engine.free();
                    return {
                        result: isWhiteTurn ? 'black' : 'white',
                        reason: 'timeout',
                        moveHistory,
                        moveLines,
                        texelSamples,
                        adjudicated: false,
                        engineStats: { ...engineStats }
                    };
                }
            }

            // For the first 4 ply (2 moves each side), use a slight noise
            // to create opening variety. After ply 4, use normal search.
            const currentPly = moveHistory.length;
            const noiseAmp = currentPly < 4 ? (typeof searchNoise === 'number' ? searchNoise : 5) : null;

            let flaggedOnTime = false;
            let move;
            try {
                move = engine.get_best_move_with_time(haveClocks ? 0 : searchTimeMs, true, maxDepth, noiseAmp, seed ? BigInt(seed) : undefined);
            } catch (err) {
                // Fallback for older engines that don't support BigInt seed
                if (err instanceof TypeError || (err.message && err.message.includes('BigInt'))) {
                    move = engine.get_best_move_with_time(haveClocks ? 0 : searchTimeMs, true, maxDepth, noiseAmp, undefined);
                } else {
                    throw err;
                }
            }
            engine.free();

            const elapsed = Math.max(0, Math.round(nowMs() - startMs));

            if (haveClocks) {
                if (isWhiteTurn) {
                    let next = whiteClock - elapsed;
                    if (next < 0) {
                        flaggedOnTime = true;
                        next = 0;
                    }
                    whiteClock = next + increment;
                } else {
                    let next = blackClock - elapsed;
                    if (next < 0) {
                        flaggedOnTime = true;
                        next = 0;
                    }
                    blackClock = next + increment;
                }
            }

            if (haveClocks && flaggedOnTime) {
                moveLines.push('# Time forfeit: ' + (isWhiteTurn ? 'White' : 'Black') + ' flagged on time.');
                const result = engineName === 'new' ? 'loss' : 'win';
                const result_token = result === 'win' ? '1-0' : '0-1';
                for (const s of texelSamples) {
                    s.result_token = result_token;
                }
                return { result, log: moveLines.join('\n'), reason: 'time_forfeit', samples: texelSamples };
            }

            if (!move || !move.from || !move.to) {
                // Engine returned no move. Before treating this as a rules-based
                // terminal state, ask the WASM side whether any legal moves exist
                // from the same gameInput. If legal moves remain, classify this
                // as an engine failure instead of checkmate.

                let hasLegalMoves = false;
                try {
                    const icnString = generateSetupICN(
                        variantName,
                        startPosition.turn,
                        halfmoveClock,
                        fullmoveNumber,
                        moveHistory
                    );
                    const checker = wasmNew.Engine.from_icn(icnString, {});
                    if (typeof checker.get_legal_moves_js === 'function') {
                        const legal = checker.get_legal_moves_js();
                        if (Array.isArray(legal) && legal.length > 0) {
                            hasLegalMoves = true;
                        }
                    }
                    checker.free();
                } catch (e) {
                    // If the probe itself fails, fall back to conservative
                    // classification below.
                }

                let winningColor = sideToMove === 'w' ? 'b' : 'w';
                let reason = null;

                if (!hasLegalMoves) {
                    // True terminal: no legal moves for sideToMove.
                    const icnString = generateSetupICN(
                        variantName,
                        startPosition.turn,
                        halfmoveClock,
                        fullmoveNumber,
                        moveHistory
                    );
                    const checker = wasmNew.Engine.from_icn(icnString, {});
                    const inCheck = typeof checker.is_in_check === 'function' && checker.is_in_check();
                    checker.free();

                    // Special handling for Pawn_Horde: Black also wins by
                    // eliminating all White pieces (the pawn horde).
                    if (variantName === 'Pawn_Horde') {
                        const whitePieces = position.board.pieces.filter((p) =>
                            p.player === 'w' && p.piece_type !== 'x' && p.piece_type !== 'v'
                        );
                        if (whitePieces.length === 0) {
                            winningColor = 'b';
                            reason = 'horde_elimination';
                        }
                    }

                    if (!reason) {
                        if (inCheck) {
                            reason = 'checkmate';
                        } else {
                            reason = 'stalemate';
                            for (const s of texelSamples) {
                                s.result_token = '1/2-1/2';
                            }
                            moveLines.push('# No move returned; treated as stalemate for ' + (sideToMove === 'w' ? 'White' : 'Black') + '.');
                            return { result: 'draw', log: moveLines.join('\n'), reason, samples: texelSamples };
                        }
                    }
                } else {
                    // Engine produced no move even though legal moves exist.
                    // Treat this as an engine failure, not a rules-based result.
                    reason = 'engine_failure';
                }

                const result = winningColor === newColor ? 'win' : 'loss';
                const result_token = winningColor === 'w' ? '1-0' : '0-1';
                for (const s of texelSamples) {
                    s.result_token = result_token;
                }

                if (reason === 'horde_elimination') {
                    moveLines.push('# No move returned; treated as win by capturing all White pieces in Pawn Horde.');
                } else if (reason === 'checkmate') {
                    moveLines.push('# No move returned; treated as checkmate / no legal moves for ' + (sideToMove === 'w' ? 'White' : 'Black') + '.');
                } else {
                    moveLines.push('# No move returned; treated as engine failure (legal moves still exist).');
                }

                return { result, log: moveLines.join('\n'), reason, samples: texelSamples };
            }

            // Record this engine's last search evaluation (from White's POV) if
            // the engine returned an eval field. The Rust side reports eval from
            // the side-to-move's perspective.
            if (typeof move.eval === 'number') {
                const evalSide = move.eval;
                const evalWhite = sideToMove === 'w' ? evalSide : -evalSide;
                if (engineName === 'new') {
                    lastEvalNew = evalWhite;
                } else {
                    lastEvalOld = evalWhite;
                }
            }

            let isPawnMove = false;
            let isCapture = false;
            {
                const [fromX, fromY] = move.from.split(',');
                const [toX, toY] = move.to.split(',');
                const piecesBefore = position.board.pieces;
                const movingPiece = piecesBefore.find(p => p.x === fromX && p.y === fromY);
                if (movingPiece && typeof movingPiece.piece_type === 'string') {
                    isPawnMove = movingPiece.piece_type.toLowerCase() === 'p';
                }
                isCapture = piecesBefore.some(p => p.x === toX && p.y === toY);
            }

            // First try to apply the move to our local position. If this fails,
            // we treat it as an illegal move from the engine: the side to move
            // loses immediately, and we DO NOT record this move in the log or
            // move_history so that the resulting ICN is always playable.
            try {
                position = applyMove(position, move);
            } catch (e) {
                // Illegal move from the engine: side that moved loses. Do NOT
                // record the move itself in history so ICN remains playable.
                moveLines.push('# Illegal move from ' + (engineName === 'new' ? 'HydroChess New' : 'HydroChess Old') +
                    ': ' + (move && move.from && move.to ? (move.from + '>' + move.to) : 'null') +
                    ' (' + (e && e.message ? e.message : String(e)) + ')');
                const result = engineName === 'new' ? 'loss' : 'win';
                const result_token = result === 'win' ? '1-0' : '0-1';
                for (const s of texelSamples) {
                    s.result_token = result_token;
                }
                return { result, log: moveLines.join('\n'), reason: 'illegal_move', samples: texelSamples };
            }

            // Only after a successful apply do we log and record the move.
            let promotionSuffix = '';
            if (move.promotion) {
                // Convert engine promotion letter to site code, then apply case for side
                const siteCode = engineLetterToSiteCode(move.promotion);
                promotionSuffix = '=' + (sideToMove === 'w' ? siteCode.toUpperCase() : siteCode.toLowerCase());
            }

            // Build move comment: [%clk] only if game clocks, [%eval], and depth text comment
            let commands = '';
            if (haveClocks) {
                const clkMs = isWhiteTurn ? whiteClock : blackClock;
                commands += `[%clk ${formatClock(clkMs)}]`;
            }
            // Add eval if available
            if (typeof move.eval === 'number') {
                if (commands) commands += ' ';
                // If sideToMove is Black, negate the score so it's always from White's perspective.
                let evalVal = move.eval;
                if (sideToMove === 'b') {
                    evalVal = -evalVal;
                }

                // Check for mate score
                if (Math.abs(evalVal) >= 800000) {
                    if (evalVal > 0) {
                        const mateIn = Math.floor((900000 - evalVal + 1) / 2);
                        commands += `[%mate ${mateIn}]`;
                    } else {
                        const mateIn = Math.floor((900000 + evalVal + 1) / 2);
                        commands += `[%mate -${mateIn}]`;
                    }
                } else {
                    const evalCp = (evalVal / 100).toFixed(2);
                    const evalStr = evalVal >= 0 ? `+${evalCp}` : evalCp;
                    commands += `[%eval ${evalStr}]`;
                }
            }
            // Add depth as a text comment (not a command)
            let textComment = '';
            if (maxDepth) {
                textComment = `depth ${maxDepth}`;
            }
            // Combine: commands first, then text comment
            let comment = commands;
            if (textComment) {
                if (comment) comment += ' ';
                comment += textComment;
            }

            // Increment fullmove number after Black's move (position.turn is already flipped)
            // Note: We check for 'w' because applyMove already toggled the turn
            if (position.turn === 'w') {
                fullmoveNumber++;
            }

            moveLines.push(
                (sideToMove === 'w' ? 'W' : 'B') + ': ' + move.from + '>' + move.to + promotionSuffix + (comment ? '{' + comment + '}' : '')
            );

            // Track move history from the initial position for subsequent engine calls
            moveHistory.push({
                from: move.from,
                to: move.to,
                promotion: move.promotion || null
            });

            if (isPawnMove || isCapture) {
                halfmoveClock = 0;
            } else {
                halfmoveClock += 1;
            }

            const repCount = recordRepetition();
            if (repCount >= 3) {
                const terminal = getTerminalResult(' on repetition move');
                if (terminal) return terminal;

                for (const s of texelSamples) {
                    s.result_token = '1/2-1/2';
                }
                return { result: 'draw', log: moveLines.join('\n'), reason: 'threefold', samples: texelSamples };
            }

            if (halfmoveClock >= 100) {
                const terminal = getTerminalResult(' on 50-move rule');
                if (terminal) return terminal;

                for (const s of texelSamples) {
                    s.result_token = '1/2-1/2';
                }
                return { result: 'draw', log: moveLines.join('\n'), reason: 'fifty_move', samples: texelSamples };
            }

            // Check for insufficient material using a fresh game input representing the state AFTER the move
            try {
                const icnString = generateSetupICN(
                    variantName,
                    startPosition.turn,
                    halfmoveClock,
                    fullmoveNumber,
                    moveHistory
                );
                const checker = wasmNew.Engine.from_icn(icnString, {});
                const hasSufficientMaterial = typeof checker.is_sufficient_material === 'function'
                    ? checker.is_sufficient_material()
                    : true; // Default to true if function doesn't exist (old engine builds)
                checker.free();

                if (!hasSufficientMaterial) {
                    const terminal = getTerminalResult(' with insufficient material');
                    if (terminal) return terminal;

                    for (const s of texelSamples) {
                        s.result_token = '1/2-1/2';
                    }
                    moveLines.push('# Draw by insufficient material');
                    return { result: 'draw', log: moveLines.join('\n'), reason: 'insufficient_material', samples: texelSamples };
                }
            } catch (e) {
                // If the check fails, continue the game
            }

            // winner, stop early and award the game. Only start checking after at
            // least 20 plies, and only if both engines have provided evals.
            // Moved to the end of the loop so it only triggers if rule-based terminal states didn't match.
            if (moveHistory.length >= 20 && lastEvalNew !== null && lastEvalOld !== null) {
                const threshold = typeof materialThreshold === 'number' ? materialThreshold : 0;

                if (threshold > 0) {
                    const winnerFromWhiteEval = (score) => {
                        if (score >= threshold) return 'w';
                        if (score <= -threshold) return 'b';
                        return null;
                    };

                    const newWinner = winnerFromWhiteEval(lastEvalNew);
                    const oldWinner = winnerFromWhiteEval(lastEvalOld);

                    let winningColor = null;
                    if (newWinner && oldWinner && newWinner === oldWinner) {
                        winningColor = newWinner;
                    }

                    if (winningColor) {
                        const terminal = getTerminalResult(' (terminal state detected during adjudication)');
                        if (terminal) return terminal;

                        const evalCp = winningColor === 'w'
                            ? Math.min(lastEvalNew, lastEvalOld)
                            : Math.max(lastEvalNew, lastEvalOld);
                        const result = winningColor === newColor ? 'win' : 'loss';
                        const winnerStr = winningColor === 'w' ? 'White' : 'Black';
                        moveLines.push('# Game adjudicated by material: ~' + (evalCp > 0 ? '+' : '') + evalCp + ' cp for ' + winnerStr + ' (threshold ' + threshold + ' cp, both engines agree; search eval from main search)');
                        moveLines.push('# Engines: new=' + (newColor === 'w' ? 'White' : 'Black') + ', old=' + (newColor === 'w' ? 'Black' : 'White'));
                        const result_token = winningColor === 'w' ? '1-0' : '0-1';
                        for (const s of texelSamples) {
                            s.result_token = result_token;
                        }
                        return { result, log: moveLines.join('\n'), reason: 'material_adjudication', materialThreshold: threshold, samples: texelSamples };
                    }
                }
            }
        }
    } catch (e) {
        // Attach history to error object so we can report it
        e.moveLines = moveLines;
        e.moveHistory = moveHistory;
        throw e;
    }

    // Before declaring max_moves draw, check if the last move delivered checkmate/stalemate
    const terminal = getTerminalResult(' on final move');
    if (terminal) return terminal;

    for (const s of texelSamples) {
        s.result_token = '1/2-1/2';
    }
    return { result: 'draw', log: moveLines.join('\n'), reason: 'max_moves', samples: texelSamples };
}

// Per-game timeout to prevent hangs
const GAME_TIMEOUT_MS = 90000; // 90 seconds max per game
function withTimeout(promise, ms, fallbackValue) {
    return Promise.race([
        promise,
        new Promise((resolve) => setTimeout(() => resolve(fallbackValue), ms))
    ]);
}

self.onmessage = async (e) => {
    const msg = e.data;
    if (msg.type === 'runGame') {
        try {
            await ensureInit();

            const gamePromise = playSingleGame(
                msg.timePerMove,
                msg.maxMoves,
                msg.newPlaysWhite,
                msg.materialThreshold,
                msg.baseTimeMs,
                msg.incrementMs,
                msg.timeControl,
                msg.variantName || 'Classical',
                msg.maxDepth,
                msg.searchNoise,
                msg.seed,
                msg.oldStrength
            );

            // Timeout wrapper - treat timeout as draw
            const gameResult = await withTimeout(gamePromise, GAME_TIMEOUT_MS, {
                result: 'draw',
                log: '# Game timed out after ' + GAME_TIMEOUT_MS + 'ms',
                reason: 'timeout',
                samples: []
            });

            const { result, log, reason, materialThreshold, samples } = gameResult;

            self.postMessage({
                type: 'result',
                gameIndex: msg.gameIndex,
                result,
                log,
                newPlaysWhite: msg.newPlaysWhite,
                reason: reason || null,
                materialThreshold: materialThreshold ?? msg.materialThreshold ?? null,
                timeControl: msg.timeControl || null,
                samples: samples || [],
                variantName: msg.variantName || 'Classical',
            });
        } catch (err) {
            self.postMessage({
                type: 'error',
                gameIndex: msg.gameIndex,
                error: err.message || String(err),
                log: err.moveLines ? err.moveLines.join('\n') : null,
                moveHistory: err.moveHistory || null,
                variantName: msg.variantName || 'Classical',
            });
        }
    } else if (msg.type === 'getVariants') {
        // Return list of available variants for UI dropdown
        self.postMessage({
            type: 'variants',
            variants: getAllVariants(),
        });
    } else if (msg.type === 'probe') {
        // Lightweight probe used by the UI to determine how many workers
        // can be created before WASM instantiation runs out of memory.
        try {
            await ensureInit();
            self.postMessage({
                type: 'probeResult',
                ok: true,
            });
        } catch (err) {
            self.postMessage({
                type: 'probeResult',
                ok: false,
                error: err && err.message ? err.message : String(err),
            });
        }
    }
};
