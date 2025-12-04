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

import initOld, { Engine as EngineOld } from './pkg-old/hydrochess_wasm.js';
import initNew, { Engine as EngineNew } from './pkg-new/hydrochess_wasm.js';
import { getVariantData, getAllVariants } from './variants.js';

let wasmReady = false;

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

    // Handle castling in the worker's local board representation. The engine
    // implements castling by moving the king more than 1 square horizontally
    // and then relocating the rook on the same rank beyond the king's
    // destination. We mimic that here so our local board stays in sync.
    const isKing = movingPiece.piece_type === 'k';
    const fromXi = parseInt(fromX, 10);
    const toXi = parseInt(toX, 10);
    const fromYi = parseInt(fromY, 10);
    const toYi = parseInt(toY, 10);
    const dx = toXi - fromXi;
    const dy = toYi - fromYi;

    if (isKing && dy === 0 && Math.abs(dx) > 1) {
        const rookDir = dx > 0 ? 1 : -1;
        let rookXi = toXi + rookDir; // search beyond king's destination
        // We stop if we run into any non-rook piece or wander too far.
        while (Math.abs(rookXi - toXi) <= 16) {
            const rookXStr = String(rookXi);
            const pieceAt = pieces.find(p => p.x === rookXStr && p.y === fromY);
            if (pieceAt) {
                if (pieceAt.player === movingPiece.player && pieceAt.piece_type === 'r') {
                    // Move rook to the square the king jumped over
                    const rookToXi = toXi - rookDir;
                    pieceAt.x = String(rookToXi);
                    pieceAt.y = fromY;
                }
                break;
            }
            rookXi += rookDir;
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

// Extremely conservative game-end detection for SPRT harness.
//
// The Rust engine already handles true terminal states (no legal moves) and
// our harness adds:
//   - material/eval adjudication
//   - repetition (threefold) and 50-move rule
//   - time forfeits
//   - illegal moves / engine failure
//
// Several variants (e.g. Pawn_Horde) are *designed* to have only one king
// on the board, so treating "kings < 2" as checkmate is incorrect and was
// causing games to end after a single move in SPRT logs. To avoid
// hardcoding variant-specific rules here (and to keep the harness fast), we
// no longer try to detect wins or draws based solely on king count or total
// piece count.
//
// For now, this helper never declares the game over; we rely entirely on
// the engine and the explicit adjudication logic above.
function isGameOver(_position) {
    return { over: false };
}

function clonePosition(position) {
    // Simple deep clone for our small position objects
    return JSON.parse(JSON.stringify(position));
}

function makePositionKey(position) {
    const parts = position.board.pieces.map((p) => p.player + p.piece_type + p.x + ',' + p.y);
    parts.sort();
    return position.turn + '|' + parts.join(';');
}

function nowMs() {
    if (typeof performance !== 'undefined' && performance && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
}

async function ensureInit() {
    if (!wasmReady) {
        await initOld();
        await initNew();
        wasmReady = true;
    }
}

async function playSingleGame(timePerMove, maxMoves, newPlaysWhite, openingMove, materialThreshold, baseTimeMs, incrementMs, timeControl, variantName = 'Classical') {
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
    let halfmoveClock = 0;

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

    // Initial position before any moves
    recordRepetition();

    // Apply opening move if provided (always white's first move)
    if (openingMove) {
        moveLines.push('W: ' + openingMove.from + '>' + openingMove.to);
        position = applyMove(position, openingMove);
        moveHistory.push({
            from: openingMove.from,
            to: openingMove.to,
            promotion: openingMove.promotion || null
        });
        halfmoveClock = 0;
        recordRepetition();
    }

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

        // winner, stop early and award the game. Only start checking after at
        // least 20 plies, and only if both engines have provided evals.
        if (moveHistory.length >= 20 && lastEvalNew !== null && lastEvalOld !== null) {
            const uiThresh = typeof materialThreshold === 'number' ? materialThreshold : 0;
            const threshold = Math.max(1500, uiThresh);
            if (threshold > 0) {
                function winnerFromWhiteEval(score) {
                    if (score >= threshold) return 'w';
                    if (score <= -threshold) return 'b';
                    return null;
                }

                const newWinner = winnerFromWhiteEval(lastEvalNew);
                const oldWinner = winnerFromWhiteEval(lastEvalOld);

                let winningColor = null;
                if (newWinner && oldWinner && newWinner === oldWinner) {
                    winningColor = newWinner;
                }

                if (winningColor) {
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

        // Otherwise, let the appropriate engine choose a move from the full
        // game history starting from the standard position. We rebuild
        // gameInput each ply so the WASM side can reconstruct all dynamic
        // state (clocks, en passant, special rights) by replaying moves.
        const gameInput = clonePosition(startPosition);
        gameInput.move_history = moveHistory.slice();
        
        // Include clock info in the game state so the engine can manage its own time
        if (haveClocks) {
            gameInput.clock = {
                wtime: Math.floor(whiteClock),
                btime: Math.floor(blackClock),
                winc: Math.floor(increment),
                binc: Math.floor(increment),
            };
        }

        // Let the appropriate engine choose a move on this gameInput
        const EngineClass = isWhiteTurn
            ? (newPlaysWhite ? EngineNew : EngineOld)
            : (newPlaysWhite ? EngineOld : EngineNew);
        const engineName = isWhiteTurn
            ? (newPlaysWhite ? 'new' : 'old')
            : (newPlaysWhite ? 'old' : 'new');

        let searchTimeMs = timePerMove;
        let flaggedOnTime = false;
        const engine = new EngineClass(gameInput);
        const startMs = haveClocks ? nowMs() : 0;
        
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
        
        const move = engine.get_best_move_with_time(haveClocks ? 0 : searchTimeMs);
        engine.free();
        if (haveClocks) {
            const elapsed = Math.max(0, Math.round(nowMs() - startMs));
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
                const checker = new EngineClass(gameInput);
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
                // True terminal: no legal moves for sideToMove. From SPRT's
                // perspective this is a loss for that side (checkmate or
                // stalemate-as-loss), with one special-case variant below.
                reason = 'checkmate';

                // Special handling for Pawn_Horde: Black also wins by
                // eliminating all White pieces (the pawn horde). If that
                // condition holds, record a distinct reason.
                if (variantName === 'Pawn_Horde') {
                    const whitePieces = position.board.pieces.filter((p) =>
                        p.player === 'w' && p.piece_type !== 'x' && p.piece_type !== 'v'
                    );
                    if (whitePieces.length === 0) {
                        winningColor = 'b';
                        reason = 'horde_elimination';
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

        moveLines.push(
            (sideToMove === 'w' ? 'W' : 'B') + ': ' + move.from + '>' + move.to + promotionSuffix
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
            for (const s of texelSamples) {
                s.result_token = '1/2-1/2';
            }
            return { result: 'draw', log: moveLines.join('\n'), reason: 'threefold', samples: texelSamples };
        }

        if (halfmoveClock >= 100) {
            for (const s of texelSamples) {
                s.result_token = '1/2-1/2';
            }
            return { result: 'draw', log: moveLines.join('\n'), reason: 'fifty_move', samples: texelSamples };
        }

        const gameState = isGameOver(position);
        if (gameState.over) {
            if (gameState.reason === 'draw') {
                for (const s of texelSamples) {
                    s.result_token = '1/2-1/2';
                }
                return { result: 'draw', log: moveLines.join('\n'), reason: 'insufficient_material', samples: texelSamples };
            }
            const result = sideToMove === newColor ? 'win' : 'loss';
            const result_token = result === 'win' ? '1-0' : '0-1';
            for (const s of texelSamples) {
                s.result_token = result_token;
            }
            return { result, log: moveLines.join('\n'), reason: gameState.reason || 'checkmate', samples: texelSamples };
        }
    }

    for (const s of texelSamples) {
        s.result_token = '1/2-1/2';
    }
    return { result: 'draw', log: moveLines.join('\n'), samples: texelSamples };
}

self.onmessage = async (e) => {
    const msg = e.data;
    if (msg.type === 'runGame') {
        try {
            await ensureInit();
            const { result, log, reason, materialThreshold, samples } = await playSingleGame(
                msg.timePerMove,
                msg.maxMoves,
                msg.newPlaysWhite,
                msg.openingMove,
                msg.materialThreshold,
                msg.baseTimeMs,
                msg.incrementMs,
                msg.timeControl,
                msg.variantName || 'Classical', // Default to Classical if not specified
            );
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
