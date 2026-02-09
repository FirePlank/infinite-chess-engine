/**
 * SPSA Game Worker
 * 
 * Runs individual games for SPSA tuning with configurable search parameters.
 * Each game pits θ+ parameters against θ- parameters.
 * 
 * Based on the proven sprt-worker.js infrastructure.
 */

import init, { Engine } from './pkg-spsa/hydrochess.js';
import { getVariantData, generateSetupICN } from './variants.js';

let wasmReady = false;

// ============================================================================
// Position Setup (adapted from sprt-worker.js)
// ============================================================================

function getVariantPosition(variantName, clock = null) {
    const variantData = getVariantData(variantName);
    const pieces = [];
    const special_rights = [];

    for (const pieceStr of variantData.position.split('|')) {
        if (!pieceStr) continue;
        const parts = pieceStr.split(',');
        if (parts.length !== 2) continue;

        const pieceInfo = parts[0];
        const yStr = parts[1];
        if (!pieceInfo) continue;

        let splitIndex = 0;
        while (splitIndex < pieceInfo.length) {
            const ch = pieceInfo[splitIndex];
            if ((ch >= '0' && ch <= '9') || ch === '-') break;
            splitIndex++;
        }

        const pieceCode = pieceInfo.slice(0, splitIndex);
        const xRaw = pieceInfo.slice(splitIndex);
        if (!pieceCode || !xRaw) continue;

        const isWhite = pieceCode[0] === pieceCode[0].toUpperCase();
        let player = isWhite ? 'w' : 'b';

        const hasSpecialRights = xRaw.endsWith('+') || yStr.endsWith('+');
        const x = xRaw.endsWith('+') ? xRaw.slice(0, -1) : xRaw;
        const y = yStr.endsWith('+') ? yStr.slice(0, -1) : yStr;

        if (isNaN(parseInt(x, 10)) || isNaN(parseInt(y, 10))) continue;

        if (hasSpecialRights) {
            special_rights.push(`${x},${y}`);
        }

        const codeLower = pieceCode.toLowerCase();
        const pieceMap = {
            'k': 'k', 'q': 'q', 'r': 'r', 'b': 'b', 'n': 'n', 'p': 'p',
            'am': 'm', 'ch': 'c', 'ar': 'a', 'ha': 'h', 'gu': 'g',
            'ca': 'l', 'gi': 'i', 'ze': 'z', 'ce': 'e', 'rq': 'y',
            'rc': 'd', 'nr': 's', 'hu': 'u', 'ro': 'o', 'ob': 'x', 'vo': 'v'
        };

        let piece_type = pieceMap[codeLower];
        if (!piece_type) continue;

        if (codeLower === 'ob' || codeLower === 'vo') {
            player = 'n';
        }

        pieces.push({ x, y, piece_type, player });
    }

    const game_rules = variantData.game_rules || null;
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
        special_rights,
        en_passant: null,
        halfmove_clock: 0,
        fullmove_number: 1,
        move_history: [],
        game_rules,
        world_bounds,
        clock,
        variant: variantName,
    };
}

function clonePosition(position) {
    return JSON.parse(JSON.stringify(position));
}

function makePositionKey(position) {
    const parts = position.board.pieces.map((p) => p.player + p.piece_type + p.x + ',' + p.y);
    parts.sort();
    return position.turn + '|' + parts.join(';');
}

async function ensureInit() {
    if (!wasmReady) {
        await init();
        wasmReady = true;
    }
}

// ============================================================================
// Game Playing Logic
// ============================================================================

/**
 * Play a single game between θ+ and θ- parameter sets.
 * 
 * @param {Object} thetaPlus - Parameters for the "plus" side
 * @param {Object} thetaMinus - Parameters for the "minus" side
 * @param {boolean} plusPlaysWhite - Whether θ+ plays as White
 * @param {number} timePerMove - Time control in milliseconds
 * @param {string} variantName - Chess variant to use
 * @param {number} maxMoves - Maximum moves before draw
 * @returns {string} 'plus', 'minus', or 'draw'
 */
async function playSingleGame(thetaPlus, thetaMinus, plusPlaysWhite, timePerMove, variantName = 'Classical', maxMoves = 200) {
    const startPosition = getVariantPosition(variantName);
    let position = clonePosition(startPosition);
    const moveHistory = [];
    const repetitionCounts = new Map();
    let halfmoveClock = 0;

    function recordRepetition() {
        const key = makePositionKey(position);
        const prev = repetitionCounts.get(key) || 0;
        const next = prev + 1;
        repetitionCounts.set(key, next);
        return next;
    }

    recordRepetition();

    for (let i = 0; i < maxMoves; i++) {
        const isWhiteTurn = position.turn === 'w';
        const isPlusTurn = (isWhiteTurn === plusPlaysWhite);
        const paramsToUse = isPlusTurn ? thetaPlus : thetaMinus;

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
            wtime: timePerMove, // Use timePerMove as base clock if no complex TC
            btime: timePerMove,
            winc: 0,
            binc: 0
        };

        const engine = Engine.from_icn(icnString, engineConfig);

        // Apply search params if engine supports it
        if (typeof engine.set_search_params === 'function') {
            engine.set_search_params(JSON.stringify(paramsToUse));
        }

        const move = engine.get_best_move_with_time(0, true);
        engine.free();

        if (!move || !move.from || !move.to) {
            // No legal moves - loss for side to move
            const winner = isWhiteTurn ? 'black' : 'white';
            return (plusPlaysWhite === (winner === 'white')) ? 'minus' : 'plus';
        }

        // Apply move to position
        const [fromX, fromY] = move.from.split(',');
        const [toX, toY] = move.to.split(',');

        // Track pawn moves and captures for 50-move rule
        const movingPiece = position.board.pieces.find(p => p.x === fromX && p.y === fromY);
        const isPawnMove = movingPiece && movingPiece.piece_type === 'p';
        const isCapture = position.board.pieces.some(p => p.x === toX && p.y === toY);

        // Remove captured piece
        const capIdx = position.board.pieces.findIndex(p => p.x === toX && p.y === toY);
        if (capIdx !== -1) {
            position.board.pieces.splice(capIdx, 1);
        }

        if (!movingPiece) {
            // Invalid state - engine failure, opponent wins
            return isPlusTurn ? 'minus' : 'plus';
        }

        // Handle castling
        const isKing = movingPiece.piece_type === 'k';
        const fromXi = parseInt(fromX, 10);
        const toXi = parseInt(toX, 10);
        const dx = toXi - fromXi;

        if (isKing && Math.abs(dx) > 1) {
            const rookDir = dx > 0 ? 1 : -1;
            let rookXi = toXi + rookDir;
            while (Math.abs(rookXi - toXi) <= 16) {
                const rookXStr = String(rookXi);
                const pieceAt = position.board.pieces.find(p => p.x === rookXStr && p.y === fromY);
                if (pieceAt) {
                    if (pieceAt.player === movingPiece.player && pieceAt.piece_type === 'r') {
                        const rookToXi = toXi - rookDir;
                        pieceAt.x = String(rookToXi);
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

        position.turn = isWhiteTurn ? 'b' : 'w';
        moveHistory.push({ from: move.from, to: move.to, promotion: move.promotion || null });

        // 50-move rule
        if (isPawnMove || isCapture) {
            halfmoveClock = 0;
        } else {
            halfmoveClock++;
        }

        if (halfmoveClock >= 100) {
            return 'draw';
        }

        // Threefold repetition
        const repCount = recordRepetition();
        if (repCount >= 3) {
            return 'draw';
        }
    }

    return 'draw';
}

// ============================================================================
// Worker Message Handler
// ============================================================================

// function withTimeout is below

function withTimeout(promise, ms, fallbackValue) {
    return Promise.race([
        promise,
        new Promise((resolve) => setTimeout(() => resolve(fallbackValue), ms))
    ]);
}

self.onmessage = async (e) => {
    const msg = e.data;

    if (msg.type === 'init') {
        try {
            await ensureInit();
            self.postMessage({ type: 'ready' });
        } catch (err) {
            self.postMessage({ type: 'error', error: err.message || String(err) });
        }
    } else if (msg.type === 'runGame') {
        try {
            await ensureInit();

            const gamePromise = playSingleGame(
                msg.thetaPlus,
                msg.thetaMinus,
                msg.plusPlaysWhite,
                msg.timePerMove,
                msg.variantName || 'Classical',
                msg.maxMoves || 150  // Reduced max moves for faster games
            );

            // Timeout wrapper - treat timeout as draw
            // Scale timeout with thinking time: 100ms -> 60s (ratio 600)
            const timeoutMs = msg.timePerMove * 600;
            const result = await withTimeout(gamePromise, timeoutMs, 'draw');

            self.postMessage({
                type: 'result',
                gameIndex: msg.gameIndex,
                result,
                plusPlaysWhite: msg.plusPlaysWhite
            });
        } catch (err) {
            self.postMessage({
                type: 'error',
                gameIndex: msg.gameIndex,
                error: err.message || String(err)
            });
        }
    }
};
