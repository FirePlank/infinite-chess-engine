import initOld, { Engine as EngineOld } from './pkg-old/hydrochess_wasm.js';
import initNew, { Engine as EngineNew } from './pkg-new/hydrochess_wasm.js';

let wasmReady = false;

function getStandardPosition() {
    const pieces = [];
    pieces.push({ x: '1', y: '1', piece_type: 'r', player: 'w' });
    pieces.push({ x: '2', y: '1', piece_type: 'n', player: 'w' });
    pieces.push({ x: '3', y: '1', piece_type: 'b', player: 'w' });
    pieces.push({ x: '4', y: '1', piece_type: 'q', player: 'w' });
    pieces.push({ x: '5', y: '1', piece_type: 'k', player: 'w' });
    pieces.push({ x: '6', y: '1', piece_type: 'b', player: 'w' });
    pieces.push({ x: '7', y: '1', piece_type: 'n', player: 'w' });
    pieces.push({ x: '8', y: '1', piece_type: 'r', player: 'w' });
    for (let i = 1; i <= 8; i++) {
        pieces.push({ x: String(i), y: '2', piece_type: 'p', player: 'w' });
    }
    pieces.push({ x: '1', y: '8', piece_type: 'r', player: 'b' });
    pieces.push({ x: '2', y: '8', piece_type: 'n', player: 'b' });
    pieces.push({ x: '3', y: '8', piece_type: 'b', player: 'b' });
    pieces.push({ x: '4', y: '8', piece_type: 'q', player: 'b' });
    pieces.push({ x: '5', y: '8', piece_type: 'k', player: 'b' });
    pieces.push({ x: '6', y: '8', piece_type: 'b', player: 'b' });
    pieces.push({ x: '7', y: '8', piece_type: 'n', player: 'b' });
    pieces.push({ x: '8', y: '8', piece_type: 'r', player: 'b' });
    for (let i = 1; i <= 8; i++) {
        pieces.push({ x: String(i), y: '7', piece_type: 'p', player: 'b' });
    }

    // Standard infinite-chess special rights: all pawns (double-step)
    // plus kings and rooks (castling and related king/rook rights).
    const special_rights = [];
    for (let i = 1; i <= 8; i++) {
        special_rights.push(i + ',2'); // white pawns
        special_rights.push(i + ',7'); // black pawns
    }
    // White rooks and king
    special_rights.push('1,1');
    special_rights.push('8,1');
    special_rights.push('5,1');
    // Black rooks and king
    special_rights.push('1,8');
    special_rights.push('8,8');
    special_rights.push('5,8');

    return {
        board: { pieces },
        // Starting side for the game; the WASM engine will reconstruct the
        // current side-to-move by replaying move_history.
        turn: 'w',
        // Support both old and new APIs: legacy castling_rights for old EngineOld
        // builds, and special_rights for the new engine.
        castling_rights: [],
        special_rights,
        en_passant: null,
        halfmove_clock: 0,
        fullmove_number: 1,
        move_history: [],
        game_rules: null,
        world_bounds: null,
    };
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

function isGameOver(position) {
    const kings = position.board.pieces.filter(p => p.piece_type === 'k');
    if (kings.length < 2) {
        return { over: true, reason: 'checkmate' };
    }
    if (position.board.pieces.length <= 2) {
        return { over: true, reason: 'draw' };
    }
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

async function playSingleGame(timePerMove, maxMoves, newPlaysWhite, openingMove, materialThreshold, baseTimeMs, incrementMs, timeControl) {
    const startPosition = getStandardPosition();
    let position = clonePosition(startPosition);
    const newColor = newPlaysWhite ? 'w' : 'b';
    const moveLines = [];
    const moveHistory = [];

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

        if (haveClocks) {
            const currentClock = isWhiteTurn ? whiteClock : blackClock;
            if (currentClock <= 0) {
                const loserEngine = isWhiteTurn
                    ? (newPlaysWhite ? 'new' : 'old')
                    : (newPlaysWhite ? 'old' : 'new');
                moveLines.push('# Time forfeit before move: ' + (isWhiteTurn ? 'White' : 'Black') + ' has no time remaining.');
                const result = loserEngine === 'new' ? 'loss' : 'win';
                return { result, log: moveLines.join('\n'), reason: 'time_forfeit' };
            }
        }

        // Smart adjudication: if both engines' most recent SEARCH evaluations
        // (taken from their normal timed move searches) clearly agree on a
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
                    return { result, log: moveLines.join('\n'), reason: 'material_adjudication', materialThreshold: threshold };
                }
            }
        }

        // Otherwise, let the appropriate engine choose a move from the full
        // game history starting from the standard position. We rebuild
        // gameInput each ply so the WASM side can reconstruct all dynamic
        // state (clocks, en passant, special rights) by replaying moves.
        const gameInput = clonePosition(startPosition);
        gameInput.move_history = moveHistory.slice();

        // Let the appropriate engine choose a move on this gameInput
        const EngineClass = isWhiteTurn
            ? (newPlaysWhite ? EngineNew : EngineOld)
            : (newPlaysWhite ? EngineOld : EngineNew);
        const engineName = isWhiteTurn
            ? (newPlaysWhite ? 'new' : 'old')
            : (newPlaysWhite ? 'old' : 'new');

        let searchTimeMs = timePerMove;
        if (haveClocks) {
            const currentClock = isWhiteTurn ? whiteClock : blackClock;
            const baseSec = Math.max(0, currentClock / 1000);
            const incSec = increment / 1000;
            // Dynamic per-move limit: (currentBase/20 + inc/2) seconds
            searchTimeMs = Math.max(10, Math.round(((baseSec / 20) + (incSec / 2)) * 1000));
        }

        const engine = new EngineClass(gameInput);
        const startMs = haveClocks ? nowMs() : 0;
        const move = engine.get_best_move_with_time(searchTimeMs);
        engine.free();
        let flaggedOnTime = false;
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
            return { result, log: moveLines.join('\n'), reason: 'time_forfeit' };
        }

        if (!move || !move.from || !move.to) {
            // Engine failed to produce a move: treat as that engine losing.
            moveLines.push('# Engine ' + (engineName === 'new' ? 'HydroChess New' : 'HydroChess Old') +
                ' failed to return a move.');
            const result = engineName === 'new' ? 'loss' : 'win';
            return { result, log: moveLines.join('\n') };
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
            return { result, log: moveLines.join('\n'), reason: 'illegal_move' };
        }

        // Only after a successful apply do we log and record the move.
        moveLines.push(
            (sideToMove === 'w' ? 'W' : 'B') + ': ' + move.from + '>' + move.to +
            (move.promotion ? '=' + move.promotion : '')
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
            return { result: 'draw', log: moveLines.join('\n'), reason: 'threefold' };
        }

        if (halfmoveClock >= 100) {
            return { result: 'draw', log: moveLines.join('\n'), reason: 'fifty_move' };
        }

        const gameState = isGameOver(position);
        if (gameState.over) {
            if (gameState.reason === 'draw') {
                return { result: 'draw', log: moveLines.join('\n'), reason: 'insufficient_material' };
            }
            const result = sideToMove === newColor ? 'win' : 'loss';
            return { result, log: moveLines.join('\n'), reason: gameState.reason || 'checkmate' };
        }
    }

    return { result: 'draw', log: moveLines.join('\n') };
}

self.onmessage = async (e) => {
    const msg = e.data;
    if (msg.type === 'runGame') {
        try {
            await ensureInit();
            const { result, log, reason, materialThreshold } = await playSingleGame(
                msg.timePerMove,
                msg.maxMoves,
                msg.newPlaysWhite,
                msg.openingMove,
                msg.materialThreshold,
                msg.baseTimeMs,
                msg.incrementMs,
                msg.timeControl,
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
            });
        } catch (err) {
            self.postMessage({
                type: 'error',
                gameIndex: msg.gameIndex,
                error: err.message || String(err),
            });
        }
    }
};
