import initWasm, * as wasm from './pkg-new/hydrochess_wasm.js';
import { getVariantData, generateSetupICN } from './variants.js';

const Engine = wasm.Engine;
const resetEngineState = wasm.reset_engine_state;
let wasmReady = false;

async function initializeWasm() {
    if (wasmReady) return true;
    try {
        await initWasm();
        wasmReady = true;
        return true;
    } catch (e) {
        console.error('[review-worker] Failed to init WASM:', e);
        return false;
    }
}

/**
 * Analyze a single position at a given move index.
 * 
 * moveIndex=0 means the starting position (before any move).
 * moveIndex=N means the position after N moves have been played.
 * 
 * We build the ICN with moves 0..moveIndex-1 applied, then evaluate.
 */
function analyzePosition(variantName, startTurn, halfmoveClock, fullmoveNumber, allMoves, moveIndex, depth) {
    // Build move history up to moveIndex
    const movesUpTo = allMoves.slice(0, moveIndex);

    const icn = generateSetupICN(variantName, startTurn, halfmoveClock, fullmoveNumber, movesUpTo);

    // Reset TT before every position so stale entries never corrupt a fresh search
    try { resetEngineState(); } catch (e) { }

    let engine;
    try {
        engine = Engine.from_icn(icn, {});
    } catch (e) {
        return { error: 'Failed to create engine: ' + e.message };
    }

    let evalResult = null;
    let bestMove = null;
    let searchDepth = 0;
    let isCheckmate = false;
    let isStalemate = false;
    let legalMoveCount = 0;

    try {
        // Check for terminal state first
        const legalMoves = engine.get_legal_moves_js();
        const inCheck = engine.is_in_check();
        legalMoveCount = legalMoves ? legalMoves.length : 0;

        if (!legalMoves || legalMoves.length === 0) {
            if (inCheck) {
                isCheckmate = true;
                // Side to move is checkmated: very bad for them
                evalResult = -900000;
            } else {
                isStalemate = true;
                evalResult = 0;
            }
        } else if (legalMoves.length === 1) {
            // Only one legal move: skip search, just return the forced move
            const m = legalMoves[0];
            bestMove = { from: m.from, to: m.to, promotion: m.promotion || null };
            evalResult = 0; // Neutral eval for forced moves (position unchanged)
            searchDepth = 0;
        } else {
            // Run engine search
            const result = engine.get_best_move_with_time(0, true, depth, undefined, undefined);
            if (result && typeof result.eval === 'number') {
                evalResult = result.eval; // From side-to-move's perspective
                searchDepth = result.depth || depth;
                bestMove = { from: result.from, to: result.to, promotion: result.promotion || null };
            }
        }
    } catch (e) {
        console.warn('[review-worker] Analysis error at moveIndex', moveIndex, e);
    } finally {
        try { engine.free(); } catch (e) { }
    }

    return {
        moveIndex,
        eval: evalResult,
        depth: searchDepth,
        bestMove,
        isCheckmate,
        isStalemate,
        legalMoveCount,
    };
}

self.onmessage = async function (e) {
    const msg = e.data;

    if (msg.type === 'init') {
        const ok = await initializeWasm();
        self.postMessage({ type: 'initResult', ok });
        return;
    }

    if (msg.type === 'analyzeBatch') {
        // Analyze a batch of positions assigned to this worker
        const { variantName, startTurn, halfmoveClock, fullmoveNumber, allMoves, positions, depth, batchId } = msg;

        for (const moveIndex of positions) {
            const result = analyzePosition(
                variantName, startTurn, halfmoveClock, fullmoveNumber,
                allMoves, moveIndex, depth
            );
            self.postMessage({
                type: 'positionResult',
                batchId,
                ...result,
            });
        }

        self.postMessage({ type: 'batchDone', batchId });
        return;
    }

    if (msg.type === 'probe') {
        const ok = await initializeWasm();
        self.postMessage({ type: 'probeResult', ok });
    }
};
