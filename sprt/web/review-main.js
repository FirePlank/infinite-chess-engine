import { getVariantData, getAllVariants, generateSetupICN, engineLetterToICNCode } from './variants.js';

// ── Constants & Configuration ───────────────────────────────────────────────

const MATE_SCORE = 800000;

// Chess.com-style classification thresholds (expected points / win probability loss)
const CLASSIFICATION = {
    BEST:       { label: 'Best',       symbol: '★',  color: '#96bc4b', min: 0.00, max: 0.00 },
    EXCELLENT:  { label: 'Excellent',  symbol: '!',  color: '#96bc4b', min: 0.00, max: 0.02 },
    GOOD:       { label: 'Good',       symbol: '',   color: '#a0a0a0', min: 0.02, max: 0.05 },
    INACCURACY: { label: 'Inaccuracy', symbol: '?!', color: '#f7c631', min: 0.05, max: 0.10 },
    MISTAKE:    { label: 'Mistake',    symbol: '?',  color: '#e6912c', min: 0.10, max: 0.20 },
    BLUNDER:    { label: 'Blunder',    symbol: '??', color: '#ca3431', min: 0.20, max: 1.00 },
    FORCED:     { label: 'Forced',     symbol: '⇒',  color: '#7b9fc7', min: 0.00, max: 1.00 },
};

// ── Win Probability & Accuracy Formulas ─────────────────────────────────────

/**
 * Convert centipawn evaluation to win probability [0, 1].
 * Formula: wp = 0.5 + 0.5 * (2 / (1 + exp(-0.004 * cp)) - 1)
 * This is the chess.com-style sigmoid. For mate scores, returns 1 or 0.
 */
function cpToWinProb(cp) {
    if (cp >= MATE_SCORE) return 1.0;
    if (cp <= -MATE_SCORE) return 0.0;
    return 0.5 + 0.5 * (2.0 / (1.0 + Math.exp(-0.004 * cp)) - 1.0);
}

/**
 * Lichess-style per-move accuracy percentage.
 * Accuracy% = 103.1668 * exp(-0.04354 * wpLoss) - 3.1669
 * where wpLoss = winPercentBefore - winPercentAfter (as percentages 0-100)
 */
function moveAccuracyPercent(wpBefore, wpAfter) {
    const wpLossPct = (wpBefore - wpAfter) * 100; // Convert to 0-100 scale
    if (wpLossPct <= 0) return 100;
    const acc = 103.1668 * Math.exp(-0.04354 * wpLossPct) - 3.1669;
    return Math.max(0, Math.min(100, acc));
}

/**
 * Classify a move based on win probability loss.
 * isBestMove: whether the played move matches the engine's best move.
 */
function classifyMove(wpLoss, isBestMove) {
    if (isBestMove || wpLoss <= 0.001) return CLASSIFICATION.BEST;
    if (wpLoss <= CLASSIFICATION.EXCELLENT.max) return CLASSIFICATION.EXCELLENT;
    if (wpLoss <= CLASSIFICATION.GOOD.max) return CLASSIFICATION.GOOD;
    if (wpLoss <= CLASSIFICATION.INACCURACY.max) return CLASSIFICATION.INACCURACY;
    if (wpLoss <= CLASSIFICATION.MISTAKE.max) return CLASSIFICATION.MISTAKE;
    return CLASSIFICATION.BLUNDER;
}

/**
 * Compute overall game accuracy using lichess-style harmonic mean + weighted mean blend.
 */
function computeGameAccuracy(moveAccuracies) {
    if (moveAccuracies.length === 0) return 0;

    // Simple harmonic mean
    let harmonicSum = 0;
    let validCount = 0;
    for (const acc of moveAccuracies) {
        if (acc > 0) {
            harmonicSum += 1.0 / acc;
            validCount++;
        }
    }
    const harmonicMean = validCount > 0 ? validCount / harmonicSum : 0;

    // Arithmetic mean as fallback/blend component
    const arithmeticMean = moveAccuracies.reduce((a, b) => a + b, 0) / moveAccuracies.length;

    // Blend: average of harmonic and arithmetic (simplified lichess approach)
    return (harmonicMean + arithmeticMean) / 2;
}

// ── ICN Parsing ─────────────────────────────────────────────────────────────

/**
 * Parse a full ICN game string into structured data.
 * Handles header tags like [Variant "..."], [White "..."], etc.
 * Then parses the position body and moves.
 */
function parseICN(icnText) {
    const text = icnText.trim();
    const headers = {};
    let body = text;

    // Extract all [Key "Value"] headers
    const headerRegex = /\[(\w+)\s+"([^"]*)"\]/g;
    let match;
    while ((match = headerRegex.exec(text)) !== null) {
        headers[match[1]] = match[2];
    }

    // Remove headers from body
    body = body.replace(/\[[^\]]*\]/g, '').trim();

    const variantName = headers.Variant || 'Classical';

    // Tokenize body
    const tokens = body.split(/\s+/).filter(t => t.length > 0);

    let startTurn = 'w';
    let halfmoveClock = 0;
    let moveRuleLimit = 100;
    let fullmoveNumber = 1;
    let piecesToken = null;
    const moveStrings = [];

    let movesRawStr = '';
    let foundMoves = false;

    for (let ti = 0; ti < tokens.length; ti++) {
        const token = tokens[ti];
        if (token === '-') continue;

        // Once we've found moves, collect the rest as moves (handles comments with spaces)
        if (foundMoves) {
            movesRawStr += ' ' + token;
            continue;
        }

        // Moves contain '>' or 'x' (capture notation)
        if (token.includes('>') || /[A-Za-z\d],[A-Za-z\d].*x/.test(token)) {
            foundMoves = true;
            movesRawStr = token;
            continue;
        }

        if (token === 'w') { startTurn = 'w'; continue; }
        if (token === 'b') { startTurn = 'b'; continue; }

        // Halfmove clock / move rule: "0/100"
        if (token.includes('/') && /^\d/.test(token)) {
            const parts = token.split('/');
            halfmoveClock = parseInt(parts[0], 10) || 0;
            if (parts.length > 1) moveRuleLimit = parseInt(parts[1], 10) || 100;
            continue;
        }

        // Promotion rules: (8;q,r,b,n|1;q,r,b,n)
        if (token.startsWith('(') && token.endsWith(')')) continue;

        // World bounds: 4 comma-separated numbers
        if (token.split(',').length === 4 && /^[-\d,]+$/.test(token)) continue;

        // Fullmove number (plain integer)
        if (/^\d+$/.test(token)) {
            fullmoveNumber = parseInt(token, 10) || 1;
            continue;
        }

        // Win conditions token (checkmate,royalcapture,etc.)
        if (/^[a-z]+(,[a-z]+)*$/.test(token) && !token.includes('|') && token.length < 100) continue;

        // Pieces token: contains '|' or uppercase piece codes with coordinates
        if (token.includes('|') || (token.includes(',') && /[A-Z]/.test(token))) {
            piecesToken = token;
            continue;
        }
    }

    // Split collected moves string by '|' delimiter
    if (movesRawStr) {
        for (const m of movesRawStr.split('|')) {
            if (m.trim()) moveStrings.push(m.trim());
        }
    }

    // Parse moves - extract the actual move coordinates and any embedded evals
    const moves = [];
    for (const ms of moveStrings) {
        // Separate move part from comment
        let movePart = ms;
        let comment = '';
        const braceIdx = ms.indexOf('{');
        if (braceIdx !== -1) {
            movePart = ms.substring(0, braceIdx).trim();
            const endBrace = ms.indexOf('}', braceIdx);
            comment = endBrace !== -1 ? ms.substring(braceIdx + 1, endBrace) : ms.substring(braceIdx + 1);
        }

        // Parse promotion
        let promotion = null;
        const eqIdx = movePart.indexOf('=');
        if (eqIdx !== -1) {
            promotion = movePart.substring(eqIdx + 1);
            movePart = movePart.substring(0, eqIdx);
        }

        // Strip annotation characters like ! ? # +
        movePart = movePart.replace(/[!?#+]+$/, '');

        // Normalize 'x' capture notation to '>' separator
        movePart = movePart.replace('x', '>');

        const parts = movePart.split('>');
        if (parts.length !== 2) continue;

        // Strip any residual annotations (+, #) from each coord part
        parts[0] = parts[0].replace(/[+#!?]+$/, '').trim();
        parts[1] = parts[1].replace(/[+#!?]+$/, '').trim();

        // Extract embedded eval from comment
        let embeddedEval = null;
        const evalMatch = comment.match(/\[%eval\s+([+-]?\d+\.?\d*)\]/);
        if (evalMatch) {
            embeddedEval = Math.round(parseFloat(evalMatch[1]) * 100); // Convert to centipawns
        }
        const mateMatch = comment.match(/\[%mate\s+([+-]?\d+)\]/);
        if (mateMatch) {
            const mateIn = parseInt(mateMatch[1], 10);
            embeddedEval = mateIn > 0 ? (900000 - mateIn * 2) : (-900000 - mateIn * 2);
        }

        moves.push({
            from: parts[0].trim(),
            to: parts[1].trim(),
            promotion,
            embeddedEval,
        });
    }

    return {
        headers,
        variantName,
        startTurn,
        halfmoveClock,
        moveRuleLimit,
        fullmoveNumber,
        piecesToken,
        moves,
    };
}

// ── UI State & DOM References ───────────────────────────────────────────────

let workers = [];
let analysisRunning = false;
let analysisResults = []; // Array of evals indexed by move index
let parsedGame = null;
let cancelRequested = false;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Analysis Engine ─────────────────────────────────────────────────────────

async function initWorkers(count) {
    // Terminate existing workers
    workers.forEach(w => { try { w.terminate(); } catch (e) { } });
    workers = [];

    const promises = [];
    for (let i = 0; i < count; i++) {
        const w = new Worker(new URL('./review-worker.js', import.meta.url), { type: 'module' });
        workers.push(w);
        promises.push(new Promise((resolve) => {
            w.onmessage = (e) => {
                if (e.data.type === 'initResult') resolve(e.data.ok);
            };
            w.postMessage({ type: 'init' });
        }));
    }

    const results = await Promise.all(promises);
    return results.every(Boolean);
}

function distributePositions(totalPositions, workerCount) {
    const assignments = Array.from({ length: workerCount }, () => []);
    for (let i = 0; i < totalPositions; i++) {
        assignments[i % workerCount].push(i);
    }
    return assignments;
}

async function runAnalysis(game, depth, workerCount) {
    analysisRunning = true;
    cancelRequested = false;

    const totalPositions = game.moves.length + 1; // N moves + 1 for starting position + final position
    analysisResults = new Array(totalPositions).fill(null);

    let completedCount = 0;
    updateProgress(0, totalPositions);

    // Convert moves to the format the worker expects
    const allMoves = game.moves.map(m => ({
        from: m.from,
        to: m.to,
        promotion: m.promotion,
    }));

    const assignments = distributePositions(totalPositions, workerCount);
    const batchId = Date.now();

    return new Promise((resolve) => {
        let doneBatches = 0;

        workers.forEach((w, idx) => {
            const positions = assignments[idx];
            if (positions.length === 0) {
                doneBatches++;
                if (doneBatches >= workerCount) {
                    analysisRunning = false;
                    resolve(analysisResults);
                }
                return;
            }

            w.onmessage = (e) => {
                const msg = e.data;
                if (msg.type === 'positionResult' && msg.batchId === batchId) {
                    analysisResults[msg.moveIndex] = msg;
                    completedCount++;
                    updateProgress(completedCount, totalPositions);
                    // Live-update the eval graph
                    renderEvalGraph();
                }
                if (msg.type === 'batchDone' && msg.batchId === batchId) {
                    doneBatches++;
                    if (doneBatches >= workerCount || cancelRequested) {
                        analysisRunning = false;
                        resolve(analysisResults);
                    }
                }
            };

            w.postMessage({
                type: 'analyzeBatch',
                variantName: game.variantName,
                startTurn: game.startTurn,
                halfmoveClock: game.halfmoveClock,
                fullmoveNumber: game.fullmoveNumber,
                allMoves,
                positions,
                depth,
                batchId,
            });
        });
    });
}

// ── Results Processing ──────────────────────────────────────────────────────

function processResults(game, results) {
    const moveAnalysis = [];
    const whiteAccuracies = [];
    const blackAccuracies = [];

    const classificationCounts = {
        white: { BEST: 0, EXCELLENT: 0, GOOD: 0, INACCURACY: 0, MISTAKE: 0, BLUNDER: 0, FORCED: 0 },
        black: { BEST: 0, EXCELLENT: 0, GOOD: 0, INACCURACY: 0, MISTAKE: 0, BLUNDER: 0, FORCED: 0 },
    };

    for (let i = 0; i < game.moves.length; i++) {
        const posBefore = results[i];
        const posAfter = results[i + 1];

        // Forced move: the position before this move had only 1 legal move (eval will be null)
        const isForced = posBefore && posBefore.legalMoveCount === 1;

        // Skip if missing critical data
        if (!posBefore || !posAfter) {
            moveAnalysis.push({
                move: game.moves[i],
                moveNumber: Math.floor(i / 2) + 1,
                isWhite: (game.startTurn === 'w') ? (i % 2 === 0) : (i % 2 === 1),
                evalBefore: null,
                evalAfter: null,
                wpBefore: null,
                wpAfter: null,
                wpLoss: null,
                classification: null,
                accuracy: null,
                bestMove: posBefore ? posBefore.bestMove : null,
                isBestMove: false,
            });
            continue;
        }

        // Process forced moves
        if (isForced) {
            const isWhite = (game.startTurn === 'w') ? (i % 2 === 0) : (i % 2 === 1);
            const side = isWhite ? 'white' : 'black';
            classificationCounts[side]['FORCED']++;

            // For forced moves:
            // evalBefore = eval from the previous position (results[i-1]), negated to current perspective
            // evalAfter = same as evalBefore (forced move doesn't change eval)
            const prevEval = i >= 1 && results[i - 1] ? results[i - 1].eval : 0;
            const evalBefore = -prevEval;
            const evalAfter = evalBefore;

            moveAnalysis.push({
                move: game.moves[i],
                moveNumber: Math.floor(i / 2) + (game.startTurn === 'w' ? 1 : (i % 2 === 0 ? 1 : 1)),
                isWhite,
                evalBefore,
                evalAfter,
                wpBefore: cpToWinProb(evalBefore),
                wpAfter: cpToWinProb(evalAfter),
                wpLoss: 0, // Forced moves have no WP loss by definition
                classification: CLASSIFICATION.FORCED,
                accuracy: 100,
                bestMove: posBefore.bestMove,
                isBestMove: true,
            });
            continue;
        }

        // Eval from side-to-move's perspective
        const evalBefore = posBefore.eval;

        // If next position is forced, evalAfter = evalBefore (no change — the forced move carries the same eval)
        // Otherwise convert posAfter to the same side's perspective
        const evalAfter = posAfter.legalMoveCount === 1 ? evalBefore : -posAfter.eval;

        // Win probability from the mover's perspective
        const wpBefore = cpToWinProb(evalBefore);
        const wpAfter = cpToWinProb(evalAfter);
        const wpLoss = Math.max(0, wpBefore - wpAfter);

        // Check if the played move matches the engine's best move
        // Strip piece prefix from ICN move coords (e.g. "Q4,1" -> "4,1")
        const bestMove = posBefore.bestMove;
        const playedMove = game.moves[i];
        const strippedFrom = playedMove.from.replace(/^[A-Za-z]+/, '');
        const strippedTo   = playedMove.to.replace(/^[A-Za-z]+/, '');
        const isBestMove = bestMove &&
            bestMove.from === strippedFrom &&
            bestMove.to === strippedTo;

        const classification = classifyMove(wpLoss, isBestMove);
        const accuracy = moveAccuracyPercent(wpBefore, wpAfter);

        const isWhite = (game.startTurn === 'w') ? (i % 2 === 0) : (i % 2 === 1);

        // Track per-side stats
        const side = isWhite ? 'white' : 'black';
        for (const [key, cls] of Object.entries(CLASSIFICATION)) {
            if (cls === classification) {
                classificationCounts[side][key]++;
                break;
            }
        }

        if (isWhite) {
            whiteAccuracies.push(accuracy);
        } else {
            blackAccuracies.push(accuracy);
        }

        moveAnalysis.push({
            move: game.moves[i],
            moveNumber: Math.floor(i / 2) + (game.startTurn === 'w' ? 1 : (i % 2 === 0 ? 1 : 1)),
            isWhite,
            evalBefore,
            evalAfter,
            wpBefore,
            wpAfter,
            wpLoss,
            classification,
            accuracy,
            bestMove,
            isBestMove,
        });
    }

    // Compute move numbers properly
    let moveNum = game.fullmoveNumber;
    for (let i = 0; i < moveAnalysis.length; i++) {
        moveAnalysis[i].moveNumber = moveNum;
        // Increment after black's move (or after white's move if game started with black)
        if (!moveAnalysis[i].isWhite) moveNum++;
    }

    return {
        moveAnalysis,
        whiteAccuracy: computeGameAccuracy(whiteAccuracies),
        blackAccuracy: computeGameAccuracy(blackAccuracies),
        classificationCounts,
    };
}

// ── Eval Graph Rendering (Canvas) ───────────────────────────────────────────

// Store last known good canvas data for hover
let _graphMeta = null;

function renderEvalGraph() {
    const canvas = $('#evalCanvas');
    if (!canvas || !parsedGame) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const rect = canvas.parentElement.getBoundingClientRect();
    // Fallback: if layout not yet computed, use the panel's offsetWidth
    const rawW = rect.width > 10 ? rect.width : (canvas.parentElement.offsetWidth || 600);
    canvas.width = rawW * dpr;
    canvas.height = 200 * dpr;
    canvas.style.width = rawW + 'px';
    canvas.style.height = '200px';
    ctx.scale(dpr, dpr);

    const W = rawW;
    const H = 200;
    const PAD_LEFT = 40;
    const PAD_RIGHT = 10;
    const PAD_TOP = 10;
    const PAD_BOTTOM = 20;
    const graphW = W - PAD_LEFT - PAD_RIGHT;
    const graphH = H - PAD_TOP - PAD_BOTTOM;

    // Background
    ctx.fillStyle = '#1c1b22';
    ctx.fillRect(0, 0, W, H);

    const totalMoves = parsedGame.moves.length;
    if (totalMoves === 0) return;

    // Collect evals (from White's perspective)
    const evals = [];
    let lastEval = 0;
    for (let i = 0; i <= totalMoves; i++) {
        const r = analysisResults[i];
        if (r && r.eval !== null && r.eval !== undefined) {
            let evalWhite;
            if (r.legalMoveCount === 1) {
                // Forced position: use same eval as before (flat line)
                evalWhite = lastEval;
            } else {
                const isWhiteTurn = (parsedGame.startTurn === 'w') ? (i % 2 === 0) : (i % 2 === 1);
                // Convert to White's perspective
                evalWhite = isWhiteTurn ? r.eval : -r.eval;
                lastEval = evalWhite;
            }
            evals.push({ idx: i, eval: evalWhite });
        }
    }

    if (evals.length < 2) return;

    // Scale: clamp eval to +-500 cp for display, use sigmoid for compression
    const evalToY = (cp) => {
        const clamped = Math.max(-600, Math.min(600, cp));
        const normalized = clamped / 600; // -1 to 1
        const y = PAD_TOP + graphH / 2 - normalized * (graphH / 2);
        return y;
    };

    const xScale = (idx) => PAD_LEFT + (idx / totalMoves) * graphW;

    // Zero line
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD_LEFT, evalToY(0));
    ctx.lineTo(PAD_LEFT + graphW, evalToY(0));
    ctx.stroke();

    // ±100, ±300 lines
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 0.5;
    for (const cp of [-300, -100, 100, 300]) {
        ctx.beginPath();
        ctx.setLineDash([3, 5]);
        ctx.moveTo(PAD_LEFT, evalToY(cp));
        ctx.lineTo(PAD_LEFT + graphW, evalToY(cp));
        ctx.stroke();
    }
    ctx.setLineDash([]);

    // Y-axis labels
    ctx.fillStyle = 'rgba(160,158,180,0.6)';
    ctx.font = '10px Inter,sans-serif';
    ctx.textAlign = 'right';
    for (const cp of [-300, -100, 0, 100, 300]) {
        const label = cp === 0 ? '0' : (cp > 0 ? '+' + (cp / 100).toFixed(0) : (cp / 100).toFixed(0));
        ctx.fillText(label, PAD_LEFT - 4, evalToY(cp) + 3);
    }

    // Fill areas above/below zero
    // White area (above zero, positive eval)
    ctx.beginPath();
    ctx.moveTo(xScale(evals[0].idx), evalToY(0));
    for (const e of evals) {
        const y = evalToY(Math.max(0, e.eval));
        ctx.lineTo(xScale(e.idx), y);
    }
    ctx.lineTo(xScale(evals[evals.length - 1].idx), evalToY(0));
    ctx.closePath();
    ctx.fillStyle = 'rgba(240,235,255,0.10)';
    ctx.fill();

    // Black area (below zero, negative eval)
    ctx.beginPath();
    ctx.moveTo(xScale(evals[0].idx), evalToY(0));
    for (const e of evals) {
        const y = evalToY(Math.min(0, e.eval));
        ctx.lineTo(xScale(e.idx), y);
    }
    ctx.lineTo(xScale(evals[evals.length - 1].idx), evalToY(0));
    ctx.closePath();
    ctx.fillStyle = 'rgba(10,8,20,0.35)';
    ctx.fill();

    // Draw eval line
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(220,215,255,0.9)';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < evals.length; i++) {
        const x = xScale(evals[i].idx);
        const y = evalToY(evals[i].eval);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw colored dots for mistakes/blunders
    if (window._moveAnalysis) {
        for (let mi = 0; mi < window._moveAnalysis.length; mi++) {
            const ma = window._moveAnalysis[mi];
            if (!ma.classification) continue;
            if (ma.classification === CLASSIFICATION.INACCURACY ||
                ma.classification === CLASSIFICATION.MISTAKE ||
                ma.classification === CLASSIFICATION.BLUNDER) {
                // Position the dot at the move's starting position
                const r = analysisResults[mi];
                if (r && r.eval !== null) {
                    const evalWhite = ma.isWhite ? r.eval : -r.eval;
                    const x = xScale(mi);
                    const y = evalToY(evalWhite);
                    ctx.beginPath();
                    ctx.arc(x, y, 4, 0, Math.PI * 2);
                    ctx.fillStyle = ma.classification.color;
                    ctx.fill();
                }
            }
        }
    }

    // X-axis: full-move numbers — only one label per full move
    ctx.fillStyle = 'rgba(160,160,180,0.7)';
    ctx.font = '10px -apple-system,BlinkMacSystemFont,Inter,sans-serif';
    ctx.textAlign = 'center';
    const startFullMove = parsedGame.fullmoveNumber;
    const totalFullMoves = Math.ceil(totalMoves / 2);
    const step = Math.max(1, Math.round(totalFullMoves / 14));
    // Iterate over full moves only (step 2 half-moves at a time)
    const plyStep = 2;
    for (let i = 0; i <= totalMoves; i += plyStep) {
        const plyOffset = parsedGame.startTurn === 'b' ? i + 1 : i;
        const fullMove = startFullMove + Math.floor(plyOffset / 2);
        if (fullMove === startFullMove || (fullMove - startFullMove) % step === 0) {
            ctx.fillText(String(fullMove), xScale(i), H - 3);
        }
    }

    // Store graph metadata for hover tooltip
    _graphMeta = { PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM, W, H, graphW, graphH, totalMoves, evalToY, xScale };
}

// ── Move List Rendering ─────────────────────────────────────────────────────

function formatEval(cp) {
    if (cp === null || cp === undefined) return '?';
    if (Math.abs(cp) >= MATE_SCORE) {
        const sign = cp > 0 ? '+' : '-';
        const mateIn = Math.ceil((900000 - Math.abs(cp)) / 2);
        return sign + 'M' + Math.max(1, mateIn);
    }
    const val = (cp / 100).toFixed(2);
    return cp >= 0 ? '+' + val : val;
}

function makeMoveCell(ma, i) {
    const cls = ma.classification;
    const clsKey = cls ? Object.keys(CLASSIFICATION).find(k => CLASSIFICATION[k] === cls) || '' : '';
    const clsClass = clsKey.toLowerCase();

    const moveEl = document.createElement('div');
    moveEl.className = 'move-cell ' + clsClass;
    moveEl.dataset.index = i;

    const moveText = document.createElement('span');
    moveText.className = 'move-text';
    const fromCoord = ma.move.from.replace(/^[A-Za-z]+/, '');
    const toCoord   = ma.move.to.replace(/^[A-Za-z]+/, '');
    moveText.textContent = fromCoord + '>' + toCoord + (ma.move.promotion ? '=' + ma.move.promotion : '');
    moveEl.appendChild(moveText);

    if (cls && cls.symbol) {
        const sym = document.createElement('span');
        sym.className = 'move-sym';
        sym.style.background = cls.color;
        sym.textContent = cls.symbol;
        sym.title = cls.label + (ma.wpLoss !== null ? ' (' + (ma.wpLoss * 100).toFixed(1) + '% WP loss)' : '');
        moveEl.appendChild(sym);
    }

    const evalEl = document.createElement('span');
    evalEl.className = 'move-ev';
    const evalForDisplay = ma.evalAfter !== null ? (ma.isWhite ? ma.evalAfter : -ma.evalAfter) : null;
    evalEl.textContent = formatEval(evalForDisplay);
    moveEl.appendChild(evalEl);

    moveEl.addEventListener('click', () => {
        const isSelected = moveEl.classList.contains('selected');
        $$('.move-cell.selected').forEach(el => el.classList.remove('selected'));
        if (isSelected) {
            // Deselect: close detail panel
            $('#moveDetail').innerHTML = '<p class="detail-empty">Click a move to see details</p>';
        } else {
            // Select: show detail
            moveEl.classList.add('selected');
            showMoveDetail(ma, i);
        }
    });
    return moveEl;
}

function renderMoveList(moveAnalysis) {
    const container = $('#moveList');
    container.innerHTML = '';

    let currentMoveNum = -1;
    let row = null;

    for (let i = 0; i < moveAnalysis.length; i++) {
        const ma = moveAnalysis[i];

        if (ma.isWhite) {
            // Start a new row for each white move
            row = document.createElement('div');
            row.className = 'move-row';
            const numEl = document.createElement('span');
            numEl.className = 'move-number';
            numEl.textContent = ma.moveNumber + '.';
            row.appendChild(numEl);
            row.appendChild(makeMoveCell(ma, i));
            container.appendChild(row);
            currentMoveNum = ma.moveNumber;
        } else {
            if (!row || currentMoveNum !== ma.moveNumber) {
                // Black move without a preceding white move on this row
                row = document.createElement('div');
                row.className = 'move-row';
                const numEl = document.createElement('span');
                numEl.className = 'move-number';
                numEl.textContent = ma.moveNumber + '.';
                row.appendChild(numEl);
                const spacer = document.createElement('div');
                spacer.className = 'move-cell spacer';
                spacer.textContent = '...';
                row.appendChild(spacer);
                container.appendChild(row);
                currentMoveNum = ma.moveNumber;
            }
            row.appendChild(makeMoveCell(ma, i));
        }
    }
}

function showMoveDetail(ma, index) {
    const detail = $('#moveDetail');
    if (!ma.classification) {
        detail.innerHTML = '<p class="dim">No analysis data for this move.</p>';
        return;
    }

    const bestMoveStr = ma.bestMove ? (ma.bestMove.from + '>' + ma.bestMove.to) : '?';
    const fromCoord = ma.move.from.replace(/^[A-Za-z]+/, '');
    const toCoord   = ma.move.to.replace(/^[A-Za-z]+/, '');
    const playedStr = fromCoord + '>' + toCoord + (ma.move.promotion ? '=' + ma.move.promotion : '');
    const sideLabel = ma.isWhite ? 'White' : 'Black';
    const evalBeforeDisp = formatEval(ma.evalBefore);
    const evalAfterDisp  = formatEval(ma.evalAfter);
    const wpBeforeDisp = ma.wpBefore !== null ? (ma.wpBefore * 100).toFixed(1) + '%' : '?';
    const wpAfterDisp  = ma.wpAfter  !== null ? (ma.wpAfter  * 100).toFixed(1) + '%' : '?';
    const isBest = ma.isBestMove;
    const isForced = ma.classification === CLASSIFICATION.FORCED;

    detail.innerHTML = `
        <div class="detail-badge" style="background:${ma.classification.color}22;color:${ma.classification.color}">
            <span style="font-size:17px">${ma.classification.symbol || '★'}</span>
            <span>${ma.classification.label}</span>
            <span style="margin-left:auto;font-size:12px;font-weight:500;opacity:0.85">${!isForced && ma.accuracy !== null ? ma.accuracy.toFixed(1) + '% acc' : ''}</span>
        </div>
        ${isForced ? `<div style="font-size:11px;color:var(--text-dim);padding:2px 0 6px 0;">Only legal move — evaluation carried forward from previous position.</div>` : ''}
        <div class="detail-meta">${sideLabel} · Move ${ma.moveNumber}</div>
        <div class="detail-moves">
            <div class="detail-move-box">
                <div class="dmb-label">Played</div>
                <div class="dmb-val" style="color:${ma.classification.color}">${playedStr}</div>
            </div>
            <div class="detail-move-box">
                <div class="dmb-label">Best</div>
                <div class="dmb-val" style="color:var(--best)">${isBest ? playedStr : bestMoveStr}</div>
            </div>
        </div>
        <div class="detail-stats">
            <div class="dstat"><span class="dstat-label">Eval before</span><span class="dstat-val">${evalBeforeDisp}</span></div>
            <div class="dstat"><span class="dstat-label">Eval after</span><span class="dstat-val">${evalAfterDisp}</span></div>
            <div class="dstat"><span class="dstat-label">Win% before</span><span class="dstat-val">${wpBeforeDisp}</span></div>
            <div class="dstat"><span class="dstat-label">Win% after</span><span class="dstat-val">${wpAfterDisp}</span></div>
            <div class="dstat"><span class="dstat-label">WP loss</span><span class="dstat-val" style="color:${ma.classification.color}">${ma.wpLoss !== null ? (ma.wpLoss * 100).toFixed(2) + '%' : '?'}</span></div>
        </div>
    `;
}

// ── Stats Rendering ─────────────────────────────────────────────────────────

function renderStats(result) {
    // Accuracy bars
    const whiteBar = $('#whiteAccBar');
    const blackBar = $('#blackAccBar');
    const whiteVal = $('#whiteAccVal');
    const blackVal = $('#blackAccVal');

    whiteVal.textContent = result.whiteAccuracy.toFixed(1) + '%';
    blackVal.textContent = result.blackAccuracy.toFixed(1) + '%';
    whiteBar.style.width = result.whiteAccuracy + '%';
    blackBar.style.width = result.blackAccuracy + '%';

    // Color the bars based on accuracy
    whiteBar.style.background = getAccuracyColor(result.whiteAccuracy);
    blackBar.style.background = getAccuracyColor(result.blackAccuracy);

    // Classification breakdown
    renderClassificationBreakdown('whiteBreakdown', result.classificationCounts.white);
    renderClassificationBreakdown('blackBreakdown', result.classificationCounts.black);
}

function getAccuracyColor(acc) {
    if (acc >= 92) return '#5fa855';
    if (acc >= 75) return '#7aaa6a';
    if (acc >= 55) return '#d4a72c';
    if (acc >= 35) return '#d47c2c';
    return '#c94040';
}

function renderClassificationBreakdown(containerId, counts) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    for (const [key, cls] of Object.entries(CLASSIFICATION)) {
        if (key === 'FORCED') continue; // Skip forced moves in breakdown
        const count = counts[key] || 0;
        if (count === 0) continue;
        const row = document.createElement('div');
        row.className = 'cls-row';
        row.innerHTML = `<span class="cls-pip" style="background:${cls.color}"></span><span class="cls-name">${cls.label}</span><span class="cls-count">${count}</span>`;
        container.appendChild(row);
    }
}

// ── Progress ────────────────────────────────────────────────────────────────

function updateProgress(completed, total) {
    const pct = total > 0 ? (completed / total * 100) : 0;
    const bar = $('#progressBar');
    const text = $('#progressText');
    const pctEl = $('#progressPct');
    if (bar) bar.style.width = pct + '%';
    if (text) text.textContent = `Analyzing ${completed} of ${total} positions`;
    if (pctEl) pctEl.textContent = pct.toFixed(0) + '%';
}

// ── Main Flow ───────────────────────────────────────────────────────────────

async function startAnalysis() {
    const icnInput = $('#icnInput').value.trim();
    if (!icnInput) return;

    const depth = parseInt($('#depthInput').value, 10) || 12;
    const workerCount = Math.max(1, parseInt($('#workersInput').value, 10) || navigator.hardwareConcurrency || 4);

    // Show analysis panel
    $('#resultsPanel').classList.remove('hidden');
    $('#moveListPanel').classList.add('hidden');
    $('#progressPanel').classList.remove('hidden');
    // Reset accuracy values for clean state
    $('#whiteAccVal').textContent = '—';
    $('#blackAccVal').textContent = '—';
    $('#whiteAccBar').style.width = '0%';
    $('#blackAccBar').style.width = '0%';
    $('#whiteBreakdown').innerHTML = '';
    $('#blackBreakdown').innerHTML = '';
    $('#analyzeBtn').disabled = true;
    $('#cancelBtn').classList.remove('hidden');
    $('#moveDetail').innerHTML = '<p class="dim">Click a move to see details</p>';
    $('#moveList').innerHTML = '';

    try {
        // Parse ICN
        parsedGame = parseICN(icnInput);

        // Override variant from dropdown if set and ICN doesn't specify one
        const variantOverride = $('#variantHint').value;
        if (variantOverride && (!parsedGame.headers.Variant || parsedGame.headers.Variant === 'Classical')) {
            parsedGame.variantName = variantOverride;
        }

        if (parsedGame.moves.length === 0) {
            alert('No moves found in the ICN input.');
            $('#analyzeBtn').disabled = false;
            $('#cancelBtn').classList.add('hidden');
            return;
        }

        // Show game info
        const white = parsedGame.headers.White || 'White';
        const black = parsedGame.headers.Black || 'Black';
        const variant = parsedGame.variantName;
        $('#gameInfo').classList.remove('hidden');
        $('#gameInfo').textContent = `${white} vs ${black} · ${variant} · ${parsedGame.moves.length} moves`;

        // Show stats panel early so canvas has proper dimensions during live updates
        $('#statsPanel').classList.remove('hidden');

        // Initialize workers
        updateProgress(0, parsedGame.moves.length + 1);
        const workersOk = await initWorkers(workerCount);
        if (!workersOk) {
            alert('Failed to initialize WASM workers.');
            $('#analyzeBtn').disabled = false;
            $('#cancelBtn').classList.add('hidden');
            return;
        }

        // Run analysis
        const startTime = performance.now();
        await runAnalysis(parsedGame, depth, workerCount);
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

        if (cancelRequested) {
            $('#progressText').textContent = 'Analysis cancelled.';
            return;
        }

        // Process results
        const result = processResults(parsedGame, analysisResults);
        window._moveAnalysis = result.moveAnalysis;

        // Update UI
        $('#progressPanel').classList.add('hidden');
        $('#statsPanel').classList.remove('hidden');
        $('#moveListPanel').classList.remove('hidden');

        renderStats(result);
        renderMoveList(result.moveAnalysis);
        renderEvalGraph();

        $('#gameInfo').textContent += ` · depth ${depth} · ${elapsed}s`;

    } catch (e) {
        console.error('Analysis error:', e);
        alert('Error: ' + e.message);
    } finally {
        analysisRunning = false;
        $('#analyzeBtn').disabled = false;
        $('#cancelBtn').classList.add('hidden');
    }
}

function cancelAnalysis() {
    cancelRequested = true;
    workers.forEach(w => { try { w.terminate(); } catch (e) { } });
    workers = [];
    analysisRunning = false;
    $('#analyzeBtn').disabled = false;
    $('#cancelBtn').classList.add('hidden');
    $('#progressText').textContent = 'Analysis cancelled.';
}

// ── Initialization ──────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    $('#analyzeBtn').addEventListener('click', startAnalysis);
    $('#cancelBtn').addEventListener('click', cancelAnalysis);

    // Populate variant dropdown
    const variantSelect = $('#variantHint');
    if (variantSelect) {
        for (const v of getAllVariants()) {
            const opt = document.createElement('option');
            opt.value = v;
            opt.textContent = v;
            variantSelect.appendChild(opt);
        }
    }

    // Default workers to navigator.hardwareConcurrency
    const workersInput = $('#workersInput');
    if (workersInput && !workersInput.value) {
        workersInput.value = Math.max(1, navigator.hardwareConcurrency || 4);
    }

    // Handle window resize for eval graph
    window.addEventListener('resize', () => {
        if (parsedGame && analysisResults.length > 0) {
            renderEvalGraph();
        }
    });

    // Canvas hover tooltip
    const canvas = $('#evalCanvas');
    const tooltip = document.createElement('div');
    tooltip.id = 'graphTooltip';
    tooltip.style.cssText = 'position:fixed;background:#252540;border:1px solid #3a3a58;border-radius:6px;padding:6px 10px;font-size:12px;color:#e8e8f0;pointer-events:none;display:none;z-index:100;white-space:nowrap;box-shadow:0 4px 12px rgba(0,0,0,0.4)';
    document.body.appendChild(tooltip);

    canvas.addEventListener('mousemove', (e) => {
        if (!_graphMeta || !window._moveAnalysis || !parsedGame) { tooltip.style.display = 'none'; return; }
        const { PAD_LEFT, PAD_RIGHT, W, graphW, totalMoves, evalToY, xScale } = _graphMeta;
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Convert mouseX to position index
        const rawIdx = ((mouseX - PAD_LEFT) / graphW) * totalMoves;
        const idx = Math.round(Math.max(0, Math.min(totalMoves, rawIdx)));

        // Only show if within graph area
        if (mouseX < PAD_LEFT || mouseX > PAD_LEFT + graphW) { tooltip.style.display = 'none'; return; }

        const result = analysisResults[idx];
        if (!result || result.eval === null) { tooltip.style.display = 'none'; return; }

        const isWhiteTurn = (parsedGame.startTurn === 'w') ? (idx % 2 === 0) : (idx % 2 === 1);
        
        // For forced positions, use inherited eval from previous position (same as graph)
        let evalWhite;
        if (result.legalMoveCount === 1 && idx >= 1) {
            const prevResult = analysisResults[idx - 1];
            if (prevResult) {
                const prevIsWhiteTurn = (parsedGame.startTurn === 'w') ? ((idx - 1) % 2 === 0) : ((idx - 1) % 2 === 1);
                evalWhite = prevIsWhiteTurn ? prevResult.eval : -prevResult.eval;
            } else {
                evalWhite = isWhiteTurn ? result.eval : -result.eval;
            }
        } else {
            // Normal: convert to White's perspective
            evalWhite = isWhiteTurn ? result.eval : -result.eval;
        }
        const evalStr = formatEval(evalWhite);

        let moveInfo = '';
        if (idx === 0) {
            moveInfo = 'Start position';
        } else {
            const ma = window._moveAnalysis[idx - 1];
            if (ma) {
                const f = ma.move.from.replace(/^[A-Za-z]+/, '');
                const t = ma.move.to.replace(/^[A-Za-z]+/, '');
                const side = ma.isWhite ? 'White' : 'Black';
                const sym = ma.classification ? ma.classification.symbol : '';
                moveInfo = `${ma.moveNumber}. ${side}: ${f}>${t}${sym ? ' ' + sym : ''}`;
            }
        }

        tooltip.innerHTML = `<span style="color:#888;margin-right:6px">${moveInfo}</span><strong>${evalStr}</strong>`;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 12) + 'px';
        tooltip.style.top  = (e.clientY - 28) + 'px';
    });

    canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

    // Click on graph selects the nearest move in the list
    canvas.addEventListener('click', (e) => {
        if (!_graphMeta || !window._moveAnalysis || !parsedGame) return;
        const { PAD_LEFT, graphW, totalMoves } = _graphMeta;
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const rawIdx = ((mouseX - PAD_LEFT) / graphW) * totalMoves;
        const posIdx = Math.round(Math.max(0, Math.min(totalMoves, rawIdx)));
        const moveIdx = Math.max(0, posIdx - 1);
        const ma = window._moveAnalysis[moveIdx];
        if (!ma) return;
        // Select in move list
        const allCells = $$('.move-cell[data-index]');
        allCells.forEach(el => el.classList.remove('selected'));
        const target = $$(`.move-cell[data-index="${moveIdx}"]`)[0];
        if (target) {
            target.classList.add('selected');
            target.scrollIntoView({ block: 'nearest' });
            showMoveDetail(ma, moveIdx);
        }
    });
    canvas.style.cursor = 'crosshair';
});

export { startAnalysis, cancelAnalysis };
