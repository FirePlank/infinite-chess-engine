import initOld, { Engine as EngineOld } from './pkg-old/hydrochess_wasm.js';
import initNew, { Engine as EngineNew } from './pkg-new/hydrochess_wasm.js';

// UI Elements
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const sprtBoundsPreset = document.getElementById('sprtBoundsPreset');
const sprtBoundsMode = document.getElementById('sprtBoundsMode');
const sprtAlphaEl = document.getElementById('sprtAlpha');
const sprtBetaEl = document.getElementById('sprtBeta');
const sprtTimeControlEl = document.getElementById('sprtTimeControl');
const sprtConcurrencyEl = document.getElementById('sprtConcurrency');
const sprtMinGames = document.getElementById('sprtMinGames');
const sprtMaxGames = document.getElementById('sprtMaxGames');
const sprtMaxMoves = document.getElementById('sprtMaxMoves');
const sprtMaterialThresholdEl = document.getElementById('sprtMaterialThreshold');
const runSprtBtn = document.getElementById('runSprt');
const stopSprtBtn = document.getElementById('stopSprt');
const sprtWinsEl = document.getElementById('sprtWins');
const sprtLossesEl = document.getElementById('sprtLosses');
const sprtDrawsEl = document.getElementById('sprtDraws');
const sprtEloEl = document.getElementById('sprtElo');
const sprtOutput = document.getElementById('sprtOutput');
const gameLogEl = document.getElementById('gameLog');
const copyLogBtn = document.getElementById('copyLog');
const downloadLogsBtn = document.getElementById('downloadLogs');
const downloadGamesBtn = document.getElementById('downloadGames');
const icnOutputEl = document.getElementById('icnOutput');
const icnTextEl = document.getElementById('icnText');
const sprtStatusEl = document.getElementById('sprtStatus');

let wasmReady = false;
let sprtRunning = false;
let stopRequested = false;
// Holds ICN strings for each completed game
let gameLogs = [];
let activeSprtWorkers = [];
// Last known stats snapshot (for final/partial result blocks)
let lastWins = 0;
let lastLosses = 0;
let lastDraws = 0;
let lastElo = 0;
let lastEloError = 0;
let lastLLR = 0;
let lastBounds = null;

// SPRT configuration (mirrors sprt.js)
const CONFIG = {
    elo0: -5,
    elo1: 5,
    alpha: 0.05,
    beta: 0.05,
    boundsPreset: 'all',
    boundsMode: 'gainer',
    timeControl: '10+0.1',
    maxGames: 1000,
    minGames: 500,
    maxMoves: 200,
    concurrency: 1,
    materialThreshold: 1500,
};

const WHITE_FIRST_MOVES = [
    // Pawn moves (16)
    { from: '1,2', to: '1,3' }, { from: '1,2', to: '1,4' },
    { from: '2,2', to: '2,3' }, { from: '2,2', to: '2,4' },
    { from: '3,2', to: '3,3' }, { from: '3,2', to: '3,4' },
    { from: '4,2', to: '4,3' }, { from: '4,2', to: '4,4' },
    { from: '5,2', to: '5,3' }, { from: '5,2', to: '5,4' },
    { from: '6,2', to: '6,3' }, { from: '6,2', to: '6,4' },
    { from: '7,2', to: '7,3' }, { from: '7,2', to: '7,4' },
    { from: '8,2', to: '8,3' }, { from: '8,2', to: '8,4' },
    // Knight moves (4)
    { from: '2,1', to: '1,3' }, { from: '2,1', to: '3,3' },
    { from: '7,1', to: '6,3' }, { from: '7,1', to: '8,3' },
];

function getRandomOpening() {
    return WHITE_FIRST_MOVES[Math.floor(Math.random() * WHITE_FIRST_MOVES.length)];
}

const BOUNDS_PRESETS = {
    stockfish_ltc: {
        gainer: [0.5, 2.5],
        nonreg: [-1.75, 0.25],
    },
    stockfish_stc: {
        gainer: [0, 2],
        nonreg: [-1.75, 0.25],
    },
    top30: {
        gainer: [0, 3],
        nonreg: [-3, 1],
    },
    top200: {
        gainer: [0, 5],
        nonreg: [-5, 0],
    },
    all: {
        gainer: [0, 10],
        nonreg: [-10, 0],
    },
};

function getStandardPosition() {
    const pieces = [];
    // White back rank
    pieces.push({ x: '1', y: '1', piece_type: 'r', player: 'w' });
    pieces.push({ x: '2', y: '1', piece_type: 'n', player: 'w' });
    pieces.push({ x: '3', y: '1', piece_type: 'b', player: 'w' });
    pieces.push({ x: '4', y: '1', piece_type: 'q', player: 'w' });
    pieces.push({ x: '5', y: '1', piece_type: 'k', player: 'w' });
    pieces.push({ x: '6', y: '1', piece_type: 'b', player: 'w' });
    pieces.push({ x: '7', y: '1', piece_type: 'n', player: 'w' });
    pieces.push({ x: '8', y: '1', piece_type: 'r', player: 'w' });
    // White pawns
    for (let i = 1; i <= 8; i++) {
        pieces.push({ x: String(i), y: '2', piece_type: 'p', player: 'w' });
    }
    // Black back rank
    pieces.push({ x: '1', y: '8', piece_type: 'r', player: 'b' });
    pieces.push({ x: '2', y: '8', piece_type: 'n', player: 'b' });
    pieces.push({ x: '3', y: '8', piece_type: 'b', player: 'b' });
    pieces.push({ x: '4', y: '8', piece_type: 'q', player: 'b' });
    pieces.push({ x: '5', y: '8', piece_type: 'k', player: 'b' });
    pieces.push({ x: '6', y: '8', piece_type: 'b', player: 'b' });
    pieces.push({ x: '7', y: '8', piece_type: 'n', player: 'b' });
    pieces.push({ x: '8', y: '8', piece_type: 'r', player: 'b' });
    // Black pawns
    for (let i = 1; i <= 8; i++) {
        pieces.push({ x: String(i), y: '7', piece_type: 'p', player: 'b' });
    }

    // Standard infinite-chess special rights: all pawns (double-step)
    // plus kings and rooks.
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
        board: { pieces: pieces },
        // Starting side; Engine::new will infer current turn from move_history.
        turn: 'w',
        // Support both old and new APIs in the browser sanity test as well.
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

// Generate a simple ICN string for a standard Classical game from worker log lines
// newPlaysWhite indicates which engine (new vs old) had White.
// endReason may be 'material_adjudication' or null.
// materialThreshold is the cp threshold used for adjudication, if any.
function generateICNFromWorkerLog(workerLog, gameIndex, result, newPlaysWhite, endReason, materialThreshold, timeControl) {
    const utc = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const utcDate = `${utc.getUTCFullYear()}.${pad(utc.getUTCMonth() + 1)}.${pad(utc.getUTCDate())}`;
    const utcTime = `${pad(utc.getUTCHours())}:${pad(utc.getUTCMinutes())}:${pad(utc.getUTCSeconds())}`;

    // Map result from SPRT's new-engine perspective to a PGN-style token
    // from the board perspective (White/Black), using newPlaysWhite.
    // - result === 'win'  means the NEW engine won.
    // - result === 'loss' means the NEW engine lost.
    // - newPlaysWhite indicates whether NEW had White.
    let resultToken = '*';
    if (result === 'draw') {
        resultToken = '1/2-1/2';
    } else if (result === 'win' || result === 'loss') {
        // Did White win from the board POV?
        const newWon = (result === 'win');
        const whiteWon = newPlaysWhite ? newWon : !newWon;
        resultToken = whiteWon ? '1-0' : '0-1';
    }

    const whiteEngine = newPlaysWhite ? 'HydroChess New' : 'HydroChess Old';
    const blackEngine = newPlaysWhite ? 'HydroChess Old' : 'HydroChess New';

    const headerList = [
        `[Event "SPRT Test Game ${gameIndex}"]`,
        `[Site "https://www.infinitechess.org/"]`,
        `[Variant "Classical"]`,
        `[Round "-"]`,
        `[UTCDate "${utcDate}"]`,
        `[UTCTime "${utcTime}"]`,
        `[Result "${resultToken}"]`,
        `[TimeControl "${(timeControl || '-').replace(/\s+/g, '')}"]`,
        `[White "${whiteEngine}"]`,
        `[Black "${blackEngine}"]`,
    ];

    if (endReason) {
        let termination = null;
        if (endReason === 'material_adjudication') {
            const th = typeof materialThreshold === 'number' && materialThreshold > 0 ? materialThreshold : 1500;
            termination = `Material adjudication (|eval| >= ${th} cp)`;
        } else if (endReason === 'illegal_move') {
            let detail = null;
            if (workerLog) {
                const illegalLine = workerLog.split('\n').find(l => l.startsWith('# Illegal move from '));
                if (illegalLine) {
                    detail = illegalLine.replace(/^# Illegal move from\s*/, '').trim();
                }
            }
            termination = detail
                ? `Loss on illegal move (${detail})`
                : 'Loss on illegal move';
        } else if (endReason === 'time_forfeit') {
            termination = 'Loss on time';
        } else if (endReason === 'checkmate') {
            termination = 'Checkmate';
        } else if (endReason === 'threefold') {
            termination = 'Draw by threefold repetition';
        } else if (endReason === 'fifty_move') {
            termination = 'Draw by fifty-move rule';
        } else if (endReason === 'insufficient_material') {
            termination = 'Draw by insufficient material';
        }
        if (termination) {
            headerList.push(`[Termination "${termination}"]`);
        }
    }

    const headers = headerList.join(' ');

    // Standard turn order / move counters from move count
    const lines = (workerLog || '').split('\n').filter(l => l.trim().length > 0 && (l.startsWith('W:') || l.startsWith('B:')));
    const moveCount = lines.length;
    const lastSide = moveCount > 0 ? (lines[moveCount - 1].startsWith('W:') ? 'w' : 'b') : 'b';
    // ICN here encodes the START position (standard classical), so we keep
    // the header state fixed at the initial values: White to move, zero
    // halfmove clock, and fullmove number 1. The move list then describes
    // the history from that start.
    const nextTurn = 'w';
    const fullmove = 1;
    const halfmove = 0;

    // Standard Classical starting position pieces (using coords x,y and simple abbreviations).
    // Mark pawns, rooks, and kings that have special rights (double-step or castling) with '+'.
    const pieces = [];
    const specials = new Set();
    for (let i = 1; i <= 8; i++) {
        specials.add(`P${i},2`);
        specials.add(`p${i},7`);
    }
    specials.add('R1,1');
    specials.add('R8,1');
    specials.add('K5,1');
    specials.add('r1,8');
    specials.add('r8,8');
    specials.add('k5,8');
    const push = (abbr, x, y) => {
        const base = `${abbr}${x},${y}`;
        pieces.push(specials.has(base) ? base + '+' : base);
    };
    // White back rank
    push('R', 1, 1); push('N', 2, 1); push('B', 3, 1); push('Q', 4, 1); push('K', 5, 1); push('B', 6, 1); push('N', 7, 1); push('R', 8, 1);
    for (let i = 1; i <= 8; i++) push('P', i, 2);
    // Black back rank
    push('r', 1, 8); push('n', 2, 8); push('b', 3, 8); push('q', 4, 8); push('k', 5, 8); push('b', 6, 8); push('n', 7, 8); push('r', 8, 8);
    for (let i = 1; i <= 8; i++) push('p', i, 7);

    // Moves string: parse worker log lines of form "W: x,y>u,v".
    const moves = lines.map((line) => {
        const idx = line.indexOf(':');
        if (idx === -1) return '';
        const rest = line.slice(idx + 1).trim(); // "x,y>u,v" or with promotion
        // Strip any leading piece info if ever added; currently we only have coords
        return rest.replace(/\s+/g, '');
    }).filter(Boolean);

    const movesStr = moves.join('|');
    return `${headers} ${nextTurn} ${halfmove}/100 ${fullmove} (8|1) ${pieces.join('|')}${movesStr ? ' ' + movesStr : ''}`;
}

function log(message, type) {
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = '<span class="log-time">[' + time + ']</span><span class="log-' + type + '">' + message + '</span>';
    gameLogEl.appendChild(entry);
    gameLogEl.scrollTop = gameLogEl.scrollHeight;
}

function sprtLog(message) {
    const entry = document.createElement('div');
    entry.textContent = message;
    sprtOutput.appendChild(entry);
    sprtOutput.scrollTop = sprtOutput.scrollHeight;
}

function clearLog() {
    gameLogEl.innerHTML = '';
}

function setStatus(status, text) {
    statusDot.className = 'status-dot ' + status;
    statusText.textContent = text;
}

function eloToScore(eloDiff) {
    return 1 / (1 + Math.pow(10, -eloDiff / 400));
}

function calculateBounds(alpha, beta) {
    const lower = Math.log(beta / (1 - alpha));
    const upper = Math.log((1 - beta) / alpha);
    return { lower, upper };
}

function parseTimeControl(str) {
    const raw = (str || '').toString().trim();
    if (!raw) return null;
    const parts = raw.split('+');
    const baseSec = parseFloat(parts[0]);
    if (!Number.isFinite(baseSec) || baseSec <= 0) return null;
    let incSec = 0;
    if (parts.length > 1 && parts[1].trim() !== '') {
        const incParsed = parseFloat(parts[1]);
        if (Number.isFinite(incParsed) && incParsed >= 0) {
            incSec = incParsed;
        }
    }
    const baseMs = Math.round(baseSec * 1000);
    const incMs = Math.round(incSec * 1000);
    return { baseSec, incSec, baseMs, incMs, tcString: raw };
}

function calculateLLR(wins, losses, draws, elo0, elo1) {
    const total = wins + losses + draws;
    if (total === 0) return 0;

    const score = (wins + draws * 0.5) / total;
    const s0 = eloToScore(elo0);
    const s1 = eloToScore(elo1);
    const clampedScore = Math.max(0.001, Math.min(0.999, score));

    const llr = total * (
        clampedScore * Math.log(s1 / s0) +
        (1 - clampedScore) * Math.log((1 - s1) / (1 - s0))
    );
    return llr;
}

function estimateElo(wins, losses, draws) {
    const total = wins + losses + draws;
    if (total === 0) return { elo: 0, error: 0 };

    const score = (wins + draws * 0.5) / total;
    if (score <= 0) return { elo: -999, error: 0 };
    if (score >= 1) return { elo: 999, error: 0 };

    const elo = -400 * Math.log10(1 / score - 1);

    const variance = (
        wins * Math.pow(1 - score, 2) +
        losses * Math.pow(0 - score, 2) +
        draws * Math.pow(0.5 - score, 2)
    ) / total;
    const stdDev = Math.sqrt(variance / total);
    const eloError = stdDev * 400 / (Math.log(10) * score * (1 - score));

    return { elo, error: Math.min(eloError, 200) };
}

function applyBoundsPreset() {
    const preset = BOUNDS_PRESETS[CONFIG.boundsPreset];
    if (!preset) return;
    const mode = CONFIG.boundsMode === 'nonreg' ? 'nonreg' : 'gainer';
    const pair = preset[mode];
    if (!pair) return;
    [CONFIG.elo0, CONFIG.elo1] = pair;
}

async function initWasm() {
    setStatus('loading', 'Loading WASM module...');
    try {
        // Initialize both old and new WASM modules
        await initOld();
        await initNew();
        wasmReady = true;
        setStatus('ready', 'WASM loaded and ready');
        runSprtBtn.disabled = false;
        log('WASM module initialized successfully', 'success');
        
        const testPos = getStandardPosition();
        const engine = new EngineNew(testPos);
        const move = engine.get_best_move_with_time(100);
        engine.free();
        log('Quick test: Best move = ' + (move ? move.from + ' to ' + move.to : 'null'), 'info');
    } catch (e) {
        setStatus('error', 'Failed to load WASM');
        log('Error loading WASM: ' + e.message, 'error');
        console.error(e);
    }
}

// No public single-move UI anymore; keep only internal sanity test in initWasm

function applyMove(position, move) {
    const pieces = position.board.pieces;
    const fromParts = move.from.split(',');
    const toParts = move.to.split(',');
    const fromX = fromParts[0];
    const fromY = fromParts[1];
    const toX = toParts[0];
    const toY = toParts[1];
    
    const capturedIdx = pieces.findIndex(function(p) {
        return p.x === toX && p.y === toY;
    });
    if (capturedIdx !== -1) {
        pieces.splice(capturedIdx, 1);
    }
    
    const movingPiece = pieces.find(function(p) {
        return p.x === fromX && p.y === fromY;
    });
    if (!movingPiece) {
        throw new Error('No piece at ' + move.from);
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
    const kings = position.board.pieces.filter(function(p) {
        return p.piece_type === 'k';
    });
    if (kings.length < 2) {
        return { over: true, reason: 'checkmate' };
    }
    if (position.board.pieces.length <= 2) {
        return { over: true, reason: 'draw' };
    }
    return { over: false };
}

async function runSprt() {
    if (!wasmReady || sprtRunning) return;
    
    sprtRunning = true;
    stopRequested = false;
    runSprtBtn.disabled = true;
    stopSprtBtn.disabled = false;
    
    // Read configuration from UI
    CONFIG.boundsPreset = sprtBoundsPreset.value || 'all';
    CONFIG.boundsMode = sprtBoundsMode.value || 'gainer';
    CONFIG.alpha = parseFloat(sprtAlphaEl.value) || 0.05;
    CONFIG.beta = parseFloat(sprtBetaEl.value) || 0.05;
    CONFIG.timeControl = (sprtTimeControlEl.value || '').trim() || '10+0.1';
    CONFIG.concurrency = parseInt(sprtConcurrencyEl.value, 10) || 1;
    CONFIG.minGames = parseInt(sprtMinGames.value, 10) || 500;
    CONFIG.maxGames = parseInt(sprtMaxGames.value, 10) || 1000;
    CONFIG.maxMoves = parseInt(sprtMaxMoves.value, 10) || 200;
    {
        const mt = parseInt(sprtMaterialThresholdEl.value, 10);
        CONFIG.materialThreshold = Number.isFinite(mt) && mt >= 0 ? mt : 1500;
    }

    // Ensure min/max games are even (for game pairing)
    if (CONFIG.minGames % 2 !== 0) CONFIG.minGames++;
    if (CONFIG.maxGames % 2 !== 0) CONFIG.maxGames++;

    applyBoundsPreset();
    const tc = parseTimeControl(CONFIG.timeControl);
    if (!tc) {
        log('Invalid time control: ' + CONFIG.timeControl + ' (expected base+inc in seconds, e.g. 10+0.1)', 'error');
        sprtRunning = false;
        runSprtBtn.disabled = false;
        stopSprtBtn.disabled = true;
        return;
    }
    const perMoveMs = Math.max(10, Math.round(((tc.baseSec / 20) + (tc.incSec / 2)) * 1000));
    const bounds = calculateBounds(CONFIG.alpha, CONFIG.beta);
    // reset last stats snapshot for this run
    lastBounds = bounds;
    lastWins = 0;
    lastLosses = 0;
    lastDraws = 0;
    lastElo = 0;
    lastEloError = 0;
    lastLLR = 0;
    const timePerMove = perMoveMs;
    const maxGames = CONFIG.maxGames;
    const maxMovesPerGame = CONFIG.maxMoves;
    
    let wins = 0;
    let losses = 0;
    let draws = 0;
    let llr = 0;
    gameLogs = [];
    
    sprtOutput.innerHTML = '';
    clearLog();
    sprtStatusEl.textContent = 'Status: running...';
    sprtStatusEl.className = 'sprt-status';
    log('Starting SPRT: ' + maxGames + ' games (' + (maxGames / 2) + ' pairs), TC=' + tc.tcString + ' (≈ ' + timePerMove + 'ms/move)', 'info');
    sprtLog('SPRT Test Started (random openings, paired games)');

    const maxConcurrent = Math.max(1, CONFIG.concurrency | 0);
    const workers = [];
    let activeWorkers = 0;
    let nextGameIndex = 0;
    // Store opening moves for each pair (pairIndex -> opening move)
    const pairOpenings = new Map();

    function getOpeningForPair(pairIndex) {
        if (!pairOpenings.has(pairIndex)) {
            pairOpenings.set(pairIndex, getRandomOpening());
        }
        return pairOpenings.get(pairIndex);
    }

    function startWorker(worker, id) {
        const gameIndex = nextGameIndex++;
        if (gameIndex >= maxGames) return false;
        activeWorkers++;
        // Games run in pairs: 0-1 share opening, 2-3 share opening, etc.
        // Even index = new engine plays white, odd index = old engine plays white
        const pairIndex = Math.floor(gameIndex / 2);
        const openingMove = getOpeningForPair(pairIndex);
        const newPlaysWhite = (gameIndex % 2) === 0;
        worker.postMessage({
            type: 'runGame',
            gameIndex,
            timePerMove,
            maxMoves: maxMovesPerGame,
            newPlaysWhite,
            openingMove,
            materialThreshold: CONFIG.materialThreshold,
            baseTimeMs: tc.baseMs,
            incrementMs: tc.incMs,
            timeControl: tc.tcString,
        });
        return true;
    }

    activeSprtWorkers = workers;

    await Promise.all(
        Array.from({ length: maxConcurrent }, async (_, i) => {
            const worker = new Worker(new URL('./sprt-worker.js', import.meta.url), { type: 'module' });
            workers.push(worker);

            await new Promise((resolve) => {
                // If no games available for this worker, resolve immediately
                if (nextGameIndex >= maxGames) {
                    resolve(undefined);
                    return;
                }
                
                worker.onmessage = (e) => {
                    const msg = e.data;
                    if (msg.type === 'result') {
                        const result = msg.result;
                        const icnLog = generateICNFromWorkerLog(
                            msg.log,
                            msg.gameIndex,
                            result,
                            msg.newPlaysWhite,
                            msg.reason,
                            msg.materialThreshold,
                            msg.timeControl,
                        );
                        gameLogs.push(icnLog);

                        if (result === 'win') wins++;
                        else if (result === 'loss') losses++;
                        else draws++;

                        const total = wins + losses + draws;
                        llr = calculateLLR(wins, losses, draws, CONFIG.elo0, CONFIG.elo1);
                        const { elo, error } = estimateElo(wins, losses, draws);

                        // update last stats snapshot so Stop can show partial results
                        lastWins = wins;
                        lastLosses = losses;
                        lastDraws = draws;
                        lastElo = elo;
                        lastEloError = error;
                        lastLLR = llr;

                        sprtWinsEl.textContent = String(wins);
                        sprtLossesEl.textContent = String(losses);
                        sprtDrawsEl.textContent = String(draws);
                        sprtEloEl.textContent = String(Math.round(elo));

                        // Extract opening move from log (first line)
                        const firstLine = (msg.log || '').split('\n')[0] || '';
                        const openingInfo = firstLine.includes('>') ? ' [' + firstLine.replace('W: ', '') + ']' : '';

                        sprtLog('Game ' + total + ': ' + result +
                            ' (W:' + wins + ' L:' + losses + ' D:' + draws + ')' +
                            ' Elo≈' + elo.toFixed(1) + '±' + error.toFixed(1) +
                            ' LLR=' + llr.toFixed(2) + openingInfo);

                        log(
                            'Games: ' + total + '/' + maxGames +
                            '  W:' + wins + ' L:' + losses + ' D:' + draws +
                            '  Elo≈' + elo.toFixed(1) + '±' + error.toFixed(1) +
                            '  LLR ' + llr.toFixed(2) +
                            ' in [' + bounds.lower.toFixed(2) + ', ' + bounds.upper.toFixed(2) + ']',
                            'info'
                        );

                        // Only check SPRT termination after even number of games (completed pairs)
                        const canTerminate = (total % 2 === 0);
                        const reachedBounds = canTerminate &&
                            total >= CONFIG.minGames && (llr >= bounds.upper || llr <= bounds.lower);
                        const reachedMax = canTerminate && total >= CONFIG.maxGames;

                        activeWorkers--;

                        if (!stopRequested && !reachedBounds && !reachedMax) {
                            // Try to start another game; if none left, resolve
                            if (!startWorker(worker, i)) {
                                resolve(undefined);
                            }
                        } else {
                            // This worker is done; resolve its promise
                            resolve(undefined);
                        }
                    } else if (msg.type === 'error') {
                        console.error('Worker error for game', msg.gameIndex, msg.error);
                        activeWorkers--;
                        // This worker encountered an error; resolve its promise
                        resolve(undefined);
                    }
                };

                startWorker(worker, i);
            });
        })
    );

    workers.forEach(w => w.terminate());
    activeSprtWorkers = [];
    
    const { elo: finalElo, error: finalErr } = estimateElo(wins, losses, draws);
    const verdict = llr >= bounds.upper ? 'PASSED (new > old)'
        : (llr <= bounds.lower ? 'FAILED (no gain)' : 'INCONCLUSIVE');

    log('SPRT Complete: ' + wins + 'W ' + losses + 'L ' + draws + 'D, Elo≈ ' +
        finalElo.toFixed(1) + '±' + finalErr.toFixed(1) + ' (' + verdict + ')', 'success');
    // Detailed final summary block similar to sprt.js printResult
    const totalGames = wins + losses + draws;
    const winRate = totalGames > 0 ? (((wins + draws * 0.5) / totalGames) * 100).toFixed(1) : '0.0';
    sprtLog('');
    sprtLog('═══════════════════════════════════════════════════════════════════');
    sprtLog('Final Results:');
    sprtLog('  Total Games: ' + totalGames);
    sprtLog('  Score: +' + wins + ' -' + losses + ' =' + draws + ' (' + winRate + '%)');
    sprtLog('  Elo Difference: ' + (finalElo >= 0 ? '+' : '') + finalElo.toFixed(1) + ' ±' + finalErr.toFixed(1));
    sprtLog('═══════════════════════════════════════════════════════════════════');
    // Update status line with colored verdict
    sprtStatusEl.textContent = 'Status: ' + verdict;
    let cls = 'sprt-status ';
    if (verdict.startsWith('PASSED')) cls += 'pass';
    else if (verdict.startsWith('FAILED')) cls += 'fail';
    else cls += 'inconclusive';
    sprtStatusEl.className = cls;
    
    sprtRunning = false;
    runSprtBtn.disabled = false;
    stopSprtBtn.disabled = true;
    // Show/enable download games if we have any ICN logs
    const hasGames = gameLogs.length > 0;
    downloadGamesBtn.disabled = !hasGames;
    downloadGamesBtn.style.display = hasGames ? '' : 'none';
}

function stopSprt() {
    stopRequested = true;
    // Immediately terminate all active workers so we don't wait for games to finish
    if (activeSprtWorkers && activeSprtWorkers.length) {
        activeSprtWorkers.forEach(w => {
            try { w.terminate(); } catch (e) {}
        });
        activeSprtWorkers = [];
    }
    sprtRunning = false;
    runSprtBtn.disabled = false;
    stopSprtBtn.disabled = true;
    log('SPRT aborted: workers terminated by user', 'warn');
    // Update status line
    sprtStatusEl.textContent = 'Status: ABORTED';
    sprtStatusEl.className = 'sprt-status inconclusive';
    // If we have any completed games, show a partial results block
    const partialTotal = lastWins + lastLosses + lastDraws;
    if (partialTotal > 0) {
        const partialWinRate = (((lastWins + lastDraws * 0.5) / partialTotal) * 100).toFixed(1);
        sprtLog('');
        sprtLog('═══════════════════════════════════════════════════════════════════');
        sprtLog('Current Results (aborted):');
        sprtLog('  Total Games: ' + partialTotal);
        sprtLog('  Score: +' + lastWins + ' -' + lastLosses + ' =' + lastDraws + ' (' + partialWinRate + '%)');
        if (lastBounds) {
            sprtLog('  Elo Difference: ' + (lastElo >= 0 ? '+' : '') + lastElo.toFixed(1) + ' ±' + lastEloError.toFixed(1));
            sprtLog('  LLR=' + lastLLR.toFixed(2) + ' bounds [' + lastBounds.lower.toFixed(2) + ', ' + lastBounds.upper.toFixed(2) + ']');
        } else {
            sprtLog('  Elo Difference: ' + (lastElo >= 0 ? '+' : '') + lastElo.toFixed(1) + ' ±' + lastEloError.toFixed(1));
        }
        sprtLog('═══════════════════════════════════════════════════════════════════');
    }
    // Allow downloads of games if any finished before abort
    const hasGamesAbort = gameLogs.length > 0;
    downloadGamesBtn.disabled = !hasGamesAbort;
    downloadGamesBtn.style.display = hasGamesAbort ? '' : 'none';
}

function copyLog() {
    const entries = gameLogEl.querySelectorAll('.log-entry');
    let text = '';
    for (let i = 0; i < entries.length; i++) {
        text += entries[i].textContent + '\n';
    }
    navigator.clipboard.writeText(text);
    log('Log copied to clipboard', 'success');
}

function downloadLogs() {
    const entries = gameLogEl.querySelectorAll('.log-entry');
    if (!entries.length) {
        log('No log entries to download yet', 'warn');
        return;
    }
    let text = '';
    for (let i = 0; i < entries.length; i++) {
        text += entries[i].textContent + '\n';
    }
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.href = url;
    a.download = 'sprt-logs-' + ts + '.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function downloadGames() {
    if (!gameLogs.length) {
        log('No games to download yet', 'warn');
        return;
    }
    const blob = new Blob([gameLogs.join('\n\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.href = url;
    a.download = 'sprt-games-' + ts + '.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

runSprtBtn.addEventListener('click', runSprt);
stopSprtBtn.addEventListener('click', stopSprt);
copyLogBtn.addEventListener('click', copyLog);
downloadLogsBtn.addEventListener('click', downloadLogs);
downloadGamesBtn.addEventListener('click', downloadGames);

initWasm();
// Initially hide & disable games download until we have results
downloadGamesBtn.disabled = true;
downloadGamesBtn.style.display = 'none';
