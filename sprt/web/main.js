import initOld, { Engine as EngineOld } from './pkg-old/hydrochess_wasm.js';
import initNew, { Engine as EngineNew } from './pkg-new/hydrochess_wasm.js';
import { VARIANTS, getVariantData } from './variants.js';

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
const sprtVariantsEl = document.getElementById('sprtVariants');
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
// Texel-style samples aggregated from workers for offline tuning
let texelSamples = [];
// Last known stats snapshot (for final/partial result blocks)
let lastWins = 0;
let lastLosses = 0;
let lastDraws = 0;
let lastElo = 0;
let lastEloError = 0;
let lastLLR = 0;
let lastBounds = null;
// Per-variant stats: variantName -> { wins, losses, draws }
let perVariantStats = {};
// Variant management
let availableVariants = [];
let selectedVariants = [];
let variantQueue = [];
let currentVariantIndex = 0;

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

const MAX_CONCURRENCY_STORAGE_KEY = 'sprtMaxSafeConcurrency';

function loadStoredMaxConcurrency() {
    try {
        const raw = localStorage.getItem(MAX_CONCURRENCY_STORAGE_KEY);
        if (!raw) return null;
        const val = parseInt(raw, 10);
        if (!Number.isFinite(val) || val <= 0) return null;
        return val;
    } catch (e) {
        return null;
    }
}

function saveStoredMaxConcurrency(val) {
    try {
        if (!Number.isFinite(val) || val <= 0) return;
        localStorage.setItem(MAX_CONCURRENCY_STORAGE_KEY, String(val));
    } catch (e) {
        // Ignore storage failures
    }
}

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

// Variant management functions
function loadVariants() {
    // Create a temporary worker to get variants
    const worker = new Worker('./sprt-worker.js', { type: 'module' });
    
    worker.onmessage = (e) => {
        if (e.data.type === 'variants') {
            availableVariants = e.data.variants;
            populateVariantDropdown();
            loadVariantSelection();
            worker.terminate();
        }
    };
    
    worker.postMessage({ type: 'getVariants' });
}

function populateVariantDropdown() {
    sprtVariantsEl.innerHTML = '';
    availableVariants.forEach(variant => {
        const option = document.createElement('option');
        option.value = variant;
        option.textContent = variant;
        option.selected = true; // Default all selected
        sprtVariantsEl.appendChild(option);
    });
}

function loadVariantSelection() {
    const saved = localStorage.getItem('sprtSelectedVariants');
    if (saved) {
        try {
            const savedArray = JSON.parse(saved);
            // Clear all selections first
            Array.from(sprtVariantsEl.options).forEach(option => {
                option.selected = false;
            });
            // Apply saved selections
            savedArray.forEach(variantName => {
                const option = Array.from(sprtVariantsEl.options).find(opt => opt.value === variantName);
                if (option) option.selected = true;
            });
        } catch (e) {
            console.warn('Failed to load saved variant selection:', e);
        }
    }
    updateSelectedVariants();
}

function saveVariantSelection() {
    localStorage.setItem('sprtSelectedVariants', JSON.stringify(selectedVariants));
}

function updateSelectedVariants() {
    selectedVariants = Array.from(sprtVariantsEl.selectedOptions).map(option => option.value);
    saveVariantSelection();
    buildVariantQueue();
}

function buildVariantQueue() {
    variantQueue = [];
    // Build queue with each variant appearing twice (for both colors)
    selectedVariants.forEach(variant => {
        variantQueue.push({ variant, newPlaysWhite: true });
        variantQueue.push({ variant, newPlaysWhite: false });
    });
    currentVariantIndex = 0;
}

function getNextVariant() {
    if (variantQueue.length === 0) {
        return { variant: 'Classical', newPlaysWhite: true };
    }
    
    const result = variantQueue[currentVariantIndex];
    currentVariantIndex = (currentVariantIndex + 1) % variantQueue.length;
    return result;
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
    // Use Classical variant definition from variants.js for sanity test
    const variantData = getVariantData('Classical');
    const pieces = [];
    const special_rights = [];

    for (const pieceStr of (variantData.position || '').split('|')) {
        if (!pieceStr) continue;
        const parts = pieceStr.split(',');
        if (parts.length !== 2) continue;

        const pieceInfo = parts[0];
        const yStr = parts[1];
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
        const player = isWhite ? 'w' : 'b';

        const hasSpecial = xRaw.endsWith('+') || yStr.endsWith('+');
        const x = xRaw.endsWith('+') ? xRaw.slice(0, -1) : xRaw;
        const y = yStr.endsWith('+') ? yStr.slice(0, -1) : yStr;
        if (hasSpecial) special_rights.push(x + ',' + y);

        const codeLower = pieceCode.toLowerCase();
        let piece_type;
        switch (codeLower) {
            case 'k': piece_type = 'k'; break;
            case 'q': piece_type = 'q'; break;
            case 'r': piece_type = 'r'; break;
            case 'b': piece_type = 'b'; break;
            case 'n': piece_type = 'n'; break;
            case 'p': piece_type = 'p'; break;
            case 'am': piece_type = 'm'; break;
            case 'ch': piece_type = 'c'; break;
            case 'ar': piece_type = 'a'; break;
            case 'ha': piece_type = 'h'; break;
            case 'gu': piece_type = 'g'; break;
            case 'ca': piece_type = 'l'; break;
            case 'gi': piece_type = 'i'; break;
            case 'ze': piece_type = 'z'; break;
            case 'ce': piece_type = 'e'; break;
            case 'rq': piece_type = 'y'; break;
            case 'rc': piece_type = 'd'; break;
            case 'nr': piece_type = 's'; break;
            case 'hu': piece_type = 'u'; break;
            case 'ro': piece_type = 'o'; break;
            case 'ob': piece_type = 'x'; break;
            case 'vo': piece_type = 'v'; break;
            default: continue;
        }

        pieces.push({ x, y, piece_type, player });
    }

    return {
        board: { pieces },
        turn: 'w',
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
function generateICNFromWorkerLog(workerLog, gameIndex, result, newPlaysWhite, endReason, materialThreshold, timeControl, variantName = 'Classical') {
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

    const displayVariantName = (variantName || 'Classical').replace(/_/g, ' ');

    const headerList = [
        `[Event "SPRT Test Game ${gameIndex}"]`,
        `[Site "https://www.infinitechess.org/"]`,
        `[Variant "${displayVariantName}"]`,
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
        } else if (endReason === 'engine_failure') {
            termination = 'Loss on engine failure (no move returned)';
        } else if (endReason === 'horde_elimination') {
            termination = `Win by capturing all White pieces in ${displayVariantName}`;
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
    // ICN encodes the START position for the selected variant, with
    // White to move, zero halfmove clock, fullmove number 1.
    const nextTurn = 'w';
    const fullmove = 1;
    const halfmove = 0;

    // Variant-specific starting position from variants.js
    let startPositionStr = null;
    try {
        const vdata = getVariantData(variantName);
        if (vdata && typeof vdata.position === 'string' && vdata.position.length > 0) {
            startPositionStr = vdata.position;
        }
    } catch (e) {
        // Fallback to Classical if variant missing for some reason
        if (VARIANTS.Classical && typeof VARIANTS.Classical.position === 'string') {
            startPositionStr = VARIANTS.Classical.position;
        }
    }

    if (!startPositionStr) {
        startPositionStr = '';
    }

    // Moves string: parse worker log lines of form "W: x,y>u,v".
    const moves = lines.map((line) => {
        const idx = line.indexOf(':');
        if (idx === -1) return '';
        const rest = line.slice(idx + 1).trim(); // "x,y>u,v" or with promotion
        // Strip any leading piece info if ever added; currently we only have coords
        return rest.replace(/\s+/g, '');
    }).filter(Boolean);

    const movesStr = moves.join('|');
    return `${headers} ${nextTurn} ${halfmove}/100 ${fullmove} (8|1) ${startPositionStr}${movesStr ? ' ' + movesStr : ''}`;
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

async function detectMaxConcurrency(maxCap = 64) {
    // If we have a stored max from a previous run, reuse it and avoid
    // probing again.
    const stored = loadStoredMaxConcurrency();
    if (stored && stored > 0) {
        log('Using stored max safe concurrency from previous run: ' + stored, 'info');
        sprtLog('Using stored max safe concurrency: ' + stored);
        return stored;
    }

    // Otherwise, probe how many workers we can create that successfully
    // initialize WASM before running out of memory. We incrementally spawn
    // workers that send a lightweight 'probe' message (handled in
    // sprt-worker.js) and stop at the first failure.
    let lastOk = 0;
    for (let n = 1; n <= maxCap; n++) {
        const worker = new Worker(new URL('./sprt-worker.js', import.meta.url), { type: 'module' });
        const ok = await new Promise((resolve) => {
            let settled = false;
            worker.onmessage = (e) => {
                const msg = e.data;
                if (msg && msg.type === 'probeResult') {
                    settled = true;
                    resolve(!!msg.ok);
                }
            };
            worker.onerror = () => {
                if (!settled) {
                    settled = true;
                    resolve(false);
                }
            };
            try {
                worker.postMessage({ type: 'probe' });
            } catch (e) {
                resolve(false);
            }
        });
        try {
            worker.terminate();
        } catch (e) {}
        if (!ok) {
            break;
        }
        lastOk = n;
    }

    if (lastOk <= 0) {
        lastOk = 1;
    }

    saveStoredMaxConcurrency(lastOk);
    log('Detected max safe concurrency: ' + lastOk, 'info');
    sprtLog('Max safe concurrency detected: ' + lastOk);
    return lastOk;
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

    const rawConcurrency = (sprtConcurrencyEl.value || '').toString().trim();
    if (rawConcurrency.toLowerCase() === 'max') {
        log('Concurrency set to "max" – probing for safe limit...', 'info');
        sprtLog('Concurrency set to "max" – probing for maximum safe workers...');
        CONFIG.concurrency = await detectMaxConcurrency(64);
    } else {
        CONFIG.concurrency = parseInt(rawConcurrency, 10) || 1;
    }
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
        log('Invalid time control: ' + CONFIG.timeControl + ' (expected base+inc in seconds)', 'error');
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
    perVariantStats = {};
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
        
        // Get next variant from the cycling queue
        const { variant: variantName, newPlaysWhite } = getNextVariant();
        
        // Games run in pairs: each variant appears twice (both colors)
        const pairIndex = Math.floor(gameIndex / 2);
        // Only use opening moves for Classical variant to avoid errors with custom positions
        const openingMove = variantName === 'Classical' ? getOpeningForPair(pairIndex) : null;
        
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
            variantName, // Add variant to the message
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
                        if (Array.isArray(msg.samples) && msg.samples.length) {
                            texelSamples.push(...msg.samples);
                        }
                        const icnLog = generateICNFromWorkerLog(
                            msg.log,
                            msg.gameIndex,
                            result,
                            msg.newPlaysWhite,
                            msg.reason,
                            msg.materialThreshold,
                            msg.timeControl,
                            msg.variantName, // Add variant to ICN log
                        );
                        gameLogs.push(icnLog);
                        // Global results
                        if (result === 'win') wins++;
                        else if (result === 'loss') losses++;
                        else draws++;

                        // Per-variant results
                        const vName = msg.variantName || 'Classical';
                        if (!perVariantStats[vName]) {
                            perVariantStats[vName] = { wins: 0, losses: 0, draws: 0 };
                        }
                        if (result === 'win') perVariantStats[vName].wins++;
                        else if (result === 'loss') perVariantStats[vName].losses++;
                        else perVariantStats[vName].draws++;

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

                        // If this looks like a WebAssembly out-of-memory error,
                        // dynamically lower the stored max safe concurrency so
                        // future runs with "max" avoid this level.
                        const errStr = (msg.error || '').toString();
                        const oomLike = errStr.includes('Out of memory') ||
                            errStr.includes('Cannot allocate Wasm memory');
                        if (oomLike) {
                            const stored = loadStoredMaxConcurrency();
                            const current = CONFIG.concurrency | 0;
                            const proposed = Math.max(1, Math.min(current - 1, stored || current));
                            if (proposed < (stored || Infinity)) {
                                saveStoredMaxConcurrency(proposed);
                                log('Detected WASM OOM at concurrency ' + current + ', lowering stored max to ' + proposed, 'warn');
                                sprtLog('WASM out-of-memory detected at concurrency ' + current + ' – new stored max: ' + proposed);
                            }
                        }

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
    sprtLog('');
    sprtLog('Per-Variant Breakdown:');
    const variantNames = Object.keys(perVariantStats).sort();
    variantNames.forEach((name) => {
        const s = perVariantStats[name];
        const { elo, error } = estimateElo(s.wins, s.losses, s.draws);
        const vtTotal = s.wins + s.losses + s.draws;
        const vtScore = vtTotal > 0 ? (((s.wins + s.draws * 0.5) / vtTotal) * 100).toFixed(1) : '0.0';
        sprtLog('  [' + name + ']: +' + s.wins + ' -' + s.losses + ' =' + s.draws + ' (' + vtScore + '%), Elo≈ ' + (elo >= 0 ? '+' : '') + elo.toFixed(1) + ' ±' + error.toFixed(1));
    });
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
        sprtLog('');
        sprtLog('Per-Variant Breakdown (partial):');
        const variantNames = Object.keys(perVariantStats).sort();
        variantNames.forEach((name) => {
            const s = perVariantStats[name];
            const { elo, error } = estimateElo(s.wins, s.losses, s.draws);
            const vtTotal = s.wins + s.losses + s.draws;
            const vtScore = vtTotal > 0 ? (((s.wins + s.draws * 0.5) / vtTotal) * 100).toFixed(1) : '0.0';
            sprtLog('  [' + name + ']: +' + s.wins + ' -' + s.losses + ' =' + s.draws + ' (' + vtScore + '%), Elo≈ ' + (elo >= 0 ? '+' : '') + elo.toFixed(1) + ' ±' + error.toFixed(1));
        });
        sprtLog('═══════════════════════════════════════════════════════════════════');
        if (lastBounds) {
            sprtLog('  Elo Difference: ' + (lastElo >= 0 ? '+' : '') + lastElo.toFixed(1) + ' ±' + lastEloError.toFixed(1));
            sprtLog('  LLR=' + lastLLR.toFixed(2) + ' bounds [' + lastBounds.lower.toFixed(2) + ', ' + lastBounds.upper.toFixed(2) + ']');
        } else {
            sprtLog('  Elo Difference: ' + (lastElo >= 0 ? '+' : '') + lastElo.toFixed(1) + ' ±' + lastEloError.toFixed(1));
        }
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
sprtVariantsEl.addEventListener('change', updateSelectedVariants);

// Initialize variant loading
loadVariants();

// Minimal hooks for headless tuning via Puppeteer. These do not change
// UI behavior but allow a Node script to inspect results and readiness.
window.__sprt_export_games = () => gameLogs.slice();
window.__sprt_export_samples = (offset = 0) => {
    const start = Number.isFinite(offset) && offset >= 0 ? offset : 0;
    return texelSamples.slice(start);
};
window.__sprt_is_ready = () => wasmReady;
window.__sprt_status = () => ({
	running: sprtRunning,
	wins: lastWins,
	losses: lastLosses,
	draws: lastDraws,
});

window.__sprt_compute_features = async (rawSamples) => {
    const samples = Array.isArray(rawSamples) ? rawSamples : [];
    const results = [];
    for (const s of samples) {
        if (!s || !Array.isArray(s.move_history) || !s.result_token || !s.side_to_move) {
            continue;
        }
        const side = s.side_to_move === 'b' ? 'b' : 'w';
        let result = 0.5;
        if (s.result_token === '1/2-1/2') {
            result = 0.5;
        } else if (s.result_token === '1-0') {
            result = side === 'w' ? 1.0 : 0.0;
        } else if (s.result_token === '0-1') {
            result = side === 'w' ? 0.0 : 1.0;
        }

        const base = getStandardPosition();
        base.move_history = s.move_history.map((m) => ({
            from: m.from,
            to: m.to,
            promotion: m.promotion || null,
        }));

        let evalWithFeatures;
        try {
            const engine = new EngineNew(base);
            evalWithFeatures = engine.evaluate_with_features();
            engine.free();
        } catch (e) {
            continue;
        }

        if (!evalWithFeatures || typeof evalWithFeatures.eval !== 'number' || !evalWithFeatures.features) {
            continue;
        }

        const positionSnapshot = s.position || null;

        results.push({
            result,
            side_to_move: side,
            ply_index: typeof s.ply_index === 'number' ? s.ply_index : null,
            piece_count: typeof s.piece_count === 'number' ? s.piece_count : null,
            features: evalWithFeatures.features,
            // Optional debugging/analysis fields: exact sampled position and
            // the move history used to reach it.
            position: positionSnapshot,
            move_history: Array.isArray(s.move_history) ? s.move_history : null,
        });
    }
    return results;
};

initWasm();
// Initially hide & disable games download until we have results
downloadGamesBtn.disabled = true;
downloadGamesBtn.style.display = 'none';
