import initOld, * as wasmOld from './pkg-old/hydrochess_wasm.js';
const EngineOld = wasmOld.Engine;
import initNew, * as wasmNew from './pkg-new/hydrochess_wasm.js';
const EngineNew = wasmNew.Engine;
const initThreadPoolNew = wasmNew.initThreadPool;
import { VARIANTS, getVariantData, getAllVariants, getVariantsWithCustomEval } from './variants.js';

let isNewEngineMT = false;

// Map internal engine piece letters to infinitechess.org ICN codes (lowercase for ICN)
function engineLetterToICNCode(letter) {
    const map = {
        'k': 'k', 'q': 'q', 'r': 'r', 'b': 'b', 'n': 'n', 'p': 'p',
        'm': 'am', 'c': 'ch', 'a': 'ar', 'h': 'ha', 'g': 'gu',
        'l': 'ca', 'i': 'gi', 'z': 'ze', 'e': 'ce', 'y': 'rq',
        'd': 'rc', 's': 'nr', 'u': 'hu', 'o': 'ro', 'x': 'ob', 'v': 'vo'
    };
    return map[letter] || letter;
}

function updateMTUI() {
    if (isNewEngineMT) {
        sprtConcurrencyEl.value = "1";
        sprtConcurrencyEl.disabled = true;
        sprtConcurrencyEl.title = "Concurrency locked to 1 game while Multithreading is active.";

        // Dim the label
        const label = sprtConcurrencyEl.previousElementSibling;
        if (label && label.tagName === 'LABEL') {
            label.classList.add('disabled-label');
            label.textContent = 'Concurrency (locked for MT)';
        }

        // Add MT indicator to header if not already there
        if (!document.getElementById('mtBadge')) {
            const h1 = document.querySelector('header h1');
            const badge = document.createElement('span');
            badge.id = 'mtBadge';
            badge.className = 'mt-badge';
            badge.textContent = 'MT';
            h1.appendChild(badge);
        }

        log('Multithreaded engine detected. Concurrency locked to 1 game for stability.', 'info');

        // Set MT defaults: 3+0.03 STC
        sprtTcMode.value = 'standard';
        sprtTimeControlEl.value = '3+0.03';
        updateTcUi(); // Sync the UI labels and disabled states
    }
}

// UI Elements
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const sprtBoundsPreset = document.getElementById('sprtBoundsPreset');
const sprtBoundsMode = document.getElementById('sprtBoundsMode');
const sprtAlphaEl = document.getElementById('sprtAlpha');
const sprtBetaEl = document.getElementById('sprtBeta');
const sprtTcMode = document.getElementById('sprtTcMode');
const sprtTimeControlEl = document.getElementById('sprtTimeControl');
const sprtConcurrencyEl = document.getElementById('sprtConcurrency');
const sprtMinGames = document.getElementById('sprtMinGames');
const sprtMaxGames = document.getElementById('sprtMaxGames');
const sprtMaxMoves = document.getElementById('sprtMaxMoves');
const sprtMaterialThresholdEl = document.getElementById('sprtMaterialAdjudication');
const sprtSearchNoiseEl = document.getElementById('sprtSearchNoise');
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
const downloadGamesTxtBtn = document.getElementById('downloadGames-txt');
const downloadGamesJsonBtn = document.getElementById('downloadGames-json');
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
let currentOldStrength = 3; // 1 = Easy, 2 = Medium, 3 = Hard
let skillPopupTimeout = null;
let activeWorkerResolvers = [];

// SPRT configuration
const CONFIG = {
    elo0: -5,
    elo1: 5,
    alpha: 0.05,
    beta: 0.05,
    boundsPreset: 'all',
    boundsMode: 'gainer',
    timeControl: '10+0.1',
    tcMode: 'smart_mix',
    maxGames: 1000,
    minGames: 500,
    maxMoves: 300,
    concurrency: navigator.hardwareConcurrency || 1,
    materialThreshold: 2000,

    searchNoise: 50,
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
    availableVariants = getAllVariants();
    populateVariantDropdown();
    loadVariantSelection();
}

function populateVariantDropdown() {
    // Get variants with custom eval (these will be disabled by default for SPRT stability)
    const customEvalVariants = new Set(getVariantsWithCustomEval());

    // Variants to disable by default: custom evals AND Abundance
    const defaultsDisabled = new Set(customEvalVariants);
    defaultsDisabled.add('Abundance');

    sprtVariantsEl.innerHTML = '';
    availableVariants.forEach(variant => {
        const option = document.createElement('option');
        option.value = variant;

        // Mark variants with custom eval in the dropdown
        option.textContent = customEvalVariants.has(variant)
            ? `${variant} (custom eval)`
            : variant;

        // Default: deselect variants that are disabled by default
        option.selected = !defaultsDisabled.has(variant);
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

function showSkillPopup() {
    const popup = document.getElementById('skillPopup');
    const levelEl = document.getElementById('skillLevel');

    const names = ["Easy", "Medium", "Hard"];
    levelEl.textContent = names[currentOldStrength - 1];

    popup.classList.add('show');

    if (skillPopupTimeout) clearTimeout(skillPopupTimeout);
    skillPopupTimeout = setTimeout(() => {
        popup.classList.remove('show');
    }, 1500);
}

document.addEventListener('keydown', (e) => {
    if (e.key.toLowerCase() === 'd') {
        const active = document.activeElement;
        const isInput = active.tagName === 'INPUT' || active.tagName === 'SELECT' || active.tagName === 'TEXTAREA';
        if (!isInput && !sprtRunning) {
            currentOldStrength = (currentOldStrength % 3) + 1;
            showSkillPopup();
        }
    }
});

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
        game_rules: variantData.game_rules,
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

    const displayVariantName = variantName || 'Classical';

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
        } else if (endReason === 'stalemate') {
            termination = 'Draw by stalemate';
        } else if (endReason === 'threefold') {
            termination = 'Draw by threefold repetition';
        } else if (endReason === 'fifty_move') {
            termination = 'Draw by fifty-move rule';
        } else if (endReason === 'insufficient_material') {
            termination = 'Draw by insufficient material';
        } else if (endReason === 'max_moves') {
            termination = 'Draw by maximum moves reached';
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
        const raw = line.slice(idx + 1).trim();

        // Handle comments like {[%clk ...]}; extract them so we don't strip spaces inside
        const commentIdx = raw.indexOf('{');
        let movePart = raw;
        let commentPart = '';
        if (commentIdx !== -1) {
            movePart = raw.slice(0, commentIdx);
            commentPart = raw.slice(commentIdx); // Keep spaces inside comment
        }

        // Strip any leading piece info/spaces from move part
        return movePart.replace(/\s+/g, '') + commentPart;
    }).filter(Boolean);

    const movesStr = moves.join('|');
    // Determine promotion ranks and allowed promotions for the variant
    // Default to standard 8 for white, 1 for black, with q,r,b,n allowed
    let whiteRank = '8';
    let blackRank = '1';
    let promotionsAllowed = ['q', 'r', 'b', 'n'];
    let worldBoundsStr = '';
    try {
        const vdata = getVariantData(variantName);
        if (vdata && vdata.game_rules) {
            if (vdata.game_rules.promotion_ranks) {
                const ranks = vdata.game_rules.promotion_ranks;
                if (ranks.white && ranks.white.length > 0) whiteRank = ranks.white[0];
                if (ranks.black && ranks.black.length > 0) blackRank = ranks.black[0];
            }
            if (vdata.game_rules.promotions_allowed && Array.isArray(vdata.game_rules.promotions_allowed)) {
                promotionsAllowed = vdata.game_rules.promotions_allowed;
            }
        }
        // Compute world bounds: if worldBorder is a number, calculate finite bounds
        // Otherwise, use infinite bounds
        if (typeof vdata.worldBorder === 'number' && startPositionStr) {
            // Parse pieces to find min/max coordinates
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (const pieceStr of startPositionStr.split('|')) {
                const parts = pieceStr.split(',');
                if (parts.length < 2) continue;
                // Extract x from first part (after piece code)
                const firstPart = parts[0];
                let xStart = 0;
                while (xStart < firstPart.length && !/[-0-9]/.test(firstPart[xStart])) xStart++;
                const xStr = firstPart.slice(xStart).replace(/\+$/, '');
                const yStr = parts[1].replace(/\+$/, '');
                const x = parseInt(xStr, 10);
                const y = parseInt(yStr, 10);
                if (Number.isFinite(x) && Number.isFinite(y)) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
            if (Number.isFinite(minX)) {
                const pad = vdata.worldBorder;
                worldBoundsStr = `${minX - pad},${maxX + pad},${minY - pad},${maxY + pad}`;
            }
        } else {
            // Infinite bounds for classical-style variants
            worldBoundsStr = '-999999999999999,1000000000000008,-999999999999999,1000000000000008';
        }
    } catch (e) {
        // Use infinite bounds on error
        worldBoundsStr = '-999999999999999,1000000000000008,-999999999999999,1000000000000008';
    }
    // Build promotion token: (whiteRank;promo1,promo2,...|blackRank;promo1,promo2,...)
    // Convert engine piece letters to ICN codes (e.g., 'm' -> 'am')
    const promotionsICN = promotionsAllowed.map(p => engineLetterToICNCode(p)).join(',');
    const promotionRanksToken = `(${whiteRank};${promotionsICN}|${blackRank};${promotionsICN})`;
    return `${headers} ${nextTurn} ${halfmove}/100 ${fullmove} ${promotionRanksToken} ${worldBoundsStr} ${startPositionStr}${movesStr ? ' ' + movesStr : ''}`;
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

function getTcParams(mode, valStr, pairIndex) {
    if (mode === 'smart_mix') {
        // Deterministic pseudo-random based on pairIndex so pairs match
        const r = (pairIndex * 137 + 13) % 100;
        if (r < 75) { // 75% Standard
            const opts = ['10+0.1', '5+0.05', '3+0.03'];
            const pick = opts[pairIndex % opts.length];
            return getTcParams('standard', pick, pairIndex);
        } else { // 25% Fixed Time
            const opts = ['0.1', '0.25', '0.5'];
            const pick = opts[pairIndex % opts.length];
            return getTcParams('fixed_time', pick, pairIndex);
        }
    }

    if (mode === 'fixed_depth') {
        const depth = parseInt(valStr, 10) || 6;
        return {
            timePerMove: 0,
            baseTimeMs: 0,
            incrementMs: 0,
            maxDepth: depth,
            tcString: `depth ${depth}`
        };
    }

    if (mode === 'fixed_time') {
        const sec = parseFloat(valStr) || 1.0;
        const ms = Math.round(sec * 1000);
        return {
            timePerMove: ms,
            baseTimeMs: 0,
            incrementMs: 0,
            maxDepth: null,
            tcString: `fixed ${sec}s`
        };
    }

    // Standard
    const parsed = parseTimeControl(valStr) || { baseMs: 10000, incMs: 100, tcString: '10+0.1' };
    const perMoveEstimate = Math.max(10, Math.round(((parsed.baseSec / 20) + (parsed.incSec / 2)) * 1000));
    return {
        timePerMove: perMoveEstimate, // Used for estimation/fallback
        baseTimeMs: parsed.baseMs,
        incrementMs: parsed.incMs,
        maxDepth: null,
        tcString: parsed.tcString
    };
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

        isNewEngineMT = (typeof initThreadPoolNew === 'function');

        wasmReady = true;
        setStatus('ready', 'WASM loaded and ready');
        runSprtBtn.disabled = false;

        if (isNewEngineMT) {
            updateMTUI();
        } else {
            sprtConcurrencyEl.value = CONFIG.concurrency;
        }

        log('WASM module initialized successfully', 'success');


        const testPos = getStandardPosition();
        const engine = new EngineNew(testPos);
        const move = engine.get_best_move_with_time(100, true);
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

    const capturedIdx = pieces.findIndex(function (p) {
        return p.x === toX && p.y === toY;
    });
    if (capturedIdx !== -1) {
        pieces.splice(capturedIdx, 1);
    }

    const movingPiece = pieces.find(function (p) {
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
    const kings = position.board.pieces.filter(function (p) {
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
        } catch (e) { }
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

    // Immediate UI Reset
    sprtWinsEl.textContent = '0';
    sprtLossesEl.textContent = '0';
    sprtDrawsEl.textContent = '0';
    sprtEloEl.textContent = '-';
    // Reset last stats snapshot
    lastWins = 0;
    lastLosses = 0;
    lastDraws = 0;
    lastElo = 0;
    lastEloError = 0;
    lastLLR = 0;
    lastBounds = null;

    stopRequested = false;
    runSprtBtn.disabled = true;
    stopSprtBtn.disabled = false;
    sprtRunning = true;

    // Read configuration from UI into global CONFIG first (to persist defaults)
    CONFIG.boundsPreset = sprtBoundsPreset.value || 'all';
    CONFIG.boundsMode = sprtBoundsMode.value || 'gainer';
    CONFIG.alpha = parseFloat(sprtAlphaEl.value) || 0.05;
    CONFIG.beta = parseFloat(sprtBetaEl.value) || 0.05;
    CONFIG.tcMode = document.getElementById('sprtTcMode').value;
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
    const valMoves = parseInt(sprtMaxMoves.value, 10);
    CONFIG.maxMoves = (Number.isFinite(valMoves) && valMoves > 0) ? valMoves : Infinity;
    {
        const mtVal = (sprtMaterialThresholdEl.value || '').trim();
        const mt = parseInt(mtVal, 10);
        CONFIG.materialThreshold = (mtVal !== "" && Number.isFinite(mt) && mt >= 0) ? mt : 0;
        const noiseVal = (sprtSearchNoiseEl.value || '').trim();
        const noise = parseInt(noiseVal, 10);
        CONFIG.searchNoise = (noiseVal !== "" && Number.isFinite(noise) && noise >= 0) ? noise : 0;
    }


    // Ensure min/max games are even (for game pairing)
    if (CONFIG.minGames % 2 !== 0) CONFIG.minGames++;
    if (CONFIG.maxGames % 2 !== 0) CONFIG.maxGames++;

    applyBoundsPreset(); // This updates CONFIG.elo0/elo1 based on preset

    // --- SNAPSHOT CONFIGURATION ---
    // Create a local runConfig to isolate this run from future UI/global changes
    const runConfig = { ...CONFIG };

    // Calculate bounds based on the SNAPSHOTTED config
    const bounds = calculateBounds(runConfig.alpha, runConfig.beta);
    // Update global lastBounds so stopSprt can display them correctly if aborted
    lastBounds = bounds;

    // --- SNAPSHOT VARIANTS ---
    // Create a local variant queue sequence to isolate from UI changes.
    // If variants were just selected, variantQueue is up-to-date.
    // We make a copy to lock it in.
    let runVariantQueue = [...variantQueue];
    if (runVariantQueue.length === 0) {
        runVariantQueue.push({ variant: 'Classical', newPlaysWhite: true });
    }
    const uniqueVariants = new Set(runVariantQueue.map(v => v.variant));
    const isMultiVariantRun = uniqueVariants.size > 1;
    let nextVariantIndex = 0;

    function getNextVariantForRun() {
        const result = runVariantQueue[nextVariantIndex];
        nextVariantIndex = (nextVariantIndex + 1) % runVariantQueue.length;
        return result;
    }

    // Validate for Standard mode only, or basic check.
    // If smart_mix, we ignore inputs essentially.
    let displayTcString = runConfig.timeControl;
    let displayPerMoveMs = 0;

    if (runConfig.tcMode === 'standard') {
        const tc = parseTimeControl(runConfig.timeControl);
        if (!tc) {
            log('Invalid time control: ' + runConfig.timeControl + ' (expected base+inc in seconds)', 'error');
            sprtRunning = false;
            runSprtBtn.disabled = false;
            stopSprtBtn.disabled = true;
            return;
        }
        displayTcString = tc.tcString;
        displayPerMoveMs = Math.max(10, Math.round(((tc.baseSec / 20) + (tc.incSec / 2)) * 1000));
    } else if (runConfig.tcMode === 'smart_mix') {
        displayTcString = 'Smart Mix';
        displayPerMoveMs = 'Var';
    } else {
        // fixed time/depth
        const p = getTcParams(runConfig.tcMode, runConfig.timeControl, 0);
        displayTcString = p.tcString;
        displayPerMoveMs = p.timePerMove || 0;
    }
    const timePerMove = displayPerMoveMs; // For info log only

    const maxGames = runConfig.maxGames;
    const maxMovesPerGame = runConfig.maxMoves;

    let wins = 0;
    let losses = 0;
    let draws = 0;
    let llr = 0;
    gameLogs = [];

    // Disable download buttons as logs are cleared
    downloadGamesTxtBtn.disabled = true;
    downloadGamesJsonBtn.disabled = true;

    sprtOutput.innerHTML = '';
    perVariantStats = {};
    clearLog();
    sprtStatusEl.textContent = 'Status: running...';
    sprtStatusEl.className = 'sprt-status';
    const sprtBaseSeed = Date.now() ^ ((Math.random() * 0xFFFFFFFF) | 0);
    log('Starting SPRT: ' + maxGames + ' games (' + (maxGames / 2) + ' pairs), Mode=' + runConfig.tcMode + ', TC=' + displayTcString + ', Seed=' + sprtBaseSeed, 'info');
    sprtLog('SPRT Test Started (noisy opening moves for first 4 ply, paired games)');

    const maxConcurrent = Math.max(1, runConfig.concurrency | 0);
    const workers = [];
    let activeWorkers = 0;
    let nextGameIndex = 0;

    function startWorker(worker, id) {
        const gameIndex = nextGameIndex++;
        if (gameIndex >= maxGames) return false;
        activeWorkers++;

        // Get next variant from the LOCAL SNAPSHOT queue
        const { variant: variantName, newPlaysWhite } = getNextVariantForRun();

        // Games run in pairs: each variant appears twice (both colors)
        const pairIndex = Math.floor(gameIndex / 2);

        // Use runConfig for TC params
        const tcParams = getTcParams(runConfig.tcMode, runConfig.timeControl, pairIndex);

        worker.postMessage({
            type: 'runGame',
            gameIndex,
            timePerMove: tcParams.timePerMove,
            maxMoves: (variantName === 'Abundance' && Number.isFinite(maxMovesPerGame) && isMultiVariantRun) ? Math.min(maxMovesPerGame * 1.5, 1000) : maxMovesPerGame,
            newPlaysWhite,
            materialThreshold: runConfig.materialThreshold,
            searchNoise: runConfig.searchNoise,
            seed: sprtBaseSeed + pairIndex,
            baseTimeMs: tcParams.baseTimeMs,
            incrementMs: tcParams.incrementMs,
            maxDepth: tcParams.maxDepth,
            timeControl: tcParams.tcString,
            variantName, // Add variant to the message
            oldStrength: currentOldStrength,
        });
        return true;
    }

    activeSprtWorkers = workers;
    activeWorkerResolvers = [];

    try {
        await Promise.all(
            Array.from({ length: maxConcurrent }, async (_, i) => {
                const worker = new Worker(new URL('./sprt-worker.js', import.meta.url), { type: 'module' });
                workers.push(worker);

                await new Promise((resolve) => {
                    activeWorkerResolvers.push(resolve);
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
                            // Enable download buttons immediately upon first result
                            downloadGamesTxtBtn.disabled = false;
                            downloadGamesJsonBtn.disabled = false;
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
                            // Use runConfig for ELO
                            llr = calculateLLR(wins, losses, draws, runConfig.elo0, runConfig.elo1);
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

                            sprtLog('Game ' + total + ': ' + result +
                                ' (W:' + wins + ' L:' + losses + ' D:' + draws + ')' +
                                ' Elo≈' + elo.toFixed(1) + '±' + error.toFixed(1) +
                                ' LLR=' + llr.toFixed(2));

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
                                total >= runConfig.minGames && (llr >= bounds.upper || llr <= bounds.lower);
                            const reachedMax = canTerminate && total >= runConfig.maxGames;

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

                            // Check for WASM unreachable or panic
                            const errStr = (msg.error || '').toString();
                            if (errStr.includes("unreachable") || errStr.includes("panic") || errStr.includes("RuntimeError")) {
                                stopRequested = true;
                                log("CRITICAL ERROR: WASM Panic detected in game " + msg.gameIndex, "error");
                                sprtLog("CRITICAL ERROR: WASM Panic detected in game " + msg.gameIndex);
                                sprtLog("Variant: " + (msg.variantName || 'Classical'));
                                sprtLog("Error: " + msg.error);

                                if (msg.log) {
                                    // Generate a full ICN log for the crashed game to make it reproducible
                                    const icnLog = generateICNFromWorkerLog(
                                        msg.log,
                                        msg.gameIndex,
                                        'draw', // Result is unknown due to crash, use draw as placeholder
                                        msg.newPlaysWhite,
                                        'WASM Panic / Crash',
                                        msg.materialThreshold,
                                        msg.timeControl,
                                        msg.variantName
                                    );
                                    sprtLog("--- Game ICN ---");
                                    sprtLog(icnLog);
                                    sprtLog("----------------");
                                }
                                // Force stop all workers immediately
                                workers.forEach(w => w.terminate());
                                activeSprtWorkers = [];
                                sprtRunning = false;
                                // Resolve all workers to unblock Promise.all
                                activeWorkerResolvers.forEach(res => res(undefined));
                                return;
                            }

                            activeWorkers--;

                            const oomLike = errStr.includes('Out of memory') ||
                                errStr.includes('Cannot allocate Wasm memory');
                            if (oomLike) {
                                const stored = loadStoredMaxConcurrency();
                                const current = runConfig.concurrency | 0;
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

                    worker.onerror = (e) => {
                        activeWorkers--;
                        resolve(undefined);
                    };

                    if (!startWorker(worker, i)) {
                        resolve(undefined);
                    }
                });
            })
        );

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
    } finally {
        workers.forEach(w => w.terminate());
        activeSprtWorkers = [];
        sprtRunning = false;
        runSprtBtn.disabled = false;
        stopSprtBtn.disabled = true;
    }
}

function stopSprt() {
    stopRequested = true;
    // Immediately terminate all active workers so we don't wait for games to finish
    if (activeSprtWorkers && activeSprtWorkers.length) {
        activeSprtWorkers.forEach(w => {
            try { w.terminate(); } catch (e) { }
        });
        activeSprtWorkers = [];
    }
    // Unblock any pending promises in runSprt
    if (activeWorkerResolvers && activeWorkerResolvers.length) {
        activeWorkerResolvers.forEach(res => res(undefined));
        activeWorkerResolvers = [];
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
    // Allow download of any completed games
    const hasGamesAbort = gameLogs.length > 0;
    downloadGamesTxtBtn.disabled = !hasGamesAbort;
    downloadGamesJsonBtn.disabled = !hasGamesAbort;
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


function downloadGamesJson() {
    if (!gameLogs.length) {
        log('No games to download yet', 'warn');
        return;
    }

    const jsonOutput = JSON.stringify(gameLogs, null, 2);
    const blob = new Blob([jsonOutput], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.href = url;
    a.download = 'sprt-games-' + ts + '.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    log('Downloaded ' + games.length + ' games as JSON', 'success');
}

runSprtBtn.addEventListener('click', runSprt);
stopSprtBtn.addEventListener('click', stopSprt);
copyLogBtn.addEventListener('click', copyLog);
downloadLogsBtn.addEventListener('click', downloadLogs);
downloadGamesTxtBtn.addEventListener('click', downloadGames);
downloadGamesJsonBtn.addEventListener('click', downloadGamesJson);
sprtVariantsEl.addEventListener('change', updateSelectedVariants);
sprtTcMode.addEventListener('change', updateTcUi);

function updateTcUi() {
    const mode = sprtTcMode.value;
    const tcLabel = document.getElementById('sprtTcLabel');

    // Reset label opacity
    if (tcLabel) tcLabel.classList.remove('disabled-label');

    if (mode === 'smart_mix') {
        sprtTimeControlEl.value = 'Smart Mix';
        sprtTimeControlEl.disabled = true;
        if (tcLabel) {
            tcLabel.classList.add('disabled-label');
            tcLabel.textContent = 'Time Control (Config Ignored)';
        }
    } else {
        sprtTimeControlEl.disabled = false;

        // Update label text based on mode
        if (tcLabel) {
            if (mode === 'standard') tcLabel.textContent = 'Time Control (base+inc)';
            else if (mode === 'fixed_time') tcLabel.textContent = 'Fixed Time per Move (s)';
            else if (mode === 'fixed_depth') tcLabel.textContent = 'Fixed Depth (ply)';
        }

        // If switching away from Smart Mix, restore a logical default
        if (sprtTimeControlEl.value === 'Smart Mix') {
            if (mode === 'fixed_time') sprtTimeControlEl.value = '0.15';
            else if (mode === 'fixed_depth') sprtTimeControlEl.value = '6';
            else sprtTimeControlEl.value = '10+0.1';
        }
    }
}

// Initialize UI state
updateTcUi();

window.addEventListener('beforeunload', (e) => {
    if (sprtRunning) {
        e.preventDefault();
        e.returnValue = ''; // Required for some browsers
    }
});

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