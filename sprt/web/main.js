import initOld, * as wasmOld from './pkg-old/apeiron.js';
const EngineOld = wasmOld.Engine;
import initNew, * as wasmNew from './pkg-new/apeiron.js';
const EngineNew = wasmNew.Engine;
// Either build may be MT (both default to the threaded build now). A build exposes
// initThreadPool only when compiled with Lazy SMP, so its presence is the MT probe.
const initThreadPoolOld = wasmOld.initThreadPool;
const initThreadPoolNew = wasmNew.initThreadPool;
import { getVariantData, getAllVariants, generateSetupICN, engineLetterToICNCode, getVariantsWithCustomEval } from './variants.js';

let isOldEngineMT = false;
let isNewEngineMT = false;

/** Highest concurrency that keeps games x threads-per-game within the machine's cores.
 * Engines alternate within a game (only one searches at a time), so a game's peak demand
 * is the larger of the two engines' pool sizes. */
function mtConcurrencyCap(oldThreads, newThreads) {
    const hc = navigator.hardwareConcurrency || 4;
    const perGame = Math.max(1, oldThreads, newThreads);
    return Math.max(1, Math.floor(hc / perGame));
}

/** Reads the current per-engine thread selections (1 for a single-threaded / ST build). */
function currentThreadCounts() {
    const o = isOldEngineMT ? Math.min(16, Math.max(1, parseInt(sprtMtThreadsOldEl.value, 10) || 1)) : 1;
    const n = isNewEngineMT ? Math.min(16, Math.max(1, parseInt(sprtMtThreadsNewEl.value, 10) || 1)) : 1;
    return { o, n };
}

/** Re-derives ONLY the concurrency cap from the current per-engine thread counts. Called when
 * the user edits a thread input — it must not touch time control or any other setting. */
function recomputeConcurrencyCap() {
    if (!isOldEngineMT && !isNewEngineMT) return;
    const { o, n } = currentThreadCounts();
    sprtConcurrencyEl.value = String(mtConcurrencyCap(o, n));
}

/** One-time MT setup on load: reveals the per-engine thread inputs, seeds the concurrency cap,
 * adds the badge, and picks a fast default time control. Not called again on thread edits. */
function updateMTUI() {
    // Show/enable each engine's thread input only when that build supports threads.
    sprtMtThreadsOldEl.closest('.form-group').style.display = isOldEngineMT ? '' : 'none';
    sprtMtThreadsNewEl.closest('.form-group').style.display = isNewEngineMT ? '' : 'none';

    if (isOldEngineMT || isNewEngineMT) {
        const { o, n } = currentThreadCounts();
        const cap = mtConcurrencyCap(o, n);
        sprtConcurrencyEl.value = String(cap);
        sprtConcurrencyEl.title = 'Auto-capped so (concurrent games) x (max threads/game) fits your ' + (navigator.hardwareConcurrency || '?') + ' cores.';

        if (!document.getElementById('mtBadge')) {
            const h1 = document.querySelector('header h1');
            const badge = document.createElement('span');
            badge.id = 'mtBadge';
            badge.className = 'mt-badge';
            badge.textContent = 'MT';
            h1.appendChild(badge);
            mtStatusEl.textContent = 'MT Enabled';
        }
        log('MT-capable build(s) detected (old=' + (isOldEngineMT ? o : 'ST') + ', new=' + (isNewEngineMT ? n : 'ST') + '). Concurrency capped at ' + cap + ' games.', 'info');

        // Default to a fast STC for MT experiments (once, at load — not on every thread edit).
        sprtTcMode.value = 'standard';
        sprtTimeControlEl.value = '3+0.03';
        updateTcUi();
    }
}

// UI Elements
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const mtStatusEl = document.getElementById('mtStatus');
const sprtBoundsPreset = document.getElementById('sprtBoundsPreset');
const sprtBoundsMode = document.getElementById('sprtBoundsMode');
const sprtAlphaEl = document.getElementById('sprtAlpha');
const sprtBetaEl = document.getElementById('sprtBeta');
const sprtTcMode = document.getElementById('sprtTcMode');
const sprtTimeControlEl = document.getElementById('sprtTimeControl');
const sprtConcurrencyEl = document.getElementById('sprtConcurrency');
const sprtMtThreadsOldEl = document.getElementById('sprtMtThreadsOld');
const sprtMtThreadsNewEl = document.getElementById('sprtMtThreadsNew');
const sprtHashOldEl = document.getElementById('sprtHashOld');
const sprtHashNewEl = document.getElementById('sprtHashNew');
// Changing either engine's thread count re-derives ONLY the concurrency cap — nothing else.
sprtMtThreadsOldEl.addEventListener('change', recomputeConcurrencyCap);
sprtMtThreadsNewEl.addEventListener('change', recomputeConcurrencyCap);
const sprtMinGames = document.getElementById('sprtMinGames');
const sprtMaxGames = document.getElementById('sprtMaxGames');
const sprtMaxMoves = document.getElementById('sprtMaxMoves');
const sprtMaterialThresholdEl = document.getElementById('sprtMaterialAdjudication');
const sprtSearchNoiseEl = document.getElementById('sprtSearchNoise');
const sprtVariantPresetsEl = document.getElementById('sprtVariantPresets');
const sprtVariantsEl = document.getElementById('sprtVariants');
const runSprtBtn = document.getElementById('runSprt');
const stopSprtBtn = document.getElementById('stopSprt');
const sprtGamesEl = document.getElementById('sprtGames');
const sprtPairsEl = document.getElementById('sprtPairs');
const sprtWinsEl = document.getElementById('sprtWins');
const sprtLossesEl = document.getElementById('sprtLosses');
const sprtDrawsEl = document.getElementById('sprtDraws');
const sprtEloEl = document.getElementById('sprtElo');
const sprtEloErrorEl = document.getElementById('sprtEloError');
const sprtLLRContainer = document.getElementById('sprtLLRContainer');
const sprtLLREl = document.getElementById('sprtLLR');
const sprtLLRBoundsEl = document.getElementById('sprtLLRBounds');
const sprtLOSContainer = document.getElementById('sprtLOSContainer');
const sprtLOSEl = document.getElementById('sprtLOS');
const sprtOutput = document.getElementById('sprtOutput');
const gameLogEl = document.getElementById('gameLog');
const copyLogBtn = document.getElementById('copyLog');
const downloadLogsBtn = document.getElementById('downloadLogs');
const downloadGamesTxtBtn = document.getElementById('downloadGames-txt');
const downloadGamesJsonBtn = document.getElementById('downloadGames-json');
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
let lastNelo = 0;
let lastNeloError = 0;
let lastLLR = 0;
let lastBounds = null;
let lastTimeoutLosses = 0;
// Pentanomial pair accounting (drives the SPRT decision, matching src/bin/sprt.rs).
let pentaCounts = {};
// NEW-perspective result per gameIndex, held until its pair partner completes.
let pendingPairResults = {};
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
    elo0: 0,
    elo1: 2,
    // SPRT model: 'normalized' (nElo bounds, draw-rate/TC independent) or 'logistic'.
    model: 'normalized',
    alpha: 0.05,
    beta: 0.05,
    boundsPreset: 'stockfish_stc',
    boundsMode: 'gainer',
    timeControl: '10+0.1',
    tcMode: 'smart_mix',
    maxGames: 1000,
    minGames: 500,
    maxMoves: 300,
    concurrency: navigator.hardwareConcurrency || 1,
    mtThreadsOld: 1,
    mtThreadsNew: 1,
    hashOldMb: 16,
    hashNewMb: 16,
    materialThreshold: 0,

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
const VARIANT_PRESETS = {
    Default: [ // base-eval standard variants (no multi-king, AllPieces, and no Abundance)
        'Classical',
        'Confined_Classical',
        'Classical_Plus',
        'CoaIP',
        'CoaIP_HO',
        'CoaIP_RO',
        'CoaIP_NO',
        'Palace',
        'Pawndard',
        'Core',
        'Standarch',
        'Space_Classic',
        'Space',
        'Knightline',
        'Scattered_Leapers',
    ],
    All: true, // every variant implemented here
    Site: [ // variants on the public site except Abundance and Showcases
        'Classical',
        'Confined_Classical',
        'Classical_Plus',
        'CoaIP',
        'CoaIP_HO',
        'CoaIP_RO',
        'CoaIP_NO',
        'Palace',
        'Pawndard',
        'Core',
        'Standarch',
        'Space_Classic',
        'Space',
        'Pawn_Horde',
        'Knightline',
        'Obstocean',
        'Chess',
    ],
    Base_full: [ // all base-eval variants + multi-king + AllPieces
        'Classical',
        'Confined_Classical',
        'Classical_Plus',
        'CoaIP',
        'CoaIP_HO',
        'CoaIP_RO',
        'CoaIP_NO',
        'Palace',
        'Pawndard',
        'Core',
        'Standarch',
        'Space_Classic',
        'Space',
        'Knightline',
        'Scattered_Leapers',
        'Double_King_Classical',
        'Double_King_Chess',
        'Triple_King_Maze',
        'All_Pieces_Classical',
    ],
    Multi_king: [ // variants with 2+ kings per side
        'Double_King_Classical',
        'Double_King_Chess',
        'Triple_King_Maze',
    ],
    Coaip: [ // "Chess on an Infinite Plane" family
        'CoaIP',
        'CoaIP_HO',
        'CoaIP_RO',
        'CoaIP_NO',
    ],
};

function loadVariants() {
    availableVariants = getAllVariants();
    populateVariantPresets();
    populateVariantDropdown();
    loadVariantSelection('Saved');
}

function populateVariantPresets() {
    Object.keys(VARIANT_PRESETS).forEach(preset => {
        const btn = document.createElement('button');
        btn.addEventListener('click', () => loadVariantSelection(preset));

        // Fill in the content and styles
        if (preset === 'Default') {
            btn.textContent = 'Default (base only)';
            btn.classList.add('btn', 'btn-sm');
        } else {
            btn.textContent = preset.replace(/_/g, ' ');
            btn.classList.add('btn', 'btn-secondary', 'btn-sm');
        }

        sprtVariantPresetsEl.appendChild(btn);
    });
}

function populateVariantDropdown() {
    // Get variants with custom eval (these will be disabled by default for SPRT stability)
    const customEvalVariants = new Set(getVariantsWithCustomEval());

    sprtVariantsEl.innerHTML = '';
    availableVariants.forEach(variant => {
        const option = document.createElement('option');
        option.value = variant;

        // Mark variants with custom eval in the dropdown
        option.textContent = customEvalVariants.has(variant)
            ? `${variant.replace(/_/g, ' ')} (custom eval)`
            : variant.replace(/_/g, ' ');

        sprtVariantsEl.appendChild(option);
    });
}

function loadVariantSelection(preset) {
    switch (preset) {
        case 'All':
            Array.from(sprtVariantsEl.options).forEach(option => {
                option.selected = true;
            });
            break;

        case 'Saved':
            // Selects the previously saved variant selection from localStorage,
            // or default variants if nothing is saved.
            const saved = localStorage.getItem('sprtSelectedVariants');
            if (!saved) {
                loadVariantSelection('Default');
                return;
            }

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
            break;

        default:
            // Load the variants in the specified preset
            const variants = VARIANT_PRESETS[preset];
            if (!variants) {
                console.warn('Preset is not there:', preset);
                return;
            }
            // Clear all selections first
            Array.from(sprtVariantsEl.options).forEach(option => {
                option.selected = false;
            });
            // Select the variants in the preset
            variants.forEach(variantName => {
                const option = Array.from(sprtVariantsEl.options).find(opt => opt.value === variantName);
                if (option) option.selected = true;
            });
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

    const whiteEngine = newPlaysWhite ? 'Apeiron New' : 'Apeiron Old';
    const blackEngine = newPlaysWhite ? 'Apeiron Old' : 'Apeiron New';

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
        } else if (endReason === 'allroyalscaptured') {
            termination = 'All royals captured';
        } else if (endReason === 'royalcapture') {
            termination = 'Royal capture';
        } else if (endReason === 'checkmate') {
            termination = 'Checkmate';
        } else if (endReason === 'stalemate') {
            termination = 'Stalemate';
        } else if (endReason === 'threefold') {
            termination = 'Threefold repetition';
        } else if (endReason === 'fifty_move') {
            termination = '50-move rule';
        } else if (endReason === 'insufficient_material') {
            termination = 'Insufficient material';
        } else if (endReason === 'max_moves') {
            termination = 'Maximum moves reached';
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
        startPositionStr = getVariantData('Classical').position;
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
    const moveHistoryList = moves.map(m => {
        const promoIdx = m.indexOf('=');
        let move = m;
        let promotion = null;
        if (promoIdx !== -1) {
            move = m.slice(0, promoIdx);
            promotion = m.slice(promoIdx + 1);
        }
        const parts = move.split('>');
        return { from: parts[0], to: parts[1], promotion };
    });

    const setupIcn = generateSetupICN(variantName, nextTurn, halfmove, fullmove, moveHistoryList);
    return `${headers} ${setupIcn}`;
}

function log(message, type) {
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = '<span class="log-time">[' + time + ']</span> <span class="log-' + type + '">' + message + '</span>';
    gameLogEl.appendChild(entry);
    gameLogEl.scrollTop = gameLogEl.scrollHeight;
}

function sprtLog(message, type) {
    const entry = document.createElement('div');
    entry.textContent = message;
    if (type) entry.className = 'log-' + type;
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

function scoreToElo(s) {
    s = Math.max(1e-9, Math.min(1 - 1e-9, s)); // clamp between 0 and 1
    return -400 * Math.log10(1 / s - 1);
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

// ── Pentanomial GSPRT (ported 1:1 from src/bin/sprt.rs, matching fastchess/Fishtest) ──
// Games are played in color-balanced pairs from the same opening line; a pair's two
// NEW-perspective results collapse to five buckets (LL/LD/WL·DD/WD/WW). This cancels
// within-pair variance so the SPRT concludes in fewer games than the trinomial model.

class PentaCounts {
    constructor(ll = 0, ld = 0, wl = 0, dd = 0, wd = 0, ww = 0) {
        Object.assign(this, {ll, ld, wl, dd, wd, ww});
    }

    // Bucket a completed pair from two NEW-perspective results ('win'|'loss'|'draw').
    addPair(a, b) {
        let w = 0, d = 0, l = 0;
        for (const r of [a, b]) {
            if (r === 'win') w++;
            else if (r === 'draw') d++;
            else l++;
        }
        if (w === 2) this.ww++;
        else if (w === 1 && d === 1) this.wd++;
        else if (w === 1 && l === 1) this.wl++;
        else if (d === 2) this.dd++;
        else if (l === 1 && d === 1) this.ld++;
        else this.ll++;
    }

    get totalPairs() {
        return this.ww + this.wd + this.wl + this.dd + this.ld + this.ll;
    }

    get score() {
        return (this.ww + 0.75 * this.wd + 0.5 * (this.wl + this.dd) + 0.25 * this.ld) / this.totalPairs;
    }

    get variance() {
        let score = this.score;
        return (
            this.ww * (1 - score) ** 2 +
            this.wd * (0.75 - score) ** 2 +
            (this.wl + this.dd) * (0.5 - score) ** 2 +
            this.ld * (0.25 - score) ** 2 +
            this.ll * (0 - score) ** 2
        ) / this.totalPairs;
    }

    get displayText() {
        return `${this.ll}, ${this.ld}, ${this.wl + this.dd}, ${this.wd}, ${this.ww}`;
    }
}

// fastchess regularize: a zero bucket becomes 1e-3 so log-likelihoods stay finite.
function regularize(v) {
    return v === 0 ? 1e-3 : v;
}

// ITP root-finder (Oliveira & Takahashi 2020), ported from fastchess itp().
function itp(f, a, b, fA, fB, k1, k2, n0, epsilon) {
    if (fA > 0) {
        [a, b] = [b, a];
        [fA, fB] = [fB, fA];
    }
    const nHalf = Math.ceil(Math.log2(Math.abs(b - a) / (2 * epsilon)));
    const nMax = nHalf + n0;
    let i = 0;
    while (Math.abs(b - a) > 2 * epsilon) {
        const xHalf = (a + b) / 2;
        const r = epsilon * Math.pow(2, nMax - i) - (b - a) / 2;
        const delta = k1 * Math.pow(b - a, k2);
        const xF = (fB * a - fA * b) / (fB - fA);
        const sigma = (xHalf - xF) / Math.abs(xHalf - xF);
        const xT = delta <= Math.abs(xHalf - xF) ? xF + sigma * delta : xHalf;
        const xItp = Math.abs(xT - xHalf) <= r ? xT : xHalf - sigma * r;
        const fItp = f(xItp);
        if (fItp === 0) { a = xItp; b = xItp; }
        else if (fItp < 0) { a = xItp; fA = fItp; }
        else { b = xItp; fB = fItp; }
        i++;
    }
    return (a + b) / 2;
}

// MLE outcome distribution constrained to expected score s (fastchess getLLR_logistic inner mle).
function mleLogistic(scores, probs, s) {
    const n = scores.length;
    const thetaEpsilon = 1e-3;
    const minTheta = -1 / (scores[n - 1] - s);
    const maxTheta = -1 / (scores[0] - s);
    const theta = itp(
        (x) => {
            let result = 0;
            for (let i = 0; i < n; i++) {
                const ai = scores[i];
                result += probs[i] * (ai - s) / (1 + x * (ai - s));
            }
            return result;
        },
        minTheta, maxTheta, Infinity, -Infinity, 0.1, 2.0, 0.99, thetaEpsilon,
    );
    return scores.map((ai, i) => probs[i] / (1 + theta * (ai - s)));
}

function llrLogistic(total, scores, probs, s0, s1) {
    const p0 = mleLogistic(scores, probs, s0);
    const p1 = mleLogistic(scores, probs, s1);
    let acc = 0;
    for (let i = 0; i < scores.length; i++) {
        acc += probs[i] * (Math.log(p1[i]) - Math.log(p0[i]));
    }
    return total * acc;
}

function meanArr(x, p) {
    let result = 0;
    for (let i = 0; i < x.length; i++) result += x[i] * p[i];
    return result;
}

function meanAndVariance(x, p) {
    const mu = meanArr(x, p);
    let variance = 0;
    for (let i = 0; i < x.length; i++) variance += p[i] * (x[i] - mu) * (x[i] - mu);
    return [mu, variance];
}

// MLE distribution for the normalized model (fastchess getLLR_normalized inner mle).
function mleNormalized(scores, probs, muRef, tStar) {
    const n = scores.length;
    const thetaEpsilon = 1e-7;
    const mleEpsilon = 1e-4;
    let p = new Array(n).fill(1 / n);

    for (let iter = 0; iter < 10; iter++) {
        const [mu, variance] = meanAndVariance(scores, p);
        const sigma = Math.sqrt(variance);
        const phi = scores.map((ai) =>
            ai - muRef - 0.5 * tStar * sigma * (1 + ((ai - mu) / sigma) * ((ai - mu) / sigma)));
        const u = Math.min(...phi);
        const v = Math.max(...phi);
        const minTheta = -1 / v;
        const maxTheta = -1 / u;
        const theta = itp(
            (x) => {
                let result = 0;
                for (let i = 0; i < n; i++) result += probs[i] * phi[i] / (1 + x * phi[i]);
                return result;
            },
            minTheta, maxTheta, Infinity, -Infinity, 0.1, 2.0, 0.99, thetaEpsilon);
        let maxDiff = 0;
        for (let i = 0; i < n; i++) {
            const newp = probs[i] / (1 + theta * phi[i]);
            maxDiff = Math.max(maxDiff, Math.abs(newp - p[i]));
            p[i] = newp;
        }
        if (maxDiff < mleEpsilon) break;
    }
    return p;
}

function llrNormalized(total, scores, probs, t0, t1) {
    const p0 = mleNormalized(scores, probs, 0.5, t0);
    const p1 = mleNormalized(scores, probs, 0.5, t1);
    let acc = 0;
    for (let i = 0; i < scores.length; i++) {
        acc += probs[i] * (Math.log(p1[i]) - Math.log(p0[i]));
    }
    return total * acc;
}

// Pentanomial LLR — the SPRT decision statistic (Fishtest model). model: 'normalized' | 'logistic'.
function calculatePentanomialLLR(p, elo0, elo1, model) {
    if (p.totalPairs === 0) return 0;
    const ll = regularize(p.ll);
    const ld = regularize(p.ld);
    const wlDd = regularize(p.dd + p.wl);
    const wd = regularize(p.wd);
    const ww = regularize(p.ww);
    const total = ww + wd + wlDd + ld + ll;
    const probs = [ll / total, ld / total, wlDd / total, wd / total, ww / total];
    const scores = [0.0, 0.25, 0.5, 0.75, 1.0];
    if (model === 'logistic') {
        return llrLogistic(total, scores, probs, eloToScore(elo0), eloToScore(elo1));
    }
    // Normalized (nElo): sqrt(2) pentanomial scale, 800/ln10 logistic constant.
    const t0 = Math.sqrt(2) * elo0 / (800 / Math.log(10));
    const t1 = Math.sqrt(2) * elo1 / (800 / Math.log(10));
    return llrNormalized(total, scores, probs, t0, t1);
}

// Highly accurate approximation of the error function (Abramowitz and Stegun formula 7.1.26).
// It has a maximum error of about 1.5e-7.
// Code from https://hewgill.com/picomath/javascript/erf.js.html
function erf(x) {
    // Constants for approximation
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;

    // Save the sign of x
    let sign = Math.sign(x);
    x = Math.abs(x);

    // A&S formula 7.1.26
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
}

// Likelihood of superiority: probability that the new engine is better than the
// old. Matches fastchess implementation.
function calculateLOS(score, variancePerPair) {
    return (1.0 - erf(-(score - 0.5) / Math.sqrt(2.0 * variancePerPair))) / 2.0;
}

// Pentanomial Elo estimate, matching fastchess EloPentanomial. Returns both the
// logistic Elo and normalized Elo (nElo); neither depends on the SPRT model.
function estimatePentanomialElo(p) {
    const pairs = p.totalPairs;
    if (pairs === 0) return { elo: 0, error: 0, nelo: 0, neloError: 0 };

    const score = p.score;
    const variance = p.variance;
    const variancePerPair = variance / pairs;

    const CI95 = 1.959963984540054;
    const s2n = (s) => (s - 0.5) / Math.sqrt(2 * variance) * (800 / Math.log(10));
    const upper = score + CI95 * Math.sqrt(variancePerPair);
    const lower = score - CI95 * Math.sqrt(variancePerPair);
    const elo = score <= 0 ? -999 : (score >= 1 ? 999 : scoreToElo(score));
    const error = (scoreToElo(upper) - scoreToElo(lower)) / 2;
    const nelo = variance <= 0 ? 0 : s2n(score);
    const neloError = variance <= 0 ? 0 : (s2n(upper) - s2n(lower)) / 2;
    return { elo, error: Math.min(error, 200), nelo, neloError };
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

        isOldEngineMT = (typeof initThreadPoolOld === 'function');
        isNewEngineMT = (typeof initThreadPoolNew === 'function');

        wasmReady = true;
        setStatus('ready', 'WASM loaded and ready');
        runSprtBtn.disabled = false;

        // Always sync the UI: show/hide per-engine thread inputs and cap concurrency.
        // Default threads stay 1, so an MT build still plays single-threaded unless raised.
        updateMTUI();
        if (!isOldEngineMT && !isNewEngineMT) {
            sprtConcurrencyEl.value = CONFIG.concurrency;
        }

        log('WASM module initialized successfully', 'success');


        const classicalIcn = getVariantData('Classical').position;
        // Use the shared generator for the sanity test to ensure it works end-to-end
        const icn = generateSetupICN('Classical', 'w', 0, 1, []);
        const engine = EngineNew.from_icn(icn, { strength_level: 3 });
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
        sprtLog('Using stored max safe concurrency: ' + stored, 'info');
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
    sprtLog('Max safe concurrency detected: ' + lastOk, 'info');
    return lastOk;
}

async function runSprt() {
    if (!wasmReady || sprtRunning) return;

    // Immediate UI Reset
    sprtGamesEl.textContent = '0';
    sprtPairsEl.textContent = '0';
    sprtWinsEl.textContent = '0';
    sprtLossesEl.textContent = '0';
    sprtDrawsEl.textContent = '0';
    sprtLLREl.textContent = '0.00';
    sprtLLRContainer.style.setProperty('--progress', '0%');
    sprtLOSEl.textContent = '-';
    sprtLOSContainer.style.setProperty('--progress', '0%');
    sprtEloEl.textContent = '-';
    sprtEloEl.style.color = 'var(--text-dim)';
    sprtEloErrorEl.textContent = '0';
    // Reset last stats snapshot
    lastWins = 0;
    lastLosses = 0;
    lastDraws = 0;
    lastElo = 0;
    lastEloError = 0;
    lastNelo = 0;
    lastNeloError = 0;
    lastLLR = 0;
    lastBounds = null;
    lastTimeoutLosses = 0;
    pentaCounts = new PentaCounts();
    pendingPairResults = {};

    stopRequested = false;
    runSprtBtn.disabled = true;
    stopSprtBtn.disabled = false;
    sprtRunning = true;

    // Read configuration from UI into global CONFIG first (to persist defaults)
    CONFIG.boundsPreset = sprtBoundsPreset.value || 'stockfish_stc';
    CONFIG.boundsMode = sprtBoundsMode.value || 'gainer';
    CONFIG.alpha = parseFloat(sprtAlphaEl.value) || 0.05;
    CONFIG.beta = parseFloat(sprtBetaEl.value) || 0.05;
    CONFIG.tcMode = document.getElementById('sprtTcMode').value;
    CONFIG.timeControl = (sprtTimeControlEl.value || '').trim() || '10+0.1';

    const rawConcurrency = (sprtConcurrencyEl.value || '').toString().trim();
    if (rawConcurrency.toLowerCase() === 'max') {
        log('Concurrency set to "max" – probing for safe limit...', 'info');
        sprtLog('Concurrency set to "max" – probing for maximum safe workers...', 'info');
        CONFIG.concurrency = await detectMaxConcurrency(64);
    } else {
        CONFIG.concurrency = parseInt(rawConcurrency, 10) || 1;
    }
    {
        const { o, n } = currentThreadCounts();
        CONFIG.mtThreadsOld = o;
        CONFIG.mtThreadsNew = n;
    }
    CONFIG.hashOldMb = Math.min(64, Math.max(1, parseInt(sprtHashOldEl.value, 10) || 16));
    CONFIG.hashNewMb = Math.min(64, Math.max(1, parseInt(sprtHashNewEl.value, 10) || 16));
    if (CONFIG.mtThreadsOld > 1 || CONFIG.mtThreadsNew > 1) {
        const cap = mtConcurrencyCap(CONFIG.mtThreadsOld, CONFIG.mtThreadsNew);
        if (CONFIG.concurrency > cap) {
            log('Concurrency lowered ' + CONFIG.concurrency + ' -> ' + cap + ' so games x max(threads) fits your cores.', 'warn');
            CONFIG.concurrency = cap;
        }
    }
    CONFIG.minGames = parseInt(sprtMinGames.value, 10) || 500;
    const maxGamesVal = (sprtMaxGames.value || '').trim().toLowerCase();
    CONFIG.maxGames = (maxGamesVal === 'unlimited' || maxGamesVal === '') ? Infinity : (Number.isFinite(parseInt(maxGamesVal, 10)) ? parseInt(maxGamesVal, 10) : Infinity);
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
    sprtLLRBoundsEl.textContent = `${bounds.lower.toFixed(2)}, ${bounds.upper.toFixed(2)}`;
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
    const numberOfVariants = new Set(runVariantQueue.map(v => v.variant)).size;
    const isMultiVariantRun = numberOfVariants > 1;
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
    let timeoutLosses = 0;
    gameLogs = [];

    // Disable download buttons as logs are cleared
    downloadGamesTxtBtn.disabled = true;
    downloadGamesJsonBtn.disabled = true;

    sprtOutput.innerHTML = '';
    perVariantStats = {};
    clearLog();
    sprtStatusEl.textContent = 'SPRT status: running...';
    sprtStatusEl.className = 'status-text';
    const sprtBaseSeed = Date.now() ^ ((Math.random() * 0xFFFFFFFF) | 0);
    const gamesDisplay = maxGames === Infinity ? 'unlimited' : maxGames + ' games (' + (maxGames / 2) + ' pairs)';
    log('Starting SPRT: ' + gamesDisplay + ', Mode=' + runConfig.tcMode + ', TC=' + displayTcString + ', Seed=' + sprtBaseSeed, 'info');
    sprtLog('SPRT Test Started (noisy opening moves for first 8 ply, paired games)', 'info');

    const maxConcurrent = Math.max(1, runConfig.concurrency | 0);
    const workers = [];
    let activeWorkers = 0;
    let nextGameIndex = 0;

    function startWorker(worker, id) {
        const gameIndex = nextGameIndex++;
        if (maxGames !== null && gameIndex >= maxGames) return false;
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
            mtThreadsOld: runConfig.mtThreadsOld,
            mtThreadsNew: runConfig.mtThreadsNew,
            hashOldMb: runConfig.hashOldMb,
            hashNewMb: runConfig.hashNewMb,
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
                            if (msg.reason === 'time_forfeit' || msg.reason === 'timeout') {
                                const timeoutMsg = 'ALERT: Game ' + (msg.gameIndex + 1) + ' lost on time [' + (msg.variantName || 'Classical') + ']';
                                sprtLog(timeoutMsg, 'error');
                                log(timeoutMsg, 'error');
                                if (result === 'loss') timeoutLosses++;
                            }
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

                            // Bucket the pentanomial pair once both games of (2k, 2k+1) are in.
                            pendingPairResults[msg.gameIndex] = result;
                            const partnerIndex = msg.gameIndex ^ 1;
                            if (Object.prototype.hasOwnProperty.call(pendingPairResults, partnerIndex)) {
                                pentaCounts.addPair(pendingPairResults[msg.gameIndex], pendingPairResults[partnerIndex]);
                                delete pendingPairResults[msg.gameIndex];
                                delete pendingPairResults[partnerIndex];
                            }

                            // Pentanomial LLR/Elo drive the decision (matches src/bin/sprt.rs & Fishtest).
                            llr = calculatePentanomialLLR(pentaCounts, runConfig.elo0, runConfig.elo1, runConfig.model);
                            const { elo, error, nelo, neloError } = estimatePentanomialElo(pentaCounts);

                            // update last stats snapshot so Stop can show partial results
                            lastWins = wins;
                            lastLosses = losses;
                            lastDraws = draws;
                            lastElo = elo;
                            lastEloError = error;
                            lastNelo = nelo;
                            lastNeloError = neloError;
                            lastLLR = llr;
                            lastTimeoutLosses = timeoutLosses;

                            let totalPairs = pentaCounts.totalPairs;
                            sprtGamesEl.textContent = String(total);
                            sprtPairsEl.textContent = String(totalPairs);
                            sprtWinsEl.textContent = String(wins);
                            sprtLossesEl.textContent = String(losses);
                            sprtDrawsEl.textContent = String(draws);

                            sprtLLREl.textContent = String(llr.toFixed(2));
                            if (llr > 0) {
                                sprtLLRContainer.style.setProperty('--progress', Math.min(llr / bounds.upper, 1) * 100 + '%');
                            } else {
                                sprtLLRContainer.style.setProperty('--progress', Math.min(llr / bounds.lower, 1) * 100 + '%');
                            }

                            let los_percent = calculateLOS(pentaCounts.score, pentaCounts.variance / totalPairs) * 100;
                            if (isNaN(los_percent)) {
                                los_percent = 50;
                            }
                            sprtLOSEl.textContent = los_percent.toFixed(1) + '%';
                            sprtLOSContainer.style.setProperty('--progress', los_percent + '%');

                            sprtEloEl.textContent = String(elo.toFixed(1));
                            if (elo >= 1) {
                                sprtEloEl.style.color = 'var(--success)';
                            } else if (elo <= -1) {
                                sprtEloEl.style.color = 'var(--error)';
                            } else {
                                sprtEloEl.style.color = 'var(--text-dim)';
                            }
                            sprtEloErrorEl.textContent = String(error.toFixed(1));

                            sprtLog('Game ' + total + ': ' + result +
                                ' | W: ' + wins + ' L: ' + losses + ' D: ' + draws +
                                ' | Elo: ' + elo.toFixed(2) + ' ± ' + error.toFixed(2) +
                                ' | LLR: ' + llr.toFixed(2)
                            );

                            log(
                                'Games: ' + total + (maxGames === Infinity ? '' : '/' + maxGames) +
                                '  W:' + wins + ' L:' + losses + ' D:' + draws +
                                '  Elo: ' + elo.toFixed(2) + ' ± ' + error.toFixed(2) +
                                '  nElo: ' + nelo.toFixed(1) +
                                '  LLR: ' + llr.toFixed(2) +
                                ' [' + bounds.lower.toFixed(2) + ', ' + bounds.upper.toFixed(2) + ']',
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
                                log("CRITICAL ERROR: WASM Panic detected in game " + msg.gameIndex, 'error');
                                sprtLog("CRITICAL ERROR: WASM Panic detected in game " + msg.gameIndex, 'error');
                                sprtLog("Variant: " + (msg.variantName || 'Classical'), 'error');
                                sprtLog("Error: " + msg.error, 'error');

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
                                    sprtLog('WASM out-of-memory detected at concurrency ' + current + ' – new stored max: ' + proposed, 'warn');
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

        const { elo: finalElo, error: finalErr, nelo: finalNelo, neloError: finalNeloErr } = estimatePentanomialElo(pentaCounts);
        const verdict = llr >= bounds.upper ? 'PASSED (new > old)'
            : (llr <= bounds.lower ? 'FAILED (no gain)' : 'INCONCLUSIVE');

        log('SPRT Completed: ' + wins + 'W - ' + losses + 'L - ' + draws + 'D, Elo: ' +
            finalElo.toFixed(2) + ' ± ' + finalErr.toFixed(2) + ' (' + verdict + ')', 'success');

        // Detailed final summary block similar to sprt.js printResult
        const totalGames = wins + losses + draws;
        const winRate = totalGames > 0 ? (((wins + draws * 0.5) / totalGames) * 100).toFixed(1) : '0.0';
        const eloTextType = (finalElo >= 1 ? 'success' : (finalElo <= -1 ? 'error' : 'neutral'));
        sprtLog('');
        sprtLog('═══════════════════════════════════════════════════════════════════');
        sprtLog('Final Results:');
        sprtLog(`  [Tested in web] Mode: ${runConfig.tcMode} | TC: ${displayTcString} | Concurrency: ${runConfig.concurrency} | Variants: ${numberOfVariants}`);
        sprtLog(`  Elo Difference: ${(finalElo > 0 ? '+' : '') + finalElo.toFixed(2)} ± ${finalErr.toFixed(2)}`, eloTextType);
        sprtLog(`  nElo: ${(finalNelo > 0 ? '+' : '') + finalNelo.toFixed(2)} ± ${finalNeloErr.toFixed(2)}`, eloTextType);
        sprtLog(`  Games: ${totalGames} | W: ${wins} L: ${losses} D: ${draws} (${winRate}%)`);
        sprtLog(`  Pentanomial [${pentaCounts.totalPairs} pairs] (0-2): ${pentaCounts.displayText}`);
        if (lastBounds) {
            sprtLog(
                `  LLR: ${lastLLR.toFixed(2)}  bounds [${lastBounds.lower.toFixed(2)}, ${lastBounds.upper.toFixed(2)}]` +
                ` (${runConfig.model} model, [${runConfig.elo0}, ${runConfig.elo1}])`,
                (llr >= lastBounds.upper ? 'success' : (llr <= lastBounds.lower ? 'error' : 'warn'))
            );
        }
        if (timeoutLosses > 0) {
            sprtLog(`  ALERT: ${timeoutLosses} games ended by timeout (NEW ENGINE ONLY)`, 'error');
        }
        sprtLog('');

        sprtLog('Per-Variant Breakdown:');
        const variantNames = Object.keys(perVariantStats).sort();
        variantNames.forEach((name) => {
            const s = perVariantStats[name];
            const { elo, error } = estimateElo(s.wins, s.losses, s.draws);
            const vtTotal = s.wins + s.losses + s.draws;
            const vtScore = vtTotal > 0 ? (((s.wins + s.draws * 0.5) / vtTotal) * 100).toFixed(1) : '0.0';
            sprtLog(
                `  [${name}]: ${s.wins}W - ${s.losses}L - ${s.draws}D (${vtScore}%), Elo: ${(elo >= 0 ? '+' : '') + elo.toFixed(1)} ± ${error.toFixed(1)}`
            );
        });
        sprtLog('═══════════════════════════════════════════════════════════════════');
        // Update status line with colored verdict
        sprtStatusEl.textContent = 'SPRT status: ' + verdict;
        let cls = 'status-text ';
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
    sprtStatusEl.textContent = 'SPRT status: ABORTED';
    sprtStatusEl.className = 'status-text inconclusive';
    // If we have any completed games, show a partial results block
    const partialTotal = lastWins + lastLosses + lastDraws;
    if (partialTotal > 0) {
        const partialWinRate = (((lastWins + lastDraws * 0.5) / partialTotal) * 100).toFixed(1);
        sprtLog('');
        sprtLog('═══════════════════════════════════════════════════════════════════');
        sprtLog('Current Results (aborted):');
        sprtLog(`  Games: ${partialTotal} | W: ${lastWins} L: ${lastLosses} D: ${lastDraws} (${partialWinRate}%)`);
        sprtLog(`  Elo Difference: ${(lastElo >= 0 ? '+' : '') + lastElo.toFixed(2)} ± ${lastEloError.toFixed(2)}`);
        sprtLog(`  nElo: ${(lastNelo >= 0 ? '+' : '') + lastNelo.toFixed(1)} ± ${lastNeloError.toFixed(1)}`);
        if (lastBounds) {
            sprtLog(
                `  LLR: ${lastLLR.toFixed(2)}  bounds [${lastBounds.lower.toFixed(2)}, ${lastBounds.upper.toFixed(2)}]`
            );
        }
        sprtLog('');

        sprtLog('Per-Variant Breakdown (partial):');
        const variantNames = Object.keys(perVariantStats).sort();
        variantNames.forEach((name) => {
            const s = perVariantStats[name];
            const { elo, error } = estimateElo(s.wins, s.losses, s.draws);
            const vtTotal = s.wins + s.losses + s.draws;
            const vtScore = vtTotal > 0 ? (((s.wins + s.draws * 0.5) / vtTotal) * 100).toFixed(1) : '0.0';
            sprtLog(
                `  [${name}]: ${s.wins}W - ${s.losses}L - ${s.draws}D (${vtScore}%), Elo: ${(elo >= 0 ? '+' : '') + elo.toFixed(1)} ± ${error.toFixed(1)}`
            );
        });
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
sprtBoundsPreset.addEventListener('change', updateBoundsUi);
sprtTcMode.addEventListener('change', updateTcUi);

function updateBoundsUi() {
    const preset = sprtBoundsPreset.value;
    sprtBoundsMode.querySelector('[value="gainer"]').textContent = `Gainer [${BOUNDS_PRESETS[preset].gainer.join(', ')}]`;
    sprtBoundsMode.querySelector('[value="nonreg"]').textContent = `Non-regression [${BOUNDS_PRESETS[preset].nonreg.join(', ')}]`;
}

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
updateBoundsUi();
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

