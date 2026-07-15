/**
 * Native WASM Game Runner for SPRT/SPSA Testing (iwasm version)
 * 
 * This module runs games using iwasm (WebAssembly Micro Runtime),
 * providing native performance for WASM execution.
 * Supports all variants defined in variants.js
 * 
 * Requires iwasm in PATH and the WASI-targeted WASM binary.
 */

import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { existsSync, readFileSync, appendFileSync } from 'fs';
import { spawn } from 'child_process';
import vm from 'vm';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import variants from the web module
const variantsPath = join(__dirname, 'web', 'variants.js');
let VARIANTS = {};

// Load variants dynamically by reading file content to bypass Node ESM restrictions
async function loadVariants() {
    try {
        if (!existsSync(variantsPath)) {
            throw new Error(`Variants file not found at ${variantsPath}`);
        }

        const code = readFileSync(variantsPath, 'utf-8');

        // Strip export keywords to make it valid in VM context
        // converting "export const VARIANTS" -> "const VARIANTS"
        // converting "export function" -> "function"
        // converting "export { ... }" -> "// export { ... }"
        const scriptCode = code
            .replace(/export\s+const\s+/g, 'const ')
            .replace(/export\s+let\s+/g, 'let ')
            .replace(/export\s+var\s+/g, 'var ')
            .replace(/export\s+function\s+/g, 'function ')
            .replace(/export\s+class\s+/g, 'class ')
            .replace(/export\s+default\s+/g, ' ')
            .replace(/^\s*export\s*\{/gm, '// export {');

        const context = vm.createContext({});
        vm.runInContext(scriptCode, context);

        if (context.VARIANTS) {
            VARIANTS = context.VARIANTS;
        } else {
            // Try to reconstruct if separate exports
            if (typeof context.getAllVariants === 'function' && typeof context.getVariantData === 'function') {
                const names = context.getAllVariants();
                VARIANTS = {};
                for (const name of names) {
                    VARIANTS[name] = context.getVariantData(name);
                }
            } else {
                throw new Error("VARIANTS object not found in file and could not be reconstructed");
            }
        }
    } catch (e) {
        console.warn(`⚠️ Could not load variants from ${variantsPath}: ${e.message}`);
        // Fallback to Classical only
        VARIANTS = {
            Classical: {
                position: 'r1,8+|n2,8|b3,8|q4,8|k5,8+|b6,8|n7,8|r8,8+|p1,7+|p2,7+|p3,7+|p4,7+|p5,7+|p6,7+|p7,7+|p8,7+|P1,2+|P2,2+|P3,2+|P4,2+|P5,2+|P6,2+|P7,2+|P8,2+|R1,1+|N2,1|B3,1|Q4,1|K5,1+|B6,1|N7,1|R8,1+',
                game_rules: {
                    promotions_allowed: ['q', 'r', 'b', 'n'],
                },
            },
        };
    }
}

/**
 * Get all available variant names
 */
export function getAllVariants() {
    return Object.keys(VARIANTS);
}

/**
 * Check if we can use iwasm native execution
 */
export function canUseNativeWasm() {
    const wasmPath = join(__dirname, '..', 'target', 'wasm32-wasip1', 'release', 'spsa_engine.wasm');
    if (!existsSync(wasmPath)) {
        return false;
    }
    return true;
}

/**
 * Get the path to the WASI WASM binary
 */
export function getWasmPath() {
    return join(__dirname, '..', 'target', 'wasm32-wasip1', 'release', 'spsa_engine.wasm');
}

/**
 * Parse ICN position string into pieces array and special_rights
 */
function parseICNPosition(icnPosition) {
    const pieces = [];
    const special_rights = [];

    for (const pieceStr of icnPosition.split('|')) {
        if (!pieceStr) continue;

        const parts = pieceStr.split(',');
        if (parts.length !== 2) continue;

        const pieceInfo = parts[0];
        const yStr = parts[1];
        if (!pieceInfo) continue;

        // Split pieceInfo into piece code and x coordinate
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

        // Handle special rights (+ suffix)
        const hasSpecialRights = xRaw.endsWith('+') || yStr.endsWith('+');
        const x = xRaw.endsWith('+') ? xRaw.slice(0, -1) : xRaw;
        const y = yStr.endsWith('+') ? yStr.slice(0, -1) : yStr;

        if (isNaN(parseInt(x, 10)) || isNaN(parseInt(y, 10))) continue;

        if (hasSpecialRights) {
            special_rights.push(`${x},${y}`);
        }

        // Map piece types
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
            case 'ob':
                piece_type = 'x';
                player = 'n';
                break;
            case 'vo':
                piece_type = 'v';
                player = 'n';
                break;
            default:
                continue;
        }

        pieces.push({ x, y, piece_type, color: player });
    }

    return { pieces, special_rights };
}

/**
 * Get starting position for any variant
 */
function getVariantPosition(variantName) {
    const variantData = VARIANTS[variantName];
    if (!variantData) {
        throw new Error(`Unknown variant: ${variantName}. Available: ${Object.keys(VARIANTS).join(', ')}`);
    }

    const { pieces, special_rights } = parseICNPosition(variantData.position);

    // Calculate world bounds if variant has worldBorder
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
        fullmove_number: 1,
        game_rules: variantData.game_rules || null,
        world_bounds,
        variant: variantName,
    };
}

/**
 * Native iwasm Game Runner
 * Spawns multiple iwasm processes for parallel game execution
 */
export class NativeWasmRunner {
    constructor(options) {
        this.options = options;
        this.numWorkers = options.concurrency || 8;
        this.newWasmPath = options.newWasmPath || options.wasmPath || getWasmPath();
        this.oldWasmPath = options.oldWasmPath || this.newWasmPath; // Fallback to comparison vs self
        this.variantName = options.variant || 'Classical';
        this.materialThreshold = (options.materialThreshold !== undefined) ? options.materialThreshold : 1500;
        this.poolNew = [];

        this.poolOld = [];
    }

    async init() {
        await loadVariants();

        console.log('🚀 Using native iwasm WASM execution');
        console.log(`   New Engine: ${this.newWasmPath}`);
        console.log(`   Old Engine: ${this.oldWasmPath}`);
        console.log(`   Worker pool size: ${this.numWorkers}`);
        console.log(`   Available variants: ${getAllVariants().join(', ')}`);

        if (!existsSync(this.newWasmPath)) {
            throw new Error(`New WASM binary not found: ${this.newWasmPath}`);
        }
        if (!existsSync(this.oldWasmPath)) {
            throw new Error(`Old WASM binary not found: ${this.oldWasmPath}`);
        }

        if (this.newWasmPath === this.oldWasmPath) {
            // Self-play mode (SPSA or tuning)
        }

        // Initialize New pool
        for (let i = 0; i < this.numWorkers; i++) {
            const worker = this._createWorker(this.newWasmPath);
            this.poolNew.push({ worker, busy: false, id: `new-${i}` });
        }

        // Initialize Old pool
        if (this.newWasmPath === this.oldWasmPath) {
            this.poolOld = this.poolNew;
        } else {
            for (let i = 0; i < this.numWorkers; i++) {
                const worker = this._createWorker(this.oldWasmPath);
                this.poolOld.push({ worker, busy: false, id: `old-${i}` });
            }
        }

        console.log(`✅ ${this.numWorkers} iwasm workers ready`);
    }

    _createWorker(wasmPath) {
        const worker = spawn('iwasm', [wasmPath], {
            stdio: ['pipe', 'pipe', 'pipe']
        });

        worker.stderr.on('data', (data) => {
            // console.error(`Worker error: ${data.toString()}`);
        });

        worker.on('exit', (code) => {
            if (code !== 0 && code !== null) {
                // Determine if crash
            }
        });

        return worker;
    }

    async _sendRequest(workerEntry, request) {
        return new Promise((resolve, reject) => {
            const worker = workerEntry.worker;
            let buffer = '';

            const onData = (chunk) => {
                buffer += chunk.toString();
                if (buffer.includes('\n')) {
                    const lines = buffer.split('\n');
                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed) continue;
                        try {
                            const data = JSON.parse(trimmed);
                            if (data.from || data.error) {
                                cleanup();
                                resolve(data);
                                return;
                            }
                        } catch (e) { }
                    }
                    buffer = lines[lines.length - 1];
                }
            };

            const cleanup = () => {
                worker.stdout.removeListener('data', onData);
                worker.removeListener('error', onError);
            };

            const onError = (err) => {
                cleanup();
                reject(err);
            };

            worker.stdout.on('data', onData);
            worker.on('error', onError);

            const payload = JSON.stringify(request) + '\n';
            worker.stdin.write(payload);

            setTimeout(() => {
                cleanup();
                reject(new Error('Timeout awaiting engine response'));
            }, 30000);
        });
    }

    async _getFreeWorker(type) {
        const pool = type === 'old' ? this.poolOld : this.poolNew;
        const checkInterval = 10;
        while (true) {
            const freeWorker = pool.find(w => !w.busy);
            if (freeWorker) {
                freeWorker.busy = true;
                return freeWorker;
            }
            await new Promise(r => setTimeout(r, checkInterval));
        }
    }

    _releaseWorker(workerEntry) {
        workerEntry.busy = false;
    }

    _applyMove(position, move) {
        const pieces = position.board.pieces;
        const [fromX, fromY] = move.from.split(',');
        const [toX, toY] = move.to.split(',');

        const capturedIdx = pieces.findIndex(p => p.x === toX && p.y === toY);
        const isCapture = capturedIdx !== -1;
        if (isCapture) {
            pieces.splice(capturedIdx, 1);
        }

        const movingPiece = pieces.find(p => p.x === fromX && p.y === fromY);
        if (!movingPiece) {
            throw new Error(`No piece at ${move.from}`);
        }

        const isPawnMove = movingPiece.piece_type === 'p';

        if (movingPiece.piece_type === 'k') {
            const fromXi = parseInt(fromX, 10);
            const toXi = parseInt(toX, 10);
            const dx = toXi - fromXi;
            if (Math.abs(dx) >= 2) {
                const rookDir = dx > 0 ? 1 : -1;
                let rookXi = toXi + rookDir;
                while (true) {
                    const rookPiece = pieces.find(p =>
                        p.x === String(rookXi) && p.y === fromY &&
                        p.piece_type === 'r' && p.color === movingPiece.color
                    );
                    if (rookPiece) {
                        rookPiece.x = String(toXi - rookDir);
                        break;
                    }
                    rookXi += rookDir;
                    if (Math.abs(rookXi - toXi) > 10) break;
                }
            }
        }

        movingPiece.x = toX;
        movingPiece.y = toY;

        if (move.promotion) {
            const pType = move.promotion.toLowerCase();
            const map = {
                'q': 'q', 'r': 'r', 'b': 'b', 'n': 'n',
                'k': 'k', 'p': 'p', 'm': 'm', 'c': 'c', 'a': 'a',
                'h': 'h', 'g': 'g', 'l': 'l', 'i': 'i', 'z': 'z',
                'e': 'e', 'y': 'y', 'd': 'd', 's': 's', 'u': 'u',
                'o': 'o', 'x': 'x', 'v': 'v'
            };
            movingPiece.piece_type = map[pType] || pType;
        }

        position.turn = position.turn === 'w' ? 'b' : 'w';
        return { isCapture, isPawnMove };
    }

    _makePositionKey(position) {
        return JSON.stringify(position.board.pieces) + position.turn;
    }

    async playSingleSprtGame(gameConfig) {
        const { newPlaysWhite, timeControl, maxMoves, materialThreshold, variantName, whiteParams, blackParams } = gameConfig;

        const whiteType = newPlaysWhite ? 'new' : 'old';
        const blackType = newPlaysWhite ? 'old' : 'new';
        const newColor = newPlaysWhite ? 'w' : 'b';

        // Acquire workers
        const whiteEntry = await this._getFreeWorker(whiteType);
        const blackEntry = await this._getFreeWorker(blackType);

        const moveLines = [];
        try {
            const position = getVariantPosition(variantName);
            const gameRules = position.game_rules || {};

            // Time Control Setup
            let whiteTime = 0, blackTime = 0, increment = 0, fixedTime = null;
            const tcStr = String(timeControl || '10+0.1').trim();
            if (tcStr.includes('+')) {
                const parts = tcStr.split('+');
                const base = parseFloat(parts[0]);
                increment = parseFloat(parts[1]) * 1000;
                whiteTime = base * 1000;
                blackTime = base * 1000;
            } else {
                if (tcStr.endsWith('s') && !tcStr.endsWith('ms')) {
                    fixedTime = parseFloat(tcStr) * 1000;
                } else {
                    fixedTime = parseFloat(tcStr);
                }
            }

            let timeLimit = fixedTime || (whiteTime / 20 + increment);

            const moveHistory = [];
            const repetitionCounts = new Map();
            let halfmoveClock = 0;
            let lastEval = null;

            const initialKey = this._makePositionKey(position);
            repetitionCounts.set(initialKey, 1);

            for (let moveNum = 0; moveNum < maxMoves; moveNum++) {
                const isWhiteTurn = position.turn === 'w';
                const workerEntry = isWhiteTurn ? whiteEntry : blackEntry;
                const searchParams = isWhiteTurn ? whiteParams : blackParams;

                if (!fixedTime) {
                    const remaining = isWhiteTurn ? whiteTime : blackTime;
                    if (remaining <= 0) {
                        const winningColor = isWhiteTurn ? 'b' : 'w';
                        const result = winningColor === newColor ? 'new' : 'old';
                        return { result, reason: 'time_forfeit', log: moveLines.join('\n') };
                    }
                    timeLimit = (remaining / 20) + increment;
                    timeLimit = Math.max(50, Math.min(timeLimit, remaining - 50));
                }
                timeLimit = Math.floor(timeLimit);

                const gameInput = {
                    board: position.board,
                    turn: position.turn,
                    special_rights: position.special_rights || [],
                    move_history: moveHistory,
                    time_limit_ms: timeLimit,
                    game_rules: gameRules,
                    search_params: searchParams,
                    fullmove_number: position.fullmove_number || 1,
                    variant: position.variant,
                    world_bounds: position.world_bounds
                };

                const t0 = Date.now();
                const response = await this._sendRequest(workerEntry, gameInput);
                const elapsed = Date.now() - t0;

                if (!fixedTime) {
                    if (isWhiteTurn) whiteTime = whiteTime - elapsed + increment;
                    else blackTime = blackTime - elapsed + increment;
                }

                if (response.error) {
                    if (response.error === 'no_legal_moves') {
                        const winningColor = isWhiteTurn ? 'b' : 'w';
                        const result = winningColor === newColor ? 'new' : 'old';
                        return { result, reason: 'no_legal_moves', log: moveLines.join('\n') };
                    }
                    throw new Error(response.error);
                }

                if (!response.from || !response.to) {
                    const winningColor = isWhiteTurn ? 'b' : 'w';
                    const result = winningColor === newColor ? 'new' : 'old';
                    return { result, reason: 'checkmate', log: moveLines.join('\n') };
                }

                if (typeof response.eval === 'number') {
                    lastEval = isWhiteTurn ? response.eval : -response.eval;
                }

                let promotionSuffix = '';
                if (response.promotion) {
                    const siteCode = this._engineLetterToSiteCode(response.promotion);
                    promotionSuffix = '=' + (isWhiteTurn ? siteCode.toUpperCase() : siteCode.toLowerCase());
                }
                moveLines.push(
                    (isWhiteTurn ? 'W' : 'B') + ': ' + response.from + '>' + response.to + promotionSuffix
                );

                if (moveHistory.length >= 20 && lastEval !== null && materialThreshold > 0) {
                    if (Math.abs(lastEval) >= materialThreshold) {
                        const winningColor = lastEval > 0 ? 'w' : 'b';
                        const result = winningColor === newColor ? 'new' : 'old';
                        return { result, log: moveLines.join('\n'), reason: 'material_adjudication' };
                    }
                }

                const move = { from: response.from, to: response.to, promotion: response.promotion };
                const { isCapture, isPawnMove } = this._applyMove(position, move);
                moveHistory.push(move);

                if (isPawnMove || isCapture) halfmoveClock = 0;
                else halfmoveClock++;

                if (halfmoveClock >= 100) return { result: 'draw', reason: 'fifty_move', log: moveLines.join('\n') };

                const posKey = this._makePositionKey(position);
                const repCount = (repetitionCounts.get(posKey) || 0) + 1;
                repetitionCounts.set(posKey, repCount);
                if (repCount >= 3) return { result: 'draw', reason: 'threefold', log: moveLines.join('\n') };
            }

            return { result: 'draw', reason: 'max_moves', log: moveLines.join('\n') };
        } catch (e) {
            console.error(`Game error: ${e.message}`);
            return { result: 'draw', reason: 'error', log: moveLines ? moveLines.join('\n') : e.message };
        } finally {
            this._releaseWorker(whiteEntry);
            this._releaseWorker(blackEntry);
        }
    }

    _engineLetterToSiteCode(letter) {
        const map = {
            'k': 'K', 'q': 'Q', 'r': 'R', 'b': 'B', 'n': 'N', 'p': 'P',
            'm': 'AM', 'c': 'CH', 'a': 'AR', 'h': 'HA', 'g': 'GU',
            'l': 'CA', 'i': 'GI', 'z': 'ZE', 'e': 'CE', 'y': 'RQ',
            'd': 'RC', 's': 'NR', 'u': 'HU', 'o': 'RO', 'x': 'OB', 'v': 'VO'
        };
        return map[letter] || letter.toUpperCase();
    }

    _generateICN(workerLog, gameIndex, result, newPlaysWhite, endReason, materialThreshold, timeControl, variantName) {
        const utc = new Date();
        const pad = (n) => String(n).padStart(2, '0');
        const utcDate = `${utc.getUTCFullYear()}.${pad(utc.getUTCMonth() + 1)}.${pad(utc.getUTCDate())}`;
        const utcTime = `${pad(utc.getUTCHours())}:${pad(utc.getUTCMinutes())}:${pad(utc.getUTCSeconds())}`;

        let resultToken = '*';
        if (result === 'draw') {
            resultToken = '1/2-1/2';
        } else if (result === 'new' || result === 'old') {
            const newWon = (result === 'new');
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
            if (endReason === 'material_adjudication') termination = `Material adjudication`;
            else if (endReason === 'max_moves') termination = `Draw by maximum moves reached`;
            else if (endReason === 'threefold') termination = `Draw by threefold repetition`;
            else if (endReason === 'fifty_move') termination = `Draw by fifty-move rule`;
            else if (endReason === 'time_forfeit') termination = `Loss on time`;
            else if (endReason === 'checkmate') termination = `Checkmate`;

            if (termination) headerList.push(`[Termination "${termination}"]`);
        }


        const headers = headerList.join(' ');

        let startPositionStr = '';
        try {
            const vdata = VARIANTS[variantName];
            if (vdata && typeof vdata.position === 'string') {
                startPositionStr = vdata.position;
            } else {
                startPositionStr = getVariantPosition(variantName).position || '';
            }
        } catch (e) { }

        const lines = (workerLog || '').split('\n').filter(l => l.trim().length > 0 && (l.startsWith('W:') || l.startsWith('B:')));
        const moves = lines.map((line) => {
            const idx = line.indexOf(':');
            return idx === -1 ? '' : line.slice(idx + 1).trim().replace(/\s+/g, '');
        }).filter(Boolean);
        const movesStr = moves.join('|');

        let whiteRank = null;
        let blackRank = null;
        try {
            const vdata = VARIANTS[variantName];
            if (vdata && vdata.game_rules && vdata.game_rules.promotion_ranks) {
                const ranks = vdata.game_rules.promotion_ranks;
                if (ranks.white && ranks.white.length > 0) whiteRank = ranks.white[0];
                if (ranks.black && ranks.black.length > 0) blackRank = ranks.black[0];
            }
        } catch (e) { }

        if (whiteRank === null || blackRank === null) {
            let minY = Infinity;
            let maxY = -Infinity;
            const pieces = startPositionStr.split('|');
            for (const p of pieces) {
                const parts = p.split(',');
                if (parts.length >= 2) {
                    const y = parseInt(parts[1], 10);
                    if (!isNaN(y)) {
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                    }
                }
            }
            if (minY !== Infinity) {
                if (whiteRank === null) whiteRank = String(maxY);
                if (blackRank === null) blackRank = String(minY);
            } else {
                if (whiteRank === null) whiteRank = '8';
                if (blackRank === null) blackRank = '1';
            }
        }

        const promotionRanksToken = `(${whiteRank}|${blackRank})`;
        return `${headers} w 0/100 1 ${promotionRanksToken} ${startPositionStr}${movesStr ? ' ' + movesStr : ''}`;
    }

    async playSingleGame(gameConfig) {
        // Used by SPSA
        const res = await this.playSingleSprtGame({
            ...gameConfig,
            newPlaysWhite: gameConfig.plusPlaysWhite,
            whiteParams: gameConfig.whiteParams,
            blackParams: gameConfig.blackParams
        });

        let score = 0.5;
        // In SPSA: 'new' means Plus, 'old' means Minus
        if (res.result === 'new') score = 1;
        else if (res.result === 'old') score = 0;

        return { result: score };
    }

    /**
     * SPSA Runner Method
     * Compatible with spsa.js interface: runGames(thetaPlus, thetaMinus, numGames)
     */
    async runGames(thetaPlus, thetaMinus, numGames) {
        const jobs = [];
        const variantName = this.variantName || 'Classical';
        const tc = this.options.tc;

        for (let i = 0; i < numGames; i++) {
            // Game 1: Plus (White) vs Minus (Black)
            jobs.push({
                whiteParams: thetaPlus,
                blackParams: thetaMinus,
                plusPlaysWhite: true,
                timeControl: tc,
                maxMoves: 150,
                materialThreshold: this.materialThreshold,
                variantName

            });
            // Game 2: Minus (White) vs Plus (Black)
            jobs.push({
                whiteParams: thetaMinus,
                blackParams: thetaPlus,
                plusPlaysWhite: false,
                timeControl: tc,
                maxMoves: 150,
                materialThreshold: this.materialThreshold,
                variantName

            });
        }

        const results = { plusWins: 0, minusWins: 0, draws: 0, stalled: false };
        const tasks = jobs.map(async (job) => {
            const res = await this.playSingleGame(job); // Returns 1 (Plus win), 0 (Minus win), 0.5 (Draw)
            if (res.result === 1) results.plusWins++;
            else if (res.result === 0) results.minusWins++;
            else results.draws++;
        });

        await Promise.all(tasks);
        return results;
    }

    async runSprtGames(numGames, options = {}) {
        const results = { newWins: 0, oldWins: 0, draws: 0, logs: [] };
        const { variants = [] } = options;
        const variantList = (variants && variants.length > 0) ? variants : [this.variantName];

        console.log(`   Running with variants: ${variantList.length > 5 ? variantList.length + ' variants' : variantList.join(', ')}`);

        const tasks = [];
        let completed = 0;
        let variantIdx = 0;

        for (let i = 0; i < numGames; i++) {
            const variantName = variantList[variantIdx % variantList.length];
            variantIdx++;

            const tc = options.timeControl || '10+0.1';
            const mt = (options.materialThreshold !== undefined) ? options.materialThreshold : this.materialThreshold;
            const mxm = options.maxMoves || 150;

            tasks.push({ newPlaysWhite: true, timeControl: tc, maxMoves: mxm, materialThreshold: mt, variantName });
            tasks.push({ newPlaysWhite: false, timeControl: tc, maxMoves: mxm, materialThreshold: mt, variantName });


        }

        const promises = tasks.map(async (config, index) => {
            const res = await this.playSingleSprtGame(config);
            if (res.result === 'new') results.newWins++;
            else if (res.result === 'old') results.oldWins++;
            else results.draws++;

            try {
                const icn = this._generateICN(
                    res.log,
                    index + 1,
                    res.result,
                    config.newPlaysWhite,
                    res.reason,
                    config.materialThreshold,
                    String(config.timeControl),
                    config.variantName
                );
                appendFileSync('sprt-games.icn', icn + '\n\n');
            } catch (err) {
                console.error('Failed to save ICN:', err);
            }

            completed++;
            process.stdout.write(`\r   Games: ${completed}/${tasks.length}  (+${results.newWins} =${results.draws} -${results.oldWins})`);
            return res;
        });

        await Promise.all(promises);
        console.log();

        return results;
    }

    async close() {
        for (const w of this.poolNew) {
            w.worker.kill();
        }
        if (this.poolOld && this.poolOld !== this.poolNew) {
            for (const w of this.poolOld) {
                w.worker.kill();
            }
        }
    }
}
