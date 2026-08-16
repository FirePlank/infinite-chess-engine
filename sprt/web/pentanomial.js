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

function eloToScore(eloDiff) {
    return 1 / (1 + Math.pow(10, -eloDiff / 400));
}
function scoreToElo(s) {
    s = Math.max(1e-9, Math.min(1 - 1e-9, s)); // clamp between 0 and 1
    return -400 * Math.log10(1 / s - 1);
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

export { PentaCounts, calculateLOS, calculatePentanomialLLR, estimateElo, estimatePentanomialElo };
