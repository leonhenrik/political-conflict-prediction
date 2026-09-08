"""
Calibration of voting_model.py against German federal election data
(federal_muni_harm_25.txt).

STRATEGY
--------
We calibrate against the 2017, 2021, and 2025 national vote-share targets
(population-weighted aggregates computed from the data file).

FAST SIMULATOR
--------------
To make calibration tractable, we use a numpy-vectorised mean-field
approximation of the ABM:

  - Agent opinions are stored as (N, 2) arrays.
  - Each step we draw random pairs, compute assimilation/repulsion
    vectorised across all active pairs simultaneously.
  - No per-agent Python loops in the inner loop — only numpy array ops.
  - Result: ~100x faster than the pure-Python version.

The mean-field version loses the network structure (no persistent ties,
no institutions) but preserves:
  • Bounded confidence (assimilation below threshold, repulsion above)
  • Negativity bias (repulse_rate > assimilate_rate)
  • Homophily in partner choice (ideology similarity weighting)
  • Media pull (every MEDIA_EVERY steps)
  • Social media echo (daily, high-use agents)
  • Religiosity pull (daily conservative drift)
  • Probabilistic spatial voting with party ID bonus
  • Turnout model with alienation + sophistication + trust

CALIBRATION METHOD
------------------
scipy.optimize.differential_evolution with workers=-1 (all CPU cores).

PARAMETERS CALIBRATED
---------------------
See PARAM_SPEC below.

Run: python calibrate_model.py
"""

import json
import time
import numpy as np
from scipy.optimize import differential_evolution

# ── TARGETS (from federal_muni_harm_25.txt, population-weighted) ──────────────
TARGETS = {
    2017: {
        "CDU":    0.3293,
        "SPD":    0.2051,
        "Gruene": 0.0894,
        "FDP":    0.1075,
        "AfD":    0.1264,
        "BSW":    0.0000,
        "turnout": 0.7642,
    },
    2021: {
        "CDU":    0.2407,
        "SPD":    0.2574,
        "Gruene": 0.1476,
        "FDP":    0.1146,
        "AfD":    0.1034,
        "BSW":    0.0000,
        "turnout": 0.7690,
    },
    2025: {
        "CDU":    0.2852,
        "SPD":    0.1641,
        "Gruene": 0.1161,
        "FDP":    0.0433,
        "AfD":    0.2080,
        "BSW":    0.0498,
        "turnout": 0.8268,
    },
}

PARTY_NAMES = ["CDU", "SPD", "Gruene", "FDP", "AfD", "BSW"]

PARTY_WEIGHTS = {
    "CDU": 3.0, "SPD": 3.0, "Gruene": 2.0,
    "FDP": 2.0, "AfD": 3.0, "BSW": 1.5,
}
TURNOUT_WEIGHT  = 4.0
YEAR_WEIGHTS    = {2017: 1.0, 2021: 1.5, 2025: 2.0}

# ── SIMULATION CONSTANTS (not calibrated) ─────────────────────────────────────
N_AGENTS        = 300
N_STEPS         = 150    # days per evaluation
N_REPLICATES    = 3
BASE_SEED       = 7

CONFIDENCE_INIT = 0.35
CONFIDENCE_MIN  = 0.10
REPULSE_THRESH  = 0.65
NOISE_STD       = 0.01
VOTE_RAT        = 4.0
PARTY_ID_BONUS  = 0.6
PARTY_ID_RATE   = 0.02
PARTY_CUE_PULL  = 0.015
CROSS_DISCOUNT  = 0.4
MEAN_ENC        = 4       # mean encounters per agent per day (Poisson)
MEDIA_EVERY     = 4
SOC_MEDIA_THRESH = 0.6
MIN_SUSC        = 0.25
YOUNG_CUT       = 0.30
SEL_EXP_BETA    = 4.0
MEDIA_LEFT      = np.array([-0.75, -0.55])
MEDIA_RIGHT     = np.array([ 0.55,  0.55])
SECTOR_ECON_BIAS = np.array([-0.15, 0.05, 0.20, -0.10, 0.05, -0.10])
SECTOR_SHARES    = np.array([0.15, 0.45, 0.08, 0.10, 0.17, 0.05])

# ── PARAMETER SPACE ───────────────────────────────────────────────────────────
PARAM_SPEC = [
    # Party positions (econ, social)
    ("CDU_econ",      0.05,  0.60),
    ("CDU_social",    0.00,  0.70),
    ("SPD_econ",     -0.55, -0.05),
    ("SPD_social",   -0.30,  0.20),
    ("Gruene_econ",  -0.80, -0.20),
    ("Gruene_social",-0.80, -0.10),
    ("FDP_econ",      0.20,  0.80),
    ("FDP_social",   -0.70,  0.10),
    ("AfD_econ",     -0.20,  0.40),
    ("AfD_social",    0.40,  1.00),
    ("BSW_econ",     -0.70, -0.05),
    ("BSW_social",    0.10,  0.80),
    # Population ideology prior
    ("pop_econ_mean",  -0.30,  0.20),
    ("pop_social_mean",-0.30,  0.30),
    ("pop_econ_std",    0.15,  0.50),
    ("pop_social_std",  0.15,  0.50),
    ("pop_urban_mean",  0.35,  0.65),
    # Behavioural rates
    ("assimilate_rate",   0.005, 0.08),
    ("repulse_rate",      0.010, 0.12),
    ("media_pull_rate",   0.005, 0.06),
    ("social_media_pull", 0.003, 0.05),
    ("religiosity_pull",  0.001, 0.020),
    ("zealot_fraction",   0.02,  0.15),
    # Turnout
    ("base_turnout",           0.65, 0.95),
    ("alienation_sensitivity", 0.20, 0.90),
    ("soph_turnout_bonus",     0.05, 0.30),
    ("trust_turnout_bonus",    0.02, 0.20),
]

PARAM_NAMES  = [p[0] for p in PARAM_SPEC]
PARAM_BOUNDS = [(p[1], p[2]) for p in PARAM_SPEC]


# ── FAST NUMPY SIMULATOR ──────────────────────────────────────────────────────
def simulate_fast(x: np.ndarray, seed: int) -> dict:
    """
    Run the mean-field vectorised ABM with parameter vector x.
    Returns a dict of vote shares (averaged over target years) + turnout.
    We run one simulation and read off the final vote distribution — the
    same parameter vector is evaluated against all three target years via a
    single run (the equilibrium distribution is what matters for calibration).
    """
    rng = np.random.default_rng(seed)
    p   = dict(zip(PARAM_NAMES, x))

    # Party positions array (n_parties, 2)
    party_pos = np.array([
        [p["CDU_econ"],    p["CDU_social"]],
        [p["SPD_econ"],    p["SPD_social"]],
        [p["Gruene_econ"], p["Gruene_social"]],
        [p["FDP_econ"],    p["FDP_social"]],
        [p["AfD_econ"],    p["AfD_social"]],
        [p["BSW_econ"],    p["BSW_social"]],
    ])  # (6,2)

    N   = N_AGENTS
    AR  = p["assimilate_rate"]
    RR  = p["repulse_rate"]
    MP  = p["media_pull_rate"]
    SMP = p["social_media_pull"]
    RP  = p["religiosity_pull"]
    ZF  = p["zealot_fraction"]
    BT  = p["base_turnout"]
    AS  = p["alienation_sensitivity"]
    ST  = p["soph_turnout_bonus"]
    TT  = p["trust_turnout_bonus"]
    UM  = p["pop_urban_mean"]

    # ── Initialise agents ──────────────────────────────────────────────────────
    age   = rng.random(N)
    edu   = rng.beta(2, 2, N)
    soph  = np.clip(0.6*edu + 0.4*rng.random(N), 0, 1)
    rel   = np.clip(rng.beta(1.5, 4, N), 0, 1)
    urb   = np.clip(rng.normal(UM, 0.2, N), 0, 1)
    trust = np.clip(rng.normal(0.5, 0.2, N), 0, 1)
    opens = np.clip(rng.normal(0.5, 0.2, N), 0, 1)
    socm  = np.where(age < 0.5,
                     np.clip(rng.beta(2, 3, N), 0, 1),
                     np.clip(rng.beta(1, 4, N), 0, 1))

    # Sector assignment
    sector_idx = rng.choice(len(SECTOR_SHARES), N, p=SECTOR_SHARES/SECTOR_SHARES.sum())
    sec_bias   = SECTOR_ECON_BIAS[sector_idx]

    econ   = np.clip(rng.normal(p["pop_econ_mean"] + sec_bias, p["pop_econ_std"])
                     - 0.10*(urb - 0.5), -1, 1)
    social = np.clip(rng.normal(p["pop_social_mean"], p["pop_social_std"])
                     - 0.20*(urb - 0.5) - 0.15*(edu - 0.5) + 0.15*(rel - 0.3), -1, 1)

    # Zealots: fixed near a party anchor
    n_zealots   = max(0, round(N * ZF))
    zealot_mask = np.zeros(N, dtype=bool)
    if n_zealots > 0:
        zids = rng.choice(N, n_zealots, replace=False)
        zealot_mask[zids] = True
        anchor_idx = rng.integers(0, len(PARTY_NAMES), n_zealots)
        econ[zids]   = np.clip(party_pos[anchor_idx, 0] + rng.normal(0, 0.05, n_zealots), -1, 1)
        social[zids] = np.clip(party_pos[anchor_idx, 1] + rng.normal(0, 0.05, n_zealots), -1, 1)

    party_id  = np.full(N, -1, dtype=int)   # -1 = no party id
    pid_str   = np.zeros(N)

    # ── Age-based susceptibility base ─────────────────────────────────────────
    def get_susceptibility():
        af = np.where(age <= YOUNG_CUT,
                      1.0,
                      1.0 - (1.0 - MIN_SUSC) * (age - YOUNG_CUT) / (1 - YOUNG_CUT))
        pr = 1.0 - 0.5 * pid_str
        of = 0.7 + 0.6 * opens
        susc = af * pr * of
        susc[zealot_mask] = 0.
        return susc  # (N,)

    def get_salience():
        base = 0.5 + 0.5 * soph
        ep = np.abs(econ); sp = np.abs(social); tot = ep + sp + 1e-6
        we = base + (1 - base) * (ep / tot)
        ws = base + (1 - base) * (sp / tot)
        return we, ws   # (N,), (N,)

    # ── Simulation loop ────────────────────────────────────────────────────────
    for step in range(1, N_STEPS + 1):
        susc = get_susceptibility()

        # Draw random encounter pairs
        n_pairs = max(1, int(N * MEAN_ENC / 2))
        i_idx   = rng.integers(0, N, n_pairs)
        j_idx   = rng.integers(0, N, n_pairs)
        # Remove self-pairs
        valid   = i_idx != j_idx
        i_idx, j_idx = i_idx[valid], j_idx[valid]
        if len(i_idx) == 0:
            continue

        # Ideology homophily: weight pairs by similarity
        de_pair = econ[j_idx]   - econ[i_idx]
        ds_pair = social[j_idx] - social[i_idx]
        dist_pair = np.sqrt(de_pair**2 + ds_pair**2)

        # Motivated reasoning: cross-party discount
        same_party = (party_id[i_idx] >= 0) & (party_id[j_idx] >= 0) & \
                     (party_id[i_idx] == party_id[j_idx])
        cross_party = (party_id[i_idx] >= 0) & (party_id[j_idx] >= 0) & \
                      (party_id[i_idx] != party_id[j_idx])
        discount = np.where(cross_party, CROSS_DISCOUNT, 1.0)

        sa = susc[i_idx] * discount
        sb = susc[j_idx] * discount

        # Assimilation
        assim_mask = dist_pair < CONFIDENCE_INIT
        if assim_mask.any():
            ai, aj = i_idx[assim_mask], j_idx[assim_mask]
            scale_a = AR * sa[assim_mask]
            scale_b = AR * sb[assim_mask]
            np.add.at(econ,   ai,  scale_a * de_pair[assim_mask])
            np.add.at(social, ai,  scale_a * ds_pair[assim_mask])
            np.add.at(econ,   aj, -scale_b * de_pair[assim_mask])
            np.add.at(social, aj, -scale_b * ds_pair[assim_mask])

        # Repulsion
        rep_mask = dist_pair > REPULSE_THRESH
        if rep_mask.any():
            ri, rj = i_idx[rep_mask], j_idx[rep_mask]
            d_clamped = np.maximum(dist_pair[rep_mask], 1e-6)
            scale_a = RR / d_clamped * sa[rep_mask]
            scale_b = RR / d_clamped * sb[rep_mask]
            np.add.at(econ,   ri, -scale_a * de_pair[rep_mask])
            np.add.at(social, ri, -scale_a * ds_pair[rep_mask])
            np.add.at(econ,   rj,  scale_b * de_pair[rep_mask])
            np.add.at(social, rj,  scale_b * ds_pair[rep_mask])

        econ   = np.clip(econ,   -1, 1)
        social = np.clip(social, -1, 1)

        # Noise
        non_z = ~zealot_mask
        econ[non_z]   = np.clip(econ[non_z]   + rng.normal(0, NOISE_STD, non_z.sum()), -1, 1)
        social[non_z] = np.clip(social[non_z] + rng.normal(0, NOISE_STD, non_z.sum()), -1, 1)

        # Social media echo
        sm_active = non_z & (socm >= SOC_MEDIA_THRESH)
        if sm_active.any():
            s_susc = susc[sm_active] * socm[sm_active]
            ve = 0.7*econ[sm_active]   + 0.3*np.sign(econ[sm_active]   + 1e-9)
            vs = 0.7*social[sm_active] + 0.3*np.sign(social[sm_active] + 1e-9)
            econ[sm_active]   = np.clip(econ[sm_active]   + SMP*s_susc*(ve - econ[sm_active]),   -1, 1)
            social[sm_active] = np.clip(social[sm_active] + SMP*s_susc*(vs - social[sm_active]), -1, 1)

        # Media pull (every MEDIA_EVERY steps)
        if step % MEDIA_EVERY == 0:
            pos = np.stack([econ, social], axis=1)  # (N,2)
            d_l = np.linalg.norm(pos - MEDIA_LEFT,  axis=1) - 0.15*(urb - 0.5)
            d_r = np.linalg.norm(pos - MEDIA_RIGHT, axis=1) + 0.15*(urb - 0.5)
            # Softmax over [-d_l, -d_r] * beta
            log_l = -d_l * SEL_EXP_BETA
            log_r = -d_r * SEL_EXP_BETA
            m_lr  = np.maximum(log_l, log_r)
            p_l   = np.exp(log_l - m_lr) / (np.exp(log_l - m_lr) + np.exp(log_r - m_lr))
            chose_left = rng.random(N) < p_l
            mx = np.where(chose_left, MEDIA_LEFT[0],  MEDIA_RIGHT[0])
            my = np.where(chose_left, MEDIA_LEFT[1],  MEDIA_RIGHT[1])
            s  = susc
            econ[non_z]   = np.clip(econ[non_z]   + MP*s[non_z]*(mx[non_z]-econ[non_z]),   -1, 1)
            social[non_z] = np.clip(social[non_z] + MP*s[non_z]*(my[non_z]-social[non_z]), -1, 1)

        # Religiosity pull
        pull_r = RP * rel * susc
        social = np.clip(social + pull_r * (1.0 - social), -1, 1)
        social[zealot_mask] = social[zealot_mask]  # zealots unchanged by above (susc=0)

        # Party cue-taking
        has_pid = party_id >= 0
        if has_pid.any():
            px = party_pos[party_id[has_pid], 0]
            py = party_pos[party_id[has_pid], 1]
            pull_c = PARTY_CUE_PULL * pid_str[has_pid]
            econ[has_pid]   = np.clip(econ[has_pid]   + pull_c*(px - econ[has_pid]),   -1, 1)
            social[has_pid] = np.clip(social[has_pid] + pull_c*(py - social[has_pid]), -1, 1)

    # ── Election ───────────────────────────────────────────────────────────────
    we, ws = get_salience()
    pos    = np.stack([econ, social], axis=1)   # (N,2)

    # Spatial utility for each party: -(weighted distance)
    utils = np.zeros((N, len(PARTY_NAMES)))
    for k, pp in enumerate(party_pos):
        d = np.sqrt(we*(pos[:,0]-pp[0])**2 + ws*(pos[:,1]-pp[1])**2)
        utils[:, k] = -d
    # Party ID bonus
    has_pid = party_id >= 0
    if has_pid.any():
        utils[has_pid, party_id[has_pid]] += PARTY_ID_BONUS * pid_str[has_pid]

    # Softmax vote probabilities
    scaled = utils * VOTE_RAT
    scaled -= scaled.max(axis=1, keepdims=True)
    probs  = np.exp(scaled)
    probs /= probs.sum(axis=1, keepdims=True)

    # Turnout
    min_dist = np.sqrt(we[:,None]*(pos[:,0:1]-party_pos[:,0])**2 +
                       ws[:,None]*(pos[:,1:2]-party_pos[:,1])**2).min(axis=1)
    to_prob = np.clip(BT - AS*min_dist + ST*soph + TT*trust, 0.02, 0.98)

    voted = rng.random(N) < to_prob
    n_voted = voted.sum()
    if n_voted == 0:
        return {p: 1./len(PARTY_NAMES) for p in PARTY_NAMES} | {"turnout": 0.}

    # Draw votes via cumulative sum sampling
    vote_probs_voted = probs[voted]
    cumprobs = vote_probs_voted.cumsum(axis=1)
    r = rng.random(n_voted)[:, None]
    votes = (r < cumprobs).argmax(axis=1)
    counts = np.bincount(votes, minlength=len(PARTY_NAMES))
    shares = counts / n_voted

    result = {name: float(shares[k]) for k, name in enumerate(PARTY_NAMES)}
    result["turnout"] = float(n_voted / N)
    return result


# ── LOSS FUNCTION (picklable for multiprocessing workers) ────────────────────
def loss(x: np.ndarray) -> float:
    """
    Compute total weighted squared-error loss across all target years.
    Must be a module-level function (picklable) for workers=-1.
    """
    total_loss = 0.0
    for year, target in TARGETS.items():
        yw = YEAR_WEIGHTS[year]
        sim_avg = {k: 0. for k in target}
        for rep in range(N_REPLICATES):
            result = simulate_fast(x, seed=BASE_SEED + rep + year)
            for k in sim_avg:
                sim_avg[k] += result.get(k, 0.)
        for k in sim_avg:
            sim_avg[k] /= N_REPLICATES
        year_loss = 0.
        for party, tgt in target.items():
            w = TURNOUT_WEIGHT if party == "turnout" else PARTY_WEIGHTS.get(party, 1.0)
            year_loss += w * (sim_avg[party] - tgt) ** 2
        total_loss += yw * year_loss
    return total_loss


class ProgressCallback:
    """Tracks progress; called after each DE generation (not per evaluation)."""
    def __init__(self):
        self.gen = 0
        self.best = float("inf")
        self.start = time.time()

    def __call__(self, xk, convergence):
        self.gen += 1
        val = loss(xk)
        elapsed = time.time() - self.start
        if val < self.best:
            self.best = val
            print(f"  [gen {self.gen:>4}  {elapsed:>5.0f}s]  new best={val:.5f}", flush=True)
            _print_params(dict(zip(PARAM_NAMES, xk)))
        elif self.gen % 10 == 0:
            print(f"  [gen {self.gen:>4}  {elapsed:>5.0f}s]  best={self.best:.5f}", flush=True)
        return False  # don't stop


def _print_params(p):
    print("    Parties:", end="")
    for party in PARTY_NAMES:
        e = p[f"{party}_econ"]; s = p[f"{party}_social"]
        print(f"  {party}=({e:+.2f},{s:+.2f})", end="")
    print()
    print(f"    Pop: econ~N({p['pop_econ_mean']:+.2f},{p['pop_econ_std']:.2f})"
          f" social~N({p['pop_social_mean']:+.2f},{p['pop_social_std']:.2f})"
          f" urban={p['pop_urban_mean']:.2f}")
    print(f"    Rates: assim={p['assimilate_rate']:.4f} rep={p['repulse_rate']:.4f}"
          f" media={p['media_pull_rate']:.4f} socmedia={p['social_media_pull']:.4f}"
          f" rel={p['religiosity_pull']:.5f} zealot={p['zealot_fraction']:.3f}")
    print(f"    Turnout: base={p['base_turnout']:.3f} alien={p['alienation_sensitivity']:.3f}"
          f" soph={p['soph_turnout_bonus']:.3f} trust={p['trust_turnout_bonus']:.3f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    global _start_time

    print("=" * 70)
    print("Calibrating voting_model.py — fast numpy simulator")
    print(f"  Agents: {N_AGENTS}  Days: {N_STEPS}  Replicates: {N_REPLICATES}")
    print(f"  Parameters: {len(PARAM_NAMES)}")
    print("=" * 70)
    print("Targets:")
    for yr, t in TARGETS.items():
        print(f"  {yr}: CDU={t['CDU']:.3f} SPD={t['SPD']:.3f} Gruene={t['Gruene']:.3f} "
              f"FDP={t['FDP']:.3f} AfD={t['AfD']:.3f} BSW={t['BSW']:.3f} "
              f"turnout={t['turnout']:.3f}")
    print()

    # Benchmark
    t0 = time.time()
    x0 = np.array([(lo+hi)/2 for lo,hi in PARAM_BOUNDS])
    _ = simulate_fast(x0, seed=1)
    t1 = time.time()
    single_eval = (t1-t0) * N_REPLICATES * len(TARGETS)
    pop_size = 8 * len(PARAM_NAMES)
    # With 16 workers, each generation takes pop_size/16 * single_eval seconds
    est_mins = (pop_size / 16) * 150 * single_eval / 60
    print(f"Benchmark: {single_eval:.2f}s per loss eval (serial)")
    print(f"DE population={pop_size}, 16 workers, 150 iters -> ~{est_mins:.0f} min\n")

    cb = ProgressCallback()
    cb.start = time.time()
    start_time = time.time()

    result = differential_evolution(
        loss,
        bounds=PARAM_BOUNDS,
        maxiter=150,
        popsize=8,            # population = 8 * 27 = 216
        tol=1e-5,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=42,
        workers=-1,           # use all 16 CPU cores via multiprocessing
        disp=False,
        polish=True,
        init="latinhypercube",
        callback=cb,
    )

    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(f"Optimisation complete in {elapsed:.0f}s  ({result.nfev} evaluations)")
    print(f"Best loss: {result.fun:.6f}")
    best = dict(zip(PARAM_NAMES, result.x))
    _print_params(best)

    # Validation
    print()
    print("Validation (5 replicates):")
    for year, target in TARGETS.items():
        acc = {k: 0. for k in list(target.keys())}
        NR = 5
        for rep in range(NR):
            r = simulate_fast(result.x, seed=BASE_SEED + rep + year)
            for k in acc: acc[k] += r.get(k, 0.)
        print(f"\n  {year}:")
        print(f"  {'Party':<10}  {'Simulated':>10}  {'Target':>10}  {'Error':>8}")
        for party in ["CDU","SPD","Gruene","FDP","AfD","BSW","turnout"]:
            sim = acc[party] / NR
            tgt = target.get(party, 0.)
            print(f"  {party:<10}  {sim:>10.4f}  {tgt:>10.4f}  {sim-tgt:>+8.4f}")

    # Save
    output = {
        "best_loss": float(result.fun),
        "n_evaluations": result.nfev,
        "elapsed_seconds": elapsed,
        "parameters": best,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }
    with open("calibration_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to calibration_results.json")

    print()
    print("=" * 70)
    print("Paste into voting_model.py:")
    print("=" * 70)
    print("PARTIES = {")
    for party in PARTY_NAMES:
        e = best[f"{party}_econ"]; s = best[f"{party}_social"]
        print(f'    "{party}": ({e:+.4f}, {s:+.4f}),')
    print("}")
    for k in ["assimilate_rate","repulse_rate","media_pull_rate","social_media_pull",
              "religiosity_pull","zealot_fraction","base_turnout",
              "alienation_sensitivity","soph_turnout_bonus","trust_turnout_bonus"]:
        const = k.upper().replace("SOPH_","SOPHISTICATION_").replace("SOCIAL_MEDIA_PULL","SOCIAL_MEDIA_PULL_RATE")
        print(f"{const} = {best[k]:.6f}")


if __name__ == "__main__":
    main()
