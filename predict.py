"""
Forward election prediction using calibrated ABM parameters.

APPROACH
--------
Rather than fitting one static equilibrium, we run the simulation as a
true time-series:

  1. Start from historically-grounded ideology distributions for 2017
     (derived from the actual 2017 vote shares in the data).
  2. Run N_DAYS_PER_PERIOD days of opinion dynamics.
  3. Hold a simulated election -> produces 2021 predicted shares.
  4. Update the agent ideology distribution to reflect the 2021 outcome
     (winning-party supporters drift toward their party; losing-party
     supporters drift away).
  5. Repeat for 2025, and then project forward to a hypothetical 2029.

CALIBRATED PARAMETERS (best found: gen 35, loss=0.125)
-------------------------------------------------------
From calibrate_model.py run against federal_muni_harm_25.txt.
"""

import numpy as np
from collections import Counter

# ── CALIBRATED PARAMETERS ─────────────────────────────────────────────────────
PARTY_POS = {
    "CDU":    ( 0.23,  0.67),
    "SPD":    (-0.47, -0.16),
    "Gruene": (-0.28, -0.38),
    "FDP":    ( 0.46, -0.23),
    "AfD":    (-0.19,  0.97),
    "BSW":    (-0.42,  0.60),
}
PARTY_NAMES = list(PARTY_POS.keys())
PARTY_ARR   = np.array(list(PARTY_POS.values()))   # (6,2)

POP_ECON_STD    = 0.41
POP_SOCIAL_STD  = 0.18
POP_URBAN_MEAN  = 0.39

ASSIMILATE_RATE    = 0.0752
REPULSE_RATE       = 0.0149
MEDIA_PULL_RATE    = 0.0254
SOCIAL_MEDIA_PULL  = 0.0214
RELIGIOSITY_PULL   = 0.01631
ZEALOT_FRACTION    = 0.086
BASE_TURNOUT       = 0.782
ALIENATION_SENS    = 0.284
SOPH_TURNOUT_BONUS = 0.237
TRUST_TURNOUT_BONUS= 0.084

# ── FIXED CONSTANTS ───────────────────────────────────────────────────────────
N_AGENTS        = 500
N_REPLICATES    = 10       # more replicates for stable prediction intervals
N_DAYS          = 365 * 4  # ~4 years between elections
CONFIDENCE_INIT = 0.35
REPULSE_THRESH  = 0.65
NOISE_STD       = 0.01
VOTE_RAT        = 4.0
PARTY_ID_BONUS  = 0.6
PARTY_ID_RATE   = 0.02
PARTY_CUE_PULL  = 0.015
CROSS_DISCOUNT  = 0.4
MEAN_ENC        = 4
MEDIA_EVERY     = 4
SOC_MEDIA_THRESH= 0.6
MIN_SUSC        = 0.25
YOUNG_CUT       = 0.30
SEL_EXP_BETA    = 4.0
MEDIA_LEFT      = np.array([-0.75, -0.55])
MEDIA_RIGHT     = np.array([ 0.55,  0.55])
SECTOR_ECON_BIAS= np.array([-0.15, 0.05, 0.20, -0.10, 0.05, -0.10])
SECTOR_SHARES   = np.array([0.15, 0.45, 0.08, 0.10, 0.17, 0.05])

# ── ACTUAL ELECTION RESULTS (for comparison) ──────────────────────────────────
ACTUAL = {
    2017: {"CDU":0.3293,"SPD":0.2051,"Gruene":0.0894,"FDP":0.1075,
           "AfD":0.1264,"BSW":0.0000,"turnout":0.7642},
    2021: {"CDU":0.2407,"SPD":0.2574,"Gruene":0.1476,"FDP":0.1146,
           "AfD":0.1034,"BSW":0.0000,"turnout":0.7690},
    2025: {"CDU":0.2852,"SPD":0.1641,"Gruene":0.1161,"FDP":0.0433,
           "AfD":0.2080,"BSW":0.0498,"turnout":0.8268},
}


# ── IDEOLOGY STARTING POINT FROM VOTE SHARES ─────────────────────────────────
def ideology_mean_from_votes(vote_shares: dict) -> tuple:
    """
    Compute the implied population ideology centroid from observed vote shares
    by taking the vote-share-weighted average of party positions.
    This is the 'revealed preference' centre of gravity of the electorate.
    """
    econ  = sum(vote_shares.get(p, 0) * PARTY_POS[p][0] for p in PARTY_NAMES)
    social= sum(vote_shares.get(p, 0) * PARTY_POS[p][1] for p in PARTY_NAMES)
    return econ, social


# ── FAST NUMPY SIMULATOR ──────────────────────────────────────────────────────
def run_period(pop_econ_mean: float, pop_social_mean: float,
               n_days: int, seed: int) -> dict:
    """
    Run one inter-election period and return vote shares + turnout.
    Agents are initialised around (pop_econ_mean, pop_social_mean).
    """
    rng = np.random.default_rng(seed)
    N   = N_AGENTS

    age   = rng.random(N)
    edu   = rng.beta(2, 2, N)
    soph  = np.clip(0.6*edu + 0.4*rng.random(N), 0, 1)
    rel   = np.clip(rng.beta(1.5, 4, N), 0, 1)
    urb   = np.clip(rng.normal(POP_URBAN_MEAN, 0.2, N), 0, 1)
    trust = np.clip(rng.normal(0.5, 0.2, N), 0, 1)
    opens = np.clip(rng.normal(0.5, 0.2, N), 0, 1)
    socm  = np.where(age < 0.5,
                     np.clip(rng.beta(2, 3, N), 0, 1),
                     np.clip(rng.beta(1, 4, N), 0, 1))

    sec_idx  = rng.choice(len(SECTOR_SHARES), N, p=SECTOR_SHARES/SECTOR_SHARES.sum())
    sec_bias = SECTOR_ECON_BIAS[sec_idx]

    econ   = np.clip(rng.normal(pop_econ_mean  + sec_bias, POP_ECON_STD,  N), -1, 1)
    social = np.clip(rng.normal(pop_social_mean, POP_SOCIAL_STD, N), -1, 1)

    # Zealots near party anchors
    n_z = max(0, round(N * ZEALOT_FRACTION))
    z_mask = np.zeros(N, dtype=bool)
    if n_z > 0:
        zids = rng.choice(N, n_z, replace=False)
        z_mask[zids] = True
        anc = rng.integers(0, len(PARTY_NAMES), n_z)
        econ[zids]   = np.clip(PARTY_ARR[anc, 0] + rng.normal(0, 0.05, n_z), -1, 1)
        social[zids] = np.clip(PARTY_ARR[anc, 1] + rng.normal(0, 0.05, n_z), -1, 1)

    party_id = np.full(N, -1, dtype=int)
    pid_str  = np.zeros(N)
    non_z    = ~z_mask

    def susc():
        af = np.where(age <= YOUNG_CUT, 1.0,
                      1.0 - (1.0-MIN_SUSC)*(age-YOUNG_CUT)/(1-YOUNG_CUT))
        pr = 1.0 - 0.5*pid_str
        of = 0.7 + 0.6*opens
        s  = af * pr * of
        s[z_mask] = 0.
        return s

    def salience():
        base = 0.5 + 0.5*soph
        ep = np.abs(econ); sp = np.abs(social); tot = ep + sp + 1e-6
        we = base + (1-base)*(ep/tot)
        ws = base + (1-base)*(sp/tot)
        return we, ws

    for step in range(1, n_days+1):
        s = susc()
        n_pairs = max(1, int(N * MEAN_ENC / 2))
        ii = rng.integers(0, N, n_pairs)
        jj = rng.integers(0, N, n_pairs)
        ok = ii != jj
        ii, jj = ii[ok], jj[ok]
        if len(ii) == 0:
            continue

        de = econ[jj]   - econ[ii]
        ds = social[jj] - social[ii]
        d  = np.sqrt(de**2 + ds**2)

        cross = (party_id[ii]>=0)&(party_id[jj]>=0)&(party_id[ii]!=party_id[jj])
        disc  = np.where(cross, CROSS_DISCOUNT, 1.0)
        sa, sb = s[ii]*disc, s[jj]*disc

        am = d < CONFIDENCE_INIT
        if am.any():
            np.add.at(econ,   ii[am],  ASSIMILATE_RATE*sa[am]*de[am])
            np.add.at(social, ii[am],  ASSIMILATE_RATE*sa[am]*ds[am])
            np.add.at(econ,   jj[am], -ASSIMILATE_RATE*sb[am]*de[am])
            np.add.at(social, jj[am], -ASSIMILATE_RATE*sb[am]*ds[am])

        rm = d > REPULSE_THRESH
        if rm.any():
            dc = np.maximum(d[rm], 1e-6)
            np.add.at(econ,   ii[rm], -REPULSE_RATE/dc*sa[rm]*de[rm])
            np.add.at(social, ii[rm], -REPULSE_RATE/dc*sa[rm]*ds[rm])
            np.add.at(econ,   jj[rm],  REPULSE_RATE/dc*sb[rm]*de[rm])
            np.add.at(social, jj[rm],  REPULSE_RATE/dc*sb[rm]*ds[rm])

        econ   = np.clip(econ,   -1, 1)
        social = np.clip(social, -1, 1)

        econ[non_z]   = np.clip(econ[non_z]   + rng.normal(0, NOISE_STD, non_z.sum()), -1, 1)
        social[non_z] = np.clip(social[non_z] + rng.normal(0, NOISE_STD, non_z.sum()), -1, 1)

        sm = non_z & (socm >= SOC_MEDIA_THRESH)
        if sm.any():
            ss = s[sm]*socm[sm]
            ve = 0.7*econ[sm]   + 0.3*np.sign(econ[sm]+1e-9)
            vs = 0.7*social[sm] + 0.3*np.sign(social[sm]+1e-9)
            econ[sm]   = np.clip(econ[sm]   + SOCIAL_MEDIA_PULL*ss*(ve-econ[sm]),   -1, 1)
            social[sm] = np.clip(social[sm] + SOCIAL_MEDIA_PULL*ss*(vs-social[sm]), -1, 1)

        if step % MEDIA_EVERY == 0:
            pos = np.stack([econ, social], axis=1)
            dl = np.linalg.norm(pos - MEDIA_LEFT,  axis=1) - 0.15*(urb-0.5)
            dr = np.linalg.norm(pos - MEDIA_RIGHT, axis=1) + 0.15*(urb-0.5)
            ll = -dl*SEL_EXP_BETA; lr = -dr*SEL_EXP_BETA
            m  = np.maximum(ll, lr)
            pl = np.exp(ll-m)/(np.exp(ll-m)+np.exp(lr-m))
            cl = rng.random(N) < pl
            mx = np.where(cl, MEDIA_LEFT[0],  MEDIA_RIGHT[0])
            my = np.where(cl, MEDIA_LEFT[1],  MEDIA_RIGHT[1])
            econ[non_z]   = np.clip(econ[non_z]   + MEDIA_PULL_RATE*s[non_z]*(mx[non_z]-econ[non_z]),   -1, 1)
            social[non_z] = np.clip(social[non_z] + MEDIA_PULL_RATE*s[non_z]*(my[non_z]-social[non_z]), -1, 1)

        pull_r = RELIGIOSITY_PULL * rel * s
        social = np.clip(social + pull_r*(1.-social), -1, 1)

        has_pid = party_id >= 0
        if has_pid.any():
            px = PARTY_ARR[party_id[has_pid], 0]
            py = PARTY_ARR[party_id[has_pid], 1]
            pc = PARTY_CUE_PULL * pid_str[has_pid]
            econ[has_pid]   = np.clip(econ[has_pid]   + pc*(px-econ[has_pid]),   -1, 1)
            social[has_pid] = np.clip(social[has_pid] + pc*(py-social[has_pid]), -1, 1)

    # Election
    we, ws = salience()
    utils = np.zeros((N, len(PARTY_NAMES)))
    for k, pp in enumerate(PARTY_ARR):
        d = np.sqrt(we*(econ-pp[0])**2 + ws*(social-pp[1])**2)
        utils[:, k] = -d
    has_pid = party_id >= 0
    if has_pid.any():
        utils[has_pid, party_id[has_pid]] += PARTY_ID_BONUS * pid_str[has_pid]

    scaled = utils * VOTE_RAT
    scaled -= scaled.max(axis=1, keepdims=True)
    probs   = np.exp(scaled)
    probs  /= probs.sum(axis=1, keepdims=True)

    min_dist = np.sqrt(
        we[:,None]*(econ[:,None]-PARTY_ARR[:,0])**2 +
        ws[:,None]*(social[:,None]-PARTY_ARR[:,1])**2
    ).min(axis=1)
    to_prob = np.clip(BASE_TURNOUT - ALIENATION_SENS*min_dist
                      + SOPH_TURNOUT_BONUS*soph + TRUST_TURNOUT_BONUS*trust, 0.02, 0.98)

    voted = rng.random(N) < to_prob
    n_voted = voted.sum()
    if n_voted == 0:
        return {p: 1/len(PARTY_NAMES) for p in PARTY_NAMES} | {"turnout": 0.}

    cp = probs[voted].cumsum(axis=1)
    r  = rng.random(n_voted)[:, None]
    v  = (r < cp).argmax(axis=1)
    counts = np.bincount(v, minlength=len(PARTY_NAMES))
    shares = counts / n_voted
    result = {name: float(shares[k]) for k, name in enumerate(PARTY_NAMES)}
    result["turnout"] = float(n_voted / N)
    # Also return end-state ideology centroid for next period initialisation
    result["_econ_mean"]   = float(econ.mean())
    result["_social_mean"] = float(social.mean())
    return result


# ── MULTI-REPLICATE RUNNER ────────────────────────────────────────────────────
def predict_period(pop_econ_mean, pop_social_mean, n_days, base_seed, label):
    results = []
    for rep in range(N_REPLICATES):
        r = run_period(pop_econ_mean, pop_social_mean, n_days, seed=base_seed+rep)
        results.append(r)

    keys = PARTY_NAMES + ["turnout"]
    means = {k: float(np.mean([r[k] for r in results])) for k in keys}
    stds  = {k: float(np.std( [r[k] for r in results])) for k in keys}
    # Next period's starting ideology = mean of end-state centroids
    next_econ   = float(np.mean([r["_econ_mean"]   for r in results]))
    next_social = float(np.mean([r["_social_mean"] for r in results]))

    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"  Starting ideology: econ={pop_econ_mean:+.3f}, social={pop_social_mean:+.3f}")
    print(f"{'='*64}")
    print(f"  {'Party':<8}  {'Predicted':>9}  {'±95%CI':>8}", end="")
    if label.split()[0].isdigit() and int(label.split()[0]) in ACTUAL:
        print(f"  {'Actual':>8}  {'Error':>7}", end="")
    print()
    print(f"  {'-'*8}  {'-'*9}  {'-'*8}", end="")
    if label.split()[0].isdigit() and int(label.split()[0]) in ACTUAL:
        print(f"  {'-'*8}  {'-'*7}", end="")
    print()

    yr = None
    try:
        yr = int(label.split()[0])
    except Exception:
        pass

    for k in keys:
        m = means[k]; s = stds[k]
        ci = 1.96 * s
        line = f"  {k:<8}  {m:>8.1%}  ±{ci:>5.1%} "
        if yr and yr in ACTUAL:
            act = ACTUAL[yr].get(k, float('nan'))
            err = m - act
            line += f"  {act:>8.1%}  {err:>+6.1%}"
        print(line)

    print(f"\n  End-state centroid -> next period: econ={next_econ:+.3f}, social={next_social:+.3f}")
    return means, stds, next_econ, next_social


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("Forward election prediction — calibrated ABM")
    print(f"  {N_AGENTS} agents, {N_DAYS} days/period, {N_REPLICATES} replicates")
    print(f"  Party positions (calibrated):")
    for p, (e, s) in PARTY_POS.items():
        print(f"    {p:<8}: econ={e:+.2f}, social={s:+.2f}")

    # Step 0: derive 2017 starting ideology from actual 2017 vote shares
    e0, s0 = ideology_mean_from_votes(ACTUAL[2017])
    print(f"\n  2017 starting ideology (from actual vote shares): econ={e0:+.3f}, social={s0:+.3f}")

    # Period 1: 2017 -> 2021
    m21, sd21, e1, s1 = predict_period(e0, s0, N_DAYS, base_seed=100, label="2021 prediction")

    # Period 2: 2021 -> 2025 (start from end-state of previous period)
    m25, sd25, e2, s2 = predict_period(e1, s1, N_DAYS, base_seed=200, label="2025 prediction")

    # Period 3: 2025 -> 2029 (projection, no actual to compare)
    m29, sd29, e3, s3 = predict_period(e2, s2, N_DAYS, base_seed=300, label="2029 projection")

    # Summary table
    print(f"\n\n{'='*72}")
    print("SUMMARY — Predicted vs Actual vote shares")
    print(f"{'='*72}")
    header = f"  {'Party':<8}"
    for yr in [2021, 2025, "2029*"]:
        header += f"  {str(yr):>20}"
    print(header)
    sub = f"  {'':8}"
    for yr in ["Pred / Actual", "Pred / Actual", "Pred (projection)"]:
        sub += f"  {yr:>20}"
    print(sub)
    print(f"  {'-'*8}" + f"  {'-'*20}"*3)

    for k in PARTY_NAMES + ["turnout"]:
        row = f"  {k:<8}"
        for yr, m, sd in [(2021, m21, sd21), (2025, m25, sd25), (None, m29, sd29)]:
            pred = m[k]
            ci   = 1.96*sd[k]
            if yr in ACTUAL:
                act = ACTUAL[yr].get(k, float('nan'))
                cell = f"{pred:.1%}±{ci:.1%} / {act:.1%}"
            else:
                cell = f"{pred:.1%}±{ci:.1%}"
            row += f"  {cell:>20}"
        print(row)

    print(f"\n  * 2029 is a model projection with no ground truth yet.")
    print(f"    Confidence intervals reflect replicate variance only,")
    print(f"    not structural uncertainty in the model.")


if __name__ == "__main__":
    main()
