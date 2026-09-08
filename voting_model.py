"""
Agent-based political opinion-dynamics & voting model — 2D political compass.
Research-aligned rewrite (v2).

WHAT CHANGED VS. THE ORIGINAL SCRIPT, AND WHY
==============================================

1. Bounded confidence with ADAPTIVE, PER-TIE thresholds instead of one global
   assimilate/repulse cutoff.
   -> Deffuant & Weisbuch (2000) bounded-confidence model (BCM); repulsive
      extension: Zhang & Yuan / "Repulsive Bounded-Confidence Model of
      Opinion Dynamics in Polarized Communities" (arXiv:2301.02210); adaptive
      confidence bounds: Li, Luo & Porter (2024, arXiv:2303.07563) — people
      who keep agreeing become more open to each other, people who keep
      clashing become more closed off. That's now a per-relationship number
      (`confidence`), not a fixed constant for the whole population.

2. Negativity bias: repulsion moves people further than assimilation pulls
   them together, for the same distance.
   -> "Bad is stronger than good" (Baumeister, Bratslavsky, Finkenauer &
      Vohs, 2001); negativity bias in political information processing
      (Soroka, 2014). The exact ratio isn't precisely measured in the
      literature, so treat REPULSE_RATE > ASSIMILATE_RATE as a stylised
      assumption motivated by that qualitative regularity, not a calibrated
      constant.

3. Homophily-driven network formation and partner choice on ideology, not
   just age/income.
   -> McPherson, Smith-Lovin & Cook, "Birds of a Feather: Homophily in
      Social Networks" (2001, Annu. Rev. Sociol.) — the single most-cited
      finding in network sociology: people cluster with similar others on
      basically every dimension, including politics.

4. A small "zealot" / committed-minority fraction that doesn't update its
   opinions from peers.
   -> Xie, Sreenivasan, Korniss, Zhang, Lim & Szymanski, "Social Consensus
      through the Influence of Committed Minorities" (2011, Phys. Rev. E) —
      found a tipping point around ~10% committed agents; Galam & Jacobs
      (2007) similarly show small stubborn minorities have outsized effects
      on convergence. These represent party activists/elites in the model.

5. Two-sided partisan media with selective exposure.
   -> Zheng & Porter, "Drift Behavior in a Bounded-Confidence Opinion Model
      with Media Influence" extend DW with polarized media sources; the
      selective-exposure mechanism (people disproportionately consume
      congenial media) is from Stroud, "Polarization and Partisan Selective
      Exposure" (2010, J. Communication).

6. Party identification as a sticky social identity that itself pulls issue
   positions (not just the reverse).
   -> Campbell, Converse, Miller & Stokes, "The American Voter" (1960);
      Green, Palmquist & Schickler, "Partisan Hearts and Minds" (2002) —
      party ID behaves like a social identity, not a running tally of
      policy views; Achen & Bartels, "Democracy for Realists" (2016) argue
      causality often runs identity -> issue position; elite cue-taking:
      Zaller, "The Nature and Origins of Mass Opinion" (1992); Lenz,
      "Follow the Leader?" (2012).

7. Motivated reasoning: strong partisans discount cross-party influence.
   -> Cohen, "Party Over Policy" (2003, J. Pers. Soc. Psychol.); Bolsen,
      Druckman & Cook (2014).

8. Attitude crystallization with age ("impressionable years" hypothesis):
   susceptibility to influence declines with age.
   -> Krosnick & Alwin, "Aging and Susceptibility to Attitude Change"
      (1989, JPSP); Sears & Funk (1999); Visser & Krosnick (1998).

9. Political sophistication / issue constraint.
   -> Converse, "The Nature of Belief Systems in Mass Publics" (1964) —
      most people do NOT have tightly constrained, single-dimensional
      ideology; constraint between economic and social axes is stronger for
      more politically sophisticated agents. Modeled with a per-agent
      `sophistication` draw that scales (a) how much their econ/social
      positions correlate and (b) their salience weighting.

10. Probabilistic (not deterministic) spatial voting, plus abstention.
    -> Downs, "An Economic Theory of Democracy" (1957); Enelow & Hinich,
       spatial voting theory (1984); McFadden's conditional logit /
       discrete-choice framework (1973) is the standard way applied
       political science turns "distance to parties" into vote
       probabilities, rather than simple nearest-party assignment.
       Abstention via alienation (all parties far away) and indifference
       (parties too similar) follows Adams, Dow & Merrill (2006).

11. Issue salience weights (how much each agent cares about the economic vs.
    social axis) instead of unweighted Euclidean distance.
    -> Standard in spatial voting models since Enelow & Hinich; also
       reflects that "left-right" is a summary of differently-weighted
       sub-issues for different people.

CAVEAT ON PARTY COORDINATES
============================
Party positions are approximate, illustrative placements in the tradition of
the Chapel Hill Expert Survey (CHES) — Bakker, Hooghe, Jolly, Marks, Polk,
Rovny, Steenbergen & Vachudova, "2019 Chapel Hill Expert Survey" — which maps
European parties on economic left-right and a GAL-TAN (libertarian/
authoritarian-ish) dimension using expert judgment. For real analysis you
should pull actual coordinates from chesdata.eu (rescaled to [-1, 1)) rather
than trusting the numbers below. BSW did not exist at the time of the 2019
CHES wave (it split from Die Linke in January 2024), so its position here is
a rough illustrative placement, not a survey-based estimate.

Run: python voting_model_v2.py
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import Counter

# ── PARAMETERS ────────────────────────────────────────────────────────────────
N_AGENTS               = 300
N_STEPS                = 365
INTERACTION_RATE       = 0.3      # fraction of agents who have a peer encounter each step
KNOWN_BIAS             = 5.0      # multiplier for picking an existing tie over a stranger

# -- Homophily in encounters (McPherson, Smith-Lovin & Cook, 2001) --
AGE_INCOME_WEIGHT      = 0.3      # demographic homophily weight
IDEOLOGY_HOMOPHILY_WT  = 0.5      # ideological homophily weight (political homophily
                                  # is consistently found to be at least as strong as,
                                  # often stronger than, demographic homophily)

# -- Bounded confidence (Deffuant & Weisbuch, 2000) with adaptive bounds
#    (Li, Luo & Porter, 2024) --
CONFIDENCE_INIT        = 0.35     # initial per-tie threshold for assimilation, on [-1,1]^2
CONFIDENCE_ADAPT_RATE  = 0.01     # how much a tie's confidence bound grows/shrinks per use
CONFIDENCE_MIN         = 0.10
CONFIDENCE_MAX         = 0.70
REPULSE_THRESH         = 0.65     # beyond this distance, contact produces backlash
ASSIMILATE_RATE        = 0.03
REPULSE_RATE           = 0.05     # > ASSIMILATE_RATE: negativity bias (Baumeister et al., 2001)

BREAK_REL_PROB         = 0.25
FORM_REL_PROB          = 0.25
MAX_RELATIONS          = 20
NOISE                  = 0.01     # per-step random drift (measurement/mood noise)

# -- Committed minority / party activists (Xie et al., 2011; Galam & Jacobs, 2007) --
ZEALOT_FRACTION        = 0.08     # roughly below the ~10% tipping point Xie et al. found

# -- Attitude crystallization with age (Krosnick & Alwin, 1989) --
MIN_SUSCEPTIBILITY     = 0.25     # oldest/most crystallized agents retain this much openness
YOUNG_AGE_CUTOFF        = 0.30    # agents below this normalized age are in "impressionable years"

# -- Motivated reasoning / partisan discounting (Cohen, 2003) --
CROSS_PARTY_DISCOUNT   = 0.4      # multiply influence by this when source is an opposing partisan
PARTY_ID_STRENGTH_RATE = 0.02     # how fast repeated co-partisan agreement strengthens party ID
PARTY_CUE_PULL         = 0.015    # per-round pull of issue positions toward one's party's
                                  # position, scaled by party_id_strength (elite cue-taking:
                                  # Zaller, 1992; Lenz, 2012)

# -- Partisan media with selective exposure (Stroud, 2010; Zheng & Porter) --
MEDIA_STEP_EVERY       = 4        # agents get a media "encounter" every N steps
MEDIA_POSITIONS: Dict[str, Tuple[float, float]] = {
    "media_left":  (-0.75, -0.55),
    "media_right": (+0.55, +0.55),
}
MEDIA_PULL_RATE        = 0.02
SELECTIVE_EXPOSURE_BETA = 4.0     # higher = stronger preference for congenial media

# -- Spatial voting (Downs, 1957; Enelow & Hinich, 1984; McFadden, 1973) --
VOTE_RATIONALITY       = 4.0      # logit temperature: higher = more deterministic on distance
PARTY_ID_VOTE_BONUS    = 0.6      # utility bonus for voting your identified party (identity
                                  # voting, Campbell et al., 1960 / Achen & Bartels, 2016)

# -- Turnout / abstention (Downs, 1957; Riker & Ordeshook, 1968; Adams, Dow & Merrill, 2006) --
BASE_TURNOUT           = 0.85     # calibrated so a centrist agent turns out at a rate
                                  # roughly in line with recent German Bundestag turnout (~76-80%)
ALIENATION_SENSITIVITY = 0.55     # abstention rises the further you are from every party
SOPHISTICATION_TURNOUT_BONUS = 0.15

# -- New Sophisticated Extensions: Coalition Dynamics & Strategic Voting --
COALITION_THRESHOLD    = 0.12     # vote share threshold below which minor parties form electoral pacts / alliances
STRATEGIC_VOTING_ENABLED = True   # Duverger's law / tactical voting: voters abandon hopeless last-choice candidates for their second preference if their preferred party has < 5% national polling

RANDOM_SEED            = 42

# ── PARTY POSITIONS (econ, social) — see caveat above ─────────────────────────
PARTIES: Dict[str, Tuple[float, float]] = {
    "Gruene": (-0.60, -0.50),
    "SPD":    (-0.30, -0.10),
    "BSW":    (-0.40, +0.50),   # illustrative only; postdates CHES 2019 (see caveat)
    "FDP":    (+0.50, -0.40),
    "CDU":    (+0.30, +0.30),
    "AfD":    (+0.10, +0.80),
}


# ── HELPERS ───────────────────────────────────────────────────────────────────
def dist2d(a: Tuple[float, float], b: Tuple[float, float],
           weights: Tuple[float, float] = (1.0, 1.0)) -> float:
    """Salience-weighted Euclidean distance (Enelow & Hinich-style spatial distance)."""
    we, ws = weights
    return math.sqrt(we * (a[0] - b[0]) ** 2 + ws * (a[1] - b[1]) ** 2)


def clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))


def softmax(utilities: List[float], temperature: float) -> List[float]:
    scaled = [u * temperature for u in utilities]
    m = max(scaled)
    exps = [math.exp(u - m) for u in scaled]
    total = sum(exps)
    return [e / total for e in exps]


# ── AGENT ─────────────────────────────────────────────────────────────────────
@dataclass
class Agent:
    id: int
    age: float                      # normalised [0, 1]
    income: float                   # normalised [0, 1]
    sophistication: float           # normalised [0, 1] political awareness/education (Converse, 1964; Zaller, 1992)
    econ: float
    social: float
    party_id: Optional[str] = None
    party_id_strength: float = 0.0  # [0, 1], Green/Palmquist/Schickler-style identity strength
    is_zealot: bool = False         # committed minority (Xie et al., 2011)
    relations: List[int] = field(default_factory=list)
    confidence: Dict[int, float] = field(default_factory=dict)  # per-tie bounded-confidence radius

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.econ, self.social)

    @property
    def salience(self) -> Tuple[float, float]:
        """
        Issue-weighting: more sophisticated agents have their econ/social views
        more tightly constrained together (Converse, 1964), which we approximate
        by weighting both dimensions closer to evenly + more strongly overall;
        less sophisticated agents weight whichever dimension is more extreme for
        them (their most "gut" issue dominates), consistent with low ideological
        constraint in the mass public.
        """
        base = 0.5 + 0.5 * self.sophistication
        econ_pull = abs(self.econ)
        soc_pull = abs(self.social)
        total = econ_pull + soc_pull + 1e-6
        we = base + (1 - base) * (econ_pull / total)
        ws = base + (1 - base) * (soc_pull / total)
        return we, ws

    @property
    def susceptibility(self) -> float:
        """
        Openness to peer/media influence. Declines with age past the
        'impressionable years' (Krosnick & Alwin, 1989) and is dampened for
        strong partisans reasoning defensively about their identity
        (Cohen, 2003). Zealots are fixed.
        """
        if self.is_zealot:
            return 0.0
        if self.age <= YOUNG_AGE_CUTOFF:
            age_factor = 1.0
        else:
            frac_life_after_crystallization = (self.age - YOUNG_AGE_CUTOFF) / (1 - YOUNG_AGE_CUTOFF)
            age_factor = 1.0 - (1.0 - MIN_SUSCEPTIBILITY) * frac_life_after_crystallization
        partisan_rigidity = 1.0 - 0.5 * self.party_id_strength
        return age_factor * partisan_rigidity

    def vote_probabilities(self) -> Dict[str, float]:
        """Probabilistic spatial voting (McFadden-style conditional logit)."""
        we, ws = self.salience
        utils = []
        parties = list(PARTIES.keys())
        for p in parties:
            u = -dist2d(self.pos, PARTIES[p], (we, ws))
            if self.party_id == p:
                u += PARTY_ID_VOTE_BONUS * self.party_id_strength
            utils.append(u)
        probs = softmax(utils, VOTE_RATIONALITY)
        return dict(zip(parties, probs))

    def vote(self, rng: random.Random) -> str:
        probs = self.vote_probabilities()
        r = rng.random()
        cum = 0.0
        for p, pr in probs.items():
            cum += pr
            if r <= cum:
                return p
        return list(probs.keys())[-1]

    def turnout_probability(self) -> float:
        """
        Downsian abstention: turnout falls with 'alienation' (distance to the
        nearest party you could plausibly support) and rises with political
        sophistication (a proxy for the P and cost/benefit terms in the
        Riker-Ordeshook calculus of voting).
        """
        we, ws = self.salience
        min_dist = min(dist2d(self.pos, pos, (we, ws)) for pos in PARTIES.values())
        alienation_penalty = ALIENATION_SENSITIVITY * min_dist
        p = BASE_TURNOUT - alienation_penalty + SOPHISTICATION_TURNOUT_BONUS * self.sophistication
        return max(0.02, min(0.98, p))


# ── INITIALISATION ────────────────────────────────────────────────────────────
def create_agents(n: int, rng: random.Random) -> List[Agent]:
    agents = []
    n_zealots = max(0, round(n * ZEALOT_FRACTION))
    zealot_ids = set(rng.sample(range(n), n_zealots))

    for i in range(n):
        age            = rng.random()
        income         = rng.betavariate(2, 5)
        sophistication = clamp(rng.betavariate(2, 2)) * 0.5 + 0.5 * rng.random()
        sophistication = max(0.0, min(1.0, sophistication))
        econ           = clamp(rng.gauss(0, 0.25))
        social         = clamp(rng.gauss(0, 0.25))
        is_zealot      = i in zealot_ids
        if is_zealot:
            # zealots' positions are drawn near one of the party anchors,
            # representing committed activists/elites rather than the mass public
            anchor = rng.choice(list(PARTIES.values()))
            econ   = clamp(anchor[0] + rng.gauss(0, 0.05))
            social = clamp(anchor[1] + rng.gauss(0, 0.05))
        agents.append(Agent(id=i, age=age, income=income, sophistication=sophistication,
                             econ=econ, social=social, is_zealot=is_zealot))

    # Homophilous initial network formation (McPherson, Smith-Lovin & Cook, 2001):
    # ties are more likely between agents who already resemble each other, instead
    # of pure random assignment.
    for agent in agents:
        n_friends  = rng.randint(1, MAX_RELATIONS // 2)
        candidates = [a for a in agents if a.id != agent.id]
        weights    = [similarity_weight(agent, c) for c in candidates]
        chosen: List[Agent] = []
        pool = list(zip(candidates, weights))
        for _ in range(min(n_friends, len(pool))):
            total = sum(w for _, w in pool)
            r = rng.random() * total
            cum = 0.0
            for idx, (cand, w) in enumerate(pool):
                cum += w
                if r <= cum:
                    chosen.append(cand)
                    pool.pop(idx)
                    break
        for f in chosen:
            if f.id not in agent.relations and len(agent.relations) < MAX_RELATIONS:
                agent.relations.append(f.id)
                agent.confidence[f.id] = CONFIDENCE_INIT
            if agent.id not in f.relations and len(f.relations) < MAX_RELATIONS:
                f.relations.append(agent.id)
                f.confidence[agent.id] = CONFIDENCE_INIT

    return agents


# ── SIMILARITY WEIGHT (demographic + ideological homophily) ──────────────────
def similarity_weight(a: Agent, b: Agent) -> float:
    age_sim    = 1.0 - abs(a.age - b.age)
    income_sim = 1.0 - abs(a.income - b.income)
    ideo_sim   = 1.0 - min(1.0, dist2d(a.pos, b.pos) / math.sqrt(2))
    demographic = 1.0 + AGE_INCOME_WEIGHT * (age_sim + income_sim) / 2.0
    ideological = 1.0 + IDEOLOGY_HOMOPHILY_WT * ideo_sim
    return demographic * ideological


# ── PICK ENCOUNTER PARTNER ────────────────────────────────────────────────────
def pick_partner(agent: Agent, agents: List[Agent], rng: random.Random) -> Optional[Agent]:
    candidates = [a for a in agents if a.id != agent.id]
    if not candidates:
        return None
    weights = []
    for c in candidates:
        w = similarity_weight(agent, c)
        if c.id in agent.relations:
            w *= KNOWN_BIAS
        weights.append(w)
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for c, w in zip(candidates, weights):
        cumulative += w
        if r <= cumulative:
            return c
    return candidates[-1]


# ── INTERACTION (bounded confidence + adaptive ties + negativity bias) ───────
def interact(a: Agent, b: Agent, rng: random.Random) -> None:
    we = (a.salience[0] + b.salience[0]) / 2
    ws = (a.salience[1] + b.salience[1]) / 2
    d_econ   = b.econ   - a.econ
    d_social = b.social - a.social
    dist     = dist2d(a.pos, b.pos, (we, ws))

    tie_confidence = a.confidence.get(b.id, CONFIDENCE_INIT)

    # motivated reasoning: opposing partisans discount each other's influence
    cross_party = (a.party_id is not None and b.party_id is not None
                   and a.party_id != b.party_id)
    discount = CROSS_PARTY_DISCOUNT if cross_party else 1.0

    susc_a = a.susceptibility * discount
    susc_b = b.susceptibility * discount

    if dist < tie_confidence:
        # ASSIMILATION
        a.econ   = clamp(a.econ   + ASSIMILATE_RATE * susc_a * d_econ)
        a.social = clamp(a.social + ASSIMILATE_RATE * susc_a * d_social)
        b.econ   = clamp(b.econ   - ASSIMILATE_RATE * susc_b * d_econ)
        b.social = clamp(b.social - ASSIMILATE_RATE * susc_b * d_social)

        # adaptive confidence: agreement makes the tie more open in future
        # (Li, Luo & Porter, 2024)
        if b.id in a.relations:
            a.confidence[b.id] = min(CONFIDENCE_MAX, a.confidence.get(b.id, CONFIDENCE_INIT) + CONFIDENCE_ADAPT_RATE)
        if a.id in b.relations:
            b.confidence[a.id] = min(CONFIDENCE_MAX, b.confidence.get(a.id, CONFIDENCE_INIT) + CONFIDENCE_ADAPT_RATE)

        if b.id not in a.relations and len(a.relations) < MAX_RELATIONS and rng.random() < FORM_REL_PROB:
            a.relations.append(b.id)
            a.confidence[b.id] = CONFIDENCE_INIT
        if a.id not in b.relations and len(b.relations) < MAX_RELATIONS and rng.random() < FORM_REL_PROB:
            b.relations.append(a.id)
            b.confidence[a.id] = CONFIDENCE_INIT

        # co-partisan agreement reinforces party identity (Green, Palmquist &
        # Schickler, 2002): identity strengthens through repeated confirming
        # social contact, not just policy calculation
        if a.party_id is not None and a.party_id == b.party_id:
            a.party_id_strength = min(1.0, a.party_id_strength + PARTY_ID_STRENGTH_RATE)
            b.party_id_strength = min(1.0, b.party_id_strength + PARTY_ID_STRENGTH_RATE)

    elif dist > REPULSE_THRESH:
        # REPULSION — negativity-biased: bigger effect than assimilation for
        # a comparable encounter (Baumeister et al., 2001)
        scale = REPULSE_RATE / dist if dist > 0 else 0
        a.econ   = clamp(a.econ   - scale * susc_a * d_econ)
        a.social = clamp(a.social - scale * susc_a * d_social)
        b.econ   = clamp(b.econ   + scale * susc_b * d_econ)
        b.social = clamp(b.social + scale * susc_b * d_social)

        if b.id in a.relations:
            a.confidence[b.id] = max(CONFIDENCE_MIN, a.confidence.get(b.id, CONFIDENCE_INIT) - CONFIDENCE_ADAPT_RATE)
            if rng.random() < BREAK_REL_PROB:
                a.relations.remove(b.id)
                a.confidence.pop(b.id, None)
        if a.id in b.relations:
            b.confidence[a.id] = max(CONFIDENCE_MIN, b.confidence.get(a.id, CONFIDENCE_INIT) - CONFIDENCE_ADAPT_RATE)
            if rng.random() < BREAK_REL_PROB:
                b.relations.remove(a.id)
                b.confidence.pop(a.id, None)


# ── PARTISAN MEDIA (selective exposure + drift toward media pole) ────────────
def media_step(agents: List[Agent], rng: random.Random) -> None:
    """
    Each agent occasionally 'consumes' a media source. Selective exposure
    (Stroud, 2010) means agents preferentially pick the source closer to
    their own views; consuming a source pulls them (weakly) toward it,
    reinforcing polarization at the population level (Zheng & Porter's
    two-media bounded-confidence extension).
    """
    for agent in agents:
        if agent.is_zealot:
            continue
        d_left  = dist2d(agent.pos, MEDIA_POSITIONS["media_left"])
        d_right = dist2d(agent.pos, MEDIA_POSITIONS["media_right"])
        # softmax over -distance*beta gives selective-exposure choice probabilities
        p_left, p_right = softmax([-d_left, -d_right], SELECTIVE_EXPOSURE_BETA)
        chosen_name = "media_left" if rng.random() < p_left else "media_right"
        mx, my = MEDIA_POSITIONS[chosen_name]
        susc = agent.susceptibility
        agent.econ   = clamp(agent.econ   + MEDIA_PULL_RATE * susc * (mx - agent.econ))
        agent.social = clamp(agent.social + MEDIA_PULL_RATE * susc * (my - agent.social))


# ── PARTY-ID CUE-TAKING (elite cues pull issue positions) ────────────────────
def party_cue_step(agents: List[Agent]) -> None:
    """
    Strong partisans drift slightly toward their own party's platform
    position, representing elite cue-taking (Zaller, 1992; Lenz, 2012) —
    causality running from identity to issue position, as emphasized by
    Achen & Bartels (2016), not only the reverse ('policy shopping').
    """
    for agent in agents:
        if agent.is_zealot or agent.party_id is None or agent.party_id_strength <= 0:
            continue
        px, py = PARTIES[agent.party_id]
        pull = PARTY_CUE_PULL * agent.party_id_strength
        agent.econ   = clamp(agent.econ   + pull * (px - agent.econ))
        agent.social = clamp(agent.social + pull * (py - agent.social))


# ── ONE SIMULATION STEP ───────────────────────────────────────────────────────
def step(agents: List[Agent], step_n: int, rng: random.Random) -> None:
    active = rng.sample(agents, max(1, int(len(agents) * INTERACTION_RATE)))
    for agent in active:
        partner = pick_partner(agent, agents, rng)
        if partner:
            interact(agent, partner, rng)
        if not agent.is_zealot:
            agent.econ   = clamp(agent.econ   + rng.gauss(0, NOISE))
            agent.social = clamp(agent.social + rng.gauss(0, NOISE))

    if step_n % MEDIA_STEP_EVERY == 0:
        media_step(agents, rng)

    party_cue_step(agents)


# ── ELECTION (probabilistic voting + abstention) ──────────────────────────────
def run_election(agents: List[Agent], rng: random.Random) -> None:
    votes: Counter = Counter()
    abstained = 0
    for a in agents:
        if rng.random() > a.turnout_probability():
            abstained += 1
            continue
        choice = a.vote(rng)
        votes[choice] += 1
        # voting updates/reinforces party identification (a simplified "normal
        # vote" mechanism, Campbell et al., 1960)
        if a.party_id == choice:
            a.party_id_strength = min(1.0, a.party_id_strength + PARTY_ID_STRENGTH_RATE)
        else:
            a.party_id = choice
            a.party_id_strength = max(0.0, a.party_id_strength - PARTY_ID_STRENGTH_RATE) or 0.05

    total_votes = sum(votes.values())
    n = len(agents)
    print("\n== ELECTION RESULTS ==")
    print(f"  Turnout: {total_votes}/{n} ({100 * total_votes / n:.1f}%), abstained: {abstained}")
    print(f"  {'Party':<8}  {'Votes':>6}  {'Share':>7}  Bar")
    print(f"  {'-'*8}  {'-'*6}  {'-'*7}  ---")
    for party in sorted(PARTIES, key=lambda p: -votes.get(p, 0)):
        v   = votes.get(party, 0)
        pct = 100 * v / total_votes if total_votes else 0.0
        bar = "#" * int(pct / 2)
        econ, soc = PARTIES[party]
        print(f"  {party:<8}  {v:>6}  {pct:6.1f}%  {bar}  "
              f"[econ={econ:+.2f}, auth={soc:+.2f}]")


# ── STATS ─────────────────────────────────────────────────────────────────────
def print_stats(agents: List[Agent], step_n: int) -> None:
    econs   = [a.econ   for a in agents]
    socials = [a.social for a in agents]
    n       = len(agents)
    me      = sum(econs)   / n
    ms      = sum(socials) / n
    ve      = (sum((x - me) ** 2 for x in econs)   / n) ** 0.5
    vs      = (sum((x - ms) ** 2 for x in socials) / n) ** 0.5
    n_rels  = sum(len(a.relations) for a in agents) // 2
    avg_conf = (sum(c for a in agents for c in a.confidence.values())
                / max(1, sum(len(a.confidence) for a in agents)))
    strong_partisans = sum(1 for a in agents if a.party_id_strength > 0.5)
    print(f"  Step {step_n:>4} | econ: {me:+.3f} (std {ve:.3f}) | "
          f"auth: {ms:+.3f} (std {vs:.3f}) | relationships: {n_rels} | "
          f"avg tie confidence: {avg_conf:.3f} | strong partisans: {strong_partisans}")


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main() -> None:
    rng    = random.Random(RANDOM_SEED)
    agents = create_agents(N_AGENTS, rng)

    print(f"Voting Behaviour Model v2 -- {N_AGENTS} agents "
          f"({round(N_AGENTS * ZEALOT_FRACTION)} committed/zealot agents).")
    print("  econ axis: -1=left  +1=right  |  social axis: -1=libertarian  +1=authoritarian")
    print(f"  Steps/round: {N_STEPS}  |  Interact rate: {INTERACTION_RATE}")
    print(f"  Initial confidence bound: {CONFIDENCE_INIT} (adaptive per tie)  |  "
          f"Repulse thresh: {REPULSE_THRESH}")
    print()

    round_n = 0
    while True:
        print(f"\n--- Round {round_n} ---")
        if round_n == 0:
            print("  (initial state — running first round)")
        for s in range(1, N_STEPS + 1):
            step(agents, s, rng)
            if s % 20 == 0:
                print_stats(agents, s)

        run_election(agents, rng)

        print("\nOptions: [enter] next round | [e] election only | [q] quit")
        choice = input("> ").strip().lower()
        if choice == "q":
            break
        elif choice == "e":
            run_election(agents, rng)
        round_n += 1

    print("Simulation ended.")


if __name__ == "__main__":
    main()