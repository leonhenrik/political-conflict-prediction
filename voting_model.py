"""
Agent-based political opinion-dynamics & voting model — 2D political compass.
Research-aligned rewrite (v3): daily schedule, institutions, richer agent attributes.

WHAT'S IN THIS VERSION (v3 additions on top of v2)
===================================================

DAILY ENCOUNTER STRUCTURE
--------------------------
Each step represents one day. Agents draw a Poisson number of encounters split
across three social contexts:
  • Workplace (within the same institution): coworkers meet at higher rate.
    Blau (1977) "Inequality and Heterogeneity" — structural position shapes
    contact opportunity; Festinger, Schachter & Back (1950) "Social Pressures
    in Informal Groups" — proximity creates ties.
  • Neighbourhood (agents sharing the same residential area): local residential
    context moderates exposure (Walks, 2006, "The Causes of City-Suburban
    Political Polarization").
  • Open encounter (homophily-weighted random, as before).

JOBS & INSTITUTIONS
--------------------
Each agent is assigned a job sector (public, private, self-employed, student,
retired, unemployed) and, within employed/student sectors, an institution ID
(a shared workplace/school). Institution size follows a power law; agents at
the same institution share a social context that inflates encounter probability
by COWORKER_BIAS. Sector also shifts the prior on economic axis
(Kitschelt & Rehm, 2014, "Occupational Class as a Driver of Political
Preferences in Affluent Capitalist Societies").

NEW AGENT ATTRIBUTES
---------------------
• education:      [0,1]. Higher education → more liberal on social axis on average
                  (Stubager, 2008, "Education Effects on Authoritarian-Libertarian
                  Values"); also raises sophistication.
• religiosity:    [0,1]. Higher → pull toward socially conservative positions
                  (Norris & Inglehart, 2004, "Sacred and Secular"; Inglehart &
                  Welzel, 2005, "Modernization, Cultural Change, and Democracy").
• urban_rural:    [0,1], 0=deep rural, 1=dense urban. Urban → more left-libertarian
                  on average; also determines neighbourhood context and media ecology
                  (Scala & Johnson, 2017, "Political Polarization along the
                  Urban-Rural Continuum").
• trust:          [0,1]. Civic/institutional trust. High trust → higher turnout
                  (Putnam, 2000, "Bowling Alone"); low trust → susceptibility to
                  populist framing (Norris & Inglehart, 2019, "Cultural Backlash").
• openness:       [0,1]. Big-Five Openness to Experience. High openness correlates
                  with wider confidence bounds and more assimilation; low openness
                  → narrower, more defensive engagement (Carney et al., 2008,
                  "The Secret Lives of Liberals and Conservatives"; Jost et al.,
                  2003, "Political Conservatism as Motivated Social Cognition").
• social_media:   [0,1]. Frequency of social-media use. High use → more frequent
                  media-like encounters with algorithmically curated content (echo
                  chambers: Bail et al., 2018, "Exposure to Opposing Views on
                  Social Media Can Increase Political Polarization", PNAS; but see
                  also Guess et al., 2023 for nuanced effects). Modelled as an
                  extra partial-media pull each day.

INTERACTION EFFECTS (new)
---------------------------
• Religiosity × social axis: each day, religiosity exerts a small conservative
  pull on the social axis (Norris & Inglehart), scaled by susceptibility.
• Urban/rural × media: urban agents have a slightly higher base left-libertarian
  media diet; rural agents tilt toward right-authoritarian media (Scala & Johnson,
  2017).
• Trust × turnout: low-trust agents have lower baseline turnout (Putnam, 2000).
• Openness → confidence bounds: open agents start with a wider per-tie threshold;
  closed agents start narrower and resist widening (Carney et al., 2008).
• Education × sophistication: education directly scales the sophistication draw,
  keeping them correlated but not identical (Converse, 1964; Zaller, 1992).
• Social media → daily echo-chamber pull: heavy social-media users get an extra
  daily encounter with a same-side "virtual" media source (Bail et al., 2018).

UNCHANGED FROM v2
------------------
Bounded confidence with adaptive per-tie thresholds (Li, Luo & Porter, 2024);
negativity bias (Baumeister et al., 2001); homophily-driven network formation
(McPherson, Smith-Lovin & Cook, 2001); zealots (Xie et al., 2011); partisan media
with selective exposure (Stroud, 2010); party ID as social identity (Green,
Palmquist & Schickler, 2002; Achen & Bartels, 2016); motivated reasoning (Cohen,
2003); attitude crystallization (Krosnick & Alwin, 1989); issue salience (Enelow
& Hinich, 1984); probabilistic spatial voting + abstention (McFadden, 1973;
Downs, 1957; Adams, Dow & Merrill, 2006).

CAVEAT ON PARTY COORDINATES
============================
See v2 docstring. Unchanged here.

Run: python voting_model.py
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import Counter, defaultdict

# ── PARAMETERS ────────────────────────────────────────────────────────────────
N_AGENTS               = 300
N_STEPS                = 365          # each step = one day
N_INSTITUTIONS         = 30           # number of distinct workplaces/schools

# -- Daily encounter budget --
MEAN_ENCOUNTERS_PER_DAY = 5           # Poisson λ for total daily encounters
WORKPLACE_SHARE         = 0.40        # fraction of encounters that happen at work
NEIGHBOURHOOD_SHARE     = 0.25        # fraction in the residential neighbourhood
OPEN_SHARE              = 0.35        # fraction via the general homophily pool
COWORKER_BIAS           = 8.0         # multiplier for same-institution candidates
NEIGHBOUR_BIAS          = 4.0         # multiplier for same-neighbourhood candidates
N_NEIGHBOURHOODS        = 10          # residential neighbourhood buckets

KNOWN_BIAS              = 5.0         # multiplier for existing ties over strangers

# -- Homophily in encounters (McPherson, Smith-Lovin & Cook, 2001) --
AGE_INCOME_WEIGHT       = 0.3
IDEOLOGY_HOMOPHILY_WT   = 0.5

# -- Bounded confidence (Deffuant & Weisbuch, 2000; adaptive: Li, Luo & Porter, 2024) --
CONFIDENCE_INIT         = 0.35
CONFIDENCE_ADAPT_RATE   = 0.01
CONFIDENCE_MIN          = 0.10
CONFIDENCE_MAX          = 0.80        # openness personality can push this up
REPULSE_THRESH          = 0.65
ASSIMILATE_RATE         = 0.03
REPULSE_RATE            = 0.05        # > ASSIMILATE_RATE: negativity bias

BREAK_REL_PROB          = 0.25
FORM_REL_PROB           = 0.20
MAX_RELATIONS           = 25
NOISE                   = 0.01

# -- Committed minority (Xie et al., 2011; Galam & Jacobs, 2007) --
ZEALOT_FRACTION         = 0.08

# -- Attitude crystallization (Krosnick & Alwin, 1989) --
MIN_SUSCEPTIBILITY      = 0.25
YOUNG_AGE_CUTOFF        = 0.30

# -- Motivated reasoning (Cohen, 2003) --
CROSS_PARTY_DISCOUNT    = 0.4
PARTY_ID_STRENGTH_RATE  = 0.02
PARTY_CUE_PULL          = 0.015

# -- Partisan media (Stroud, 2010; Zheng & Porter) --
MEDIA_STEP_EVERY        = 4           # traditional/broadcast media every N days
MEDIA_POSITIONS: Dict[str, Tuple[float, float]] = {
    "media_left":  (-0.75, -0.55),
    "media_right": (+0.55, +0.55),
}
MEDIA_PULL_RATE         = 0.02
SELECTIVE_EXPOSURE_BETA = 4.0

# -- Social-media echo-chamber (Bail et al., 2018) --
SOCIAL_MEDIA_PULL_RATE  = 0.015      # pull toward virtual same-side source per day
SOCIAL_MEDIA_THRESHOLD  = 0.6        # agents above this score get the daily echo pull

# -- Religiosity pull on social axis (Norris & Inglehart, 2004) --
RELIGIOSITY_PULL        = 0.008      # daily pull toward positive social axis, scaled by religiosity

# -- Spatial voting (Downs, 1957; Enelow & Hinich, 1984; McFadden, 1973) --
VOTE_RATIONALITY        = 4.0
PARTY_ID_VOTE_BONUS     = 0.6

# -- Turnout / abstention (Putnam, 2000; Downs, 1957; Adams, Dow & Merrill, 2006) --
BASE_TURNOUT            = 0.85
ALIENATION_SENSITIVITY  = 0.55
SOPHISTICATION_TURNOUT_BONUS = 0.15
TRUST_TURNOUT_BONUS     = 0.10       # Putnam (2000): civic trust raises participation

RANDOM_SEED             = 42

# ── JOB SECTORS ───────────────────────────────────────────────────────────────
# Sector shares (approximate German labour-force structure)
SECTORS = ["public", "private", "self_employed", "student", "retired", "unemployed"]
SECTOR_SHARES = [0.15, 0.45, 0.08, 0.10, 0.17, 0.05]

# Economic-axis prior per sector (Kitschelt & Rehm, 2014): public-sector workers
# and students lean left; private-sector and self-employed lean right.
SECTOR_ECON_BIAS: Dict[str, float] = {
    "public":       -0.15,
    "private":       0.05,
    "self_employed": 0.20,
    "student":      -0.10,
    "retired":       0.05,
    "unemployed":   -0.10,
}

# Sectors that can have a shared institution (workplace/school)
EMPLOYED_SECTORS = {"public", "private", "self_employed", "student"}

# ── PARTY POSITIONS (econ, social) ────────────────────────────────────────────
PARTIES: Dict[str, Tuple[float, float]] = {
    "Gruene": (-0.60, -0.50),
    "SPD":    (-0.30, -0.10),
    "BSW":    (-0.40, +0.50),
    "FDP":    (+0.50, -0.40),
    "CDU":    (+0.30, +0.30),
    "AfD":    (+0.10, +0.80),
}


# ── HELPERS ───────────────────────────────────────────────────────────────────
def dist2d(a: Tuple[float, float], b: Tuple[float, float],
           weights: Tuple[float, float] = (1.0, 1.0)) -> float:
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


def poisson_draw(lam: float, rng: random.Random) -> int:
    """Knuth's algorithm for small Poisson λ."""
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def weighted_choice(pool: List, weights: List[float], rng: random.Random):
    total = sum(weights)
    if total <= 0:
        return rng.choice(pool) if pool else None
    r = rng.random() * total
    cum = 0.0
    for item, w in zip(pool, weights):
        cum += w
        if r <= cum:
            return item
    return pool[-1]


# ── AGENT ─────────────────────────────────────────────────────────────────────
@dataclass
class Agent:
    id: int
    age: float                      # [0,1]
    income: float                   # [0,1]
    sophistication: float           # [0,1] — correlated with education
    education: float                # [0,1] — years of education, normalised
    religiosity: float              # [0,1] — religious practice/salience
    urban_rural: float              # [0,1], 0=rural, 1=urban
    trust: float                    # [0,1] — institutional/civic trust
    openness: float                 # [0,1] — Big-Five openness to experience
    social_media: float             # [0,1] — daily social-media intensity
    econ: float
    social: float
    sector: str = "private"
    institution_id: Optional[int] = None   # shared workplace/school ID
    neighbourhood_id: int = 0              # residential neighbourhood bucket
    party_id: Optional[str] = None
    party_id_strength: float = 0.0
    is_zealot: bool = False
    relations: List[int] = field(default_factory=list)
    confidence: Dict[int, float] = field(default_factory=dict)

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.econ, self.social)

    @property
    def salience(self) -> Tuple[float, float]:
        base = 0.5 + 0.5 * self.sophistication
        econ_pull = abs(self.econ)
        soc_pull  = abs(self.social)
        total     = econ_pull + soc_pull + 1e-6
        we = base + (1 - base) * (econ_pull / total)
        ws = base + (1 - base) * (soc_pull  / total)
        return we, ws

    @property
    def confidence_cap(self) -> float:
        """Openness raises the ceiling on how wide a tie's confidence bound can grow."""
        return CONFIDENCE_MIN + (CONFIDENCE_MAX - CONFIDENCE_MIN) * (0.4 + 0.6 * self.openness)

    @property
    def susceptibility(self) -> float:
        if self.is_zealot:
            return 0.0
        if self.age <= YOUNG_AGE_CUTOFF:
            age_factor = 1.0
        else:
            frac = (self.age - YOUNG_AGE_CUTOFF) / (1 - YOUNG_AGE_CUTOFF)
            age_factor = 1.0 - (1.0 - MIN_SUSCEPTIBILITY) * frac
        partisan_rigidity = 1.0 - 0.5 * self.party_id_strength
        # Openness (Carney et al., 2008): more open = more permeable
        openness_factor   = 0.7 + 0.6 * self.openness
        return age_factor * partisan_rigidity * openness_factor

    def vote_probabilities(self) -> Dict[str, float]:
        we, ws = self.salience
        parties = list(PARTIES.keys())
        utils = []
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
        we, ws = self.salience
        min_dist = min(dist2d(self.pos, pos, (we, ws)) for pos in PARTIES.values())
        alienation_penalty = ALIENATION_SENSITIVITY * min_dist
        trust_bonus        = TRUST_TURNOUT_BONUS * self.trust
        p = (BASE_TURNOUT - alienation_penalty
             + SOPHISTICATION_TURNOUT_BONUS * self.sophistication
             + trust_bonus)
        return max(0.02, min(0.98, p))


# ── INITIALISATION ────────────────────────────────────────────────────────────
def _draw_sector(rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for s, share in zip(SECTORS, SECTOR_SHARES):
        cum += share
        if r <= cum:
            return s
    return SECTORS[-1]


def create_agents(n: int, rng: random.Random) -> List[Agent]:
    agents: List[Agent] = []
    n_zealots   = max(0, round(n * ZEALOT_FRACTION))
    zealot_ids  = set(rng.sample(range(n), n_zealots))

    # Institution size distribution: power-law-ish via random draw
    institution_sizes: Dict[int, int] = defaultdict(int)

    for i in range(n):
        age        = rng.random()
        income     = rng.betavariate(2, 5)
        education  = clamp(rng.betavariate(2, 2))  # slightly left-skewed
        # sophistication correlated with education + own random component
        sophistication = max(0.0, min(1.0, 0.6 * education + 0.4 * rng.random()))
        religiosity    = max(0.0, min(1.0, rng.betavariate(1.5, 4)))  # right-skewed; most not very religious
        urban_rural    = clamp(rng.betavariate(2, 2))
        trust          = max(0.0, min(1.0, rng.gauss(0.5, 0.2)))
        openness       = max(0.0, min(1.0, rng.gauss(0.5, 0.2)))
        social_media   = max(0.0, min(1.0, rng.betavariate(2, 3) if age < 0.5 else rng.betavariate(1, 4)))

        # Starting ideology: centred + sector bias + urban-rural pull + education pull
        # Urban → more left/libertarian (Scala & Johnson, 2017)
        # Education → more libertarian on social axis (Stubager, 2008)
        sector = _draw_sector(rng)
        sector_bias = SECTOR_ECON_BIAS.get(sector, 0.0)
        econ   = clamp(rng.gauss(sector_bias, 0.25) - 0.10 * (urban_rural - 0.5))
        social = clamp(rng.gauss(0, 0.25)
                       - 0.20 * (urban_rural - 0.5)    # urban → more libertarian
                       - 0.15 * (education - 0.5)      # educated → more libertarian
                       + 0.15 * (religiosity - 0.3))   # religious → more conservative

        is_zealot = i in zealot_ids
        if is_zealot:
            anchor = rng.choice(list(PARTIES.values()))
            econ   = clamp(anchor[0] + rng.gauss(0, 0.05))
            social = clamp(anchor[1] + rng.gauss(0, 0.05))

        # Assign institution (only for employed/student sectors)
        institution_id = None
        if sector in EMPLOYED_SECTORS:
            # Prefer smaller institutions with a slight power-law: draw from
            # a non-uniform categorical that over-represents low IDs
            inst = int(rng.betavariate(1, 3) * N_INSTITUTIONS)
            institution_id = max(0, min(N_INSTITUTIONS - 1, inst))
            institution_sizes[institution_id] += 1

        neighbourhood_id = int(urban_rural * (N_NEIGHBOURHOODS - 1) + rng.gauss(0, 0.5))
        neighbourhood_id = max(0, min(N_NEIGHBOURHOODS - 1, neighbourhood_id))

        # Initial per-tie confidence bound scaled by openness
        init_conf = CONFIDENCE_INIT + 0.10 * (openness - 0.5)

        agents.append(Agent(
            id=i, age=age, income=income, sophistication=sophistication,
            education=education, religiosity=religiosity, urban_rural=urban_rural,
            trust=trust, openness=openness, social_media=social_media,
            econ=econ, social=social, sector=sector, institution_id=institution_id,
            neighbourhood_id=neighbourhood_id, is_zealot=is_zealot,
        ))
        # Store init_conf so we set it after all agents exist
        agents[-1]._init_conf = init_conf  # type: ignore[attr-defined]

    # Build lookup tables for fast encounter candidate selection
    inst_members:   Dict[Optional[int], List[Agent]] = defaultdict(list)
    neigh_members:  Dict[int, List[Agent]]            = defaultdict(list)
    for a in agents:
        inst_members[a.institution_id].append(a)
        neigh_members[a.neighbourhood_id].append(a)

    # Homophilous initial network
    for agent in agents:
        n_friends  = rng.randint(1, MAX_RELATIONS // 2)
        candidates = [a for a in agents if a.id != agent.id]
        weights    = [similarity_weight(agent, c) for c in candidates]
        pool       = list(zip(candidates, weights))
        chosen: List[Agent] = []
        for _ in range(min(n_friends, len(pool))):
            total = sum(w for _, w in pool)
            r     = rng.random() * total
            cum   = 0.0
            for idx, (cand, w) in enumerate(pool):
                cum += w
                if r <= cum:
                    chosen.append(cand)
                    pool.pop(idx)
                    break
        init_c = getattr(agent, '_init_conf', CONFIDENCE_INIT)
        for f in chosen:
            if f.id not in agent.relations and len(agent.relations) < MAX_RELATIONS:
                agent.relations.append(f.id)
                agent.confidence[f.id] = init_c
            if agent.id not in f.relations and len(f.relations) < MAX_RELATIONS:
                f.relations.append(agent.id)
                f.confidence[agent.id] = getattr(f, '_init_conf', CONFIDENCE_INIT)

    return agents, inst_members, neigh_members


# ── SIMILARITY WEIGHT ─────────────────────────────────────────────────────────
def similarity_weight(a: Agent, b: Agent) -> float:
    age_sim    = 1.0 - abs(a.age    - b.age)
    income_sim = 1.0 - abs(a.income - b.income)
    ideo_sim   = 1.0 - min(1.0, dist2d(a.pos, b.pos) / math.sqrt(2))
    demographic = 1.0 + AGE_INCOME_WEIGHT * (age_sim + income_sim) / 2.0
    ideological = 1.0 + IDEOLOGY_HOMOPHILY_WT * ideo_sim
    return demographic * ideological


# ── PICK ENCOUNTER PARTNER ────────────────────────────────────────────────────
def pick_from_pool(agent: Agent, pool: List[Agent], extra_bias_fn,
                   rng: random.Random) -> Optional[Agent]:
    """Generic weighted pick from a pool with homophily + known-ties + optional extra bias."""
    candidates = [a for a in pool if a.id != agent.id]
    if not candidates:
        return None
    weights = []
    for c in candidates:
        w = similarity_weight(agent, c)
        if c.id in agent.relations:
            w *= KNOWN_BIAS
        if extra_bias_fn is not None:
            w *= extra_bias_fn(c)
        weights.append(w)
    return weighted_choice(candidates, weights, rng)


def pick_workplace_partner(agent: Agent,
                            inst_members: Dict[Optional[int], List[Agent]],
                            agents: List[Agent],
                            rng: random.Random) -> Optional[Agent]:
    """Prefer a coworker; fall back to general pool."""
    if agent.institution_id is not None:
        pool = inst_members[agent.institution_id]
        partner = pick_from_pool(agent, pool, None, rng)
        if partner:
            return partner
    # fallback: treat as open encounter
    return pick_from_pool(agent, agents, None, rng)


def pick_neighbour_partner(agent: Agent,
                            neigh_members: Dict[int, List[Agent]],
                            agents: List[Agent],
                            rng: random.Random) -> Optional[Agent]:
    pool    = neigh_members[agent.neighbourhood_id]
    partner = pick_from_pool(agent, pool, None, rng)
    return partner if partner else pick_from_pool(agent, agents, None, rng)


def pick_open_partner(agent: Agent, agents: List[Agent], rng: random.Random) -> Optional[Agent]:
    return pick_from_pool(agent, agents, None, rng)


# ── INTERACTION ───────────────────────────────────────────────────────────────
def interact(a: Agent, b: Agent, rng: random.Random) -> None:
    we = (a.salience[0] + b.salience[0]) / 2
    ws = (a.salience[1] + b.salience[1]) / 2
    d_econ   = b.econ   - a.econ
    d_social = b.social - a.social
    dist     = dist2d(a.pos, b.pos, (we, ws))

    tie_confidence = a.confidence.get(b.id, CONFIDENCE_INIT)
    cross_party    = (a.party_id is not None and b.party_id is not None
                      and a.party_id != b.party_id)
    discount = CROSS_PARTY_DISCOUNT if cross_party else 1.0
    susc_a   = a.susceptibility * discount
    susc_b   = b.susceptibility * discount

    if dist < tie_confidence:
        # ASSIMILATION
        a.econ   = clamp(a.econ   + ASSIMILATE_RATE * susc_a * d_econ)
        a.social = clamp(a.social + ASSIMILATE_RATE * susc_a * d_social)
        b.econ   = clamp(b.econ   - ASSIMILATE_RATE * susc_b * d_econ)
        b.social = clamp(b.social - ASSIMILATE_RATE * susc_b * d_social)

        cap_a = a.confidence_cap
        cap_b = b.confidence_cap
        if b.id in a.relations:
            a.confidence[b.id] = min(cap_a, a.confidence.get(b.id, CONFIDENCE_INIT) + CONFIDENCE_ADAPT_RATE)
        if a.id in b.relations:
            b.confidence[a.id] = min(cap_b, b.confidence.get(a.id, CONFIDENCE_INIT) + CONFIDENCE_ADAPT_RATE)

        if b.id not in a.relations and len(a.relations) < MAX_RELATIONS and rng.random() < FORM_REL_PROB:
            a.relations.append(b.id)
            a.confidence[b.id] = CONFIDENCE_INIT
        if a.id not in b.relations and len(b.relations) < MAX_RELATIONS and rng.random() < FORM_REL_PROB:
            b.relations.append(a.id)
            b.confidence[a.id] = CONFIDENCE_INIT

        if a.party_id is not None and a.party_id == b.party_id:
            a.party_id_strength = min(1.0, a.party_id_strength + PARTY_ID_STRENGTH_RATE)
            b.party_id_strength = min(1.0, b.party_id_strength + PARTY_ID_STRENGTH_RATE)

    elif dist > REPULSE_THRESH:
        # REPULSION — negativity bias (Baumeister et al., 2001)
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


# ── PARTISAN MEDIA ────────────────────────────────────────────────────────────
def media_step(agents: List[Agent], rng: random.Random) -> None:
    for agent in agents:
        if agent.is_zealot:
            continue
        # Urban-rural shifts which media pole is more accessible
        urban_tilt = 0.15 * (agent.urban_rural - 0.5)   # urban tilts left media
        d_left  = dist2d(agent.pos, MEDIA_POSITIONS["media_left"])  - urban_tilt
        d_right = dist2d(agent.pos, MEDIA_POSITIONS["media_right"]) + urban_tilt
        p_left, p_right = softmax([-d_left, -d_right], SELECTIVE_EXPOSURE_BETA)
        chosen_name = "media_left" if rng.random() < p_left else "media_right"
        mx, my = MEDIA_POSITIONS[chosen_name]
        susc = agent.susceptibility
        agent.econ   = clamp(agent.econ   + MEDIA_PULL_RATE * susc * (mx - agent.econ))
        agent.social = clamp(agent.social + MEDIA_PULL_RATE * susc * (my - agent.social))


def social_media_step(agent: Agent, rng: random.Random) -> None:
    """
    Heavy social-media users get a daily pull toward a virtual same-side source,
    simulating algorithmic curation / echo chambers (Bail et al., 2018).
    """
    if agent.is_zealot or agent.social_media < SOCIAL_MEDIA_THRESHOLD:
        return
    susc = agent.susceptibility * agent.social_media
    # Virtual source is on the same side of the econ axis as the agent
    virtual_econ   = 0.7 * agent.econ   + 0.3 * (-1 if agent.econ < 0 else 1)
    virtual_social = 0.7 * agent.social + 0.3 * (-1 if agent.social < 0 else 1)
    agent.econ   = clamp(agent.econ   + SOCIAL_MEDIA_PULL_RATE * susc * (virtual_econ   - agent.econ))
    agent.social = clamp(agent.social + SOCIAL_MEDIA_PULL_RATE * susc * (virtual_social - agent.social))


# ── RELIGIOSITY PULL ──────────────────────────────────────────────────────────
def religiosity_step(agents: List[Agent]) -> None:
    """Daily pull toward socially conservative (positive) pole, scaled by religiosity."""
    for agent in agents:
        if agent.is_zealot:
            continue
        pull = RELIGIOSITY_PULL * agent.religiosity * agent.susceptibility
        agent.social = clamp(agent.social + pull * (1.0 - agent.social))


# ── PARTY-ID CUE-TAKING ───────────────────────────────────────────────────────
def party_cue_step(agents: List[Agent]) -> None:
    for agent in agents:
        if agent.is_zealot or agent.party_id is None or agent.party_id_strength <= 0:
            continue
        px, py = PARTIES[agent.party_id]
        pull   = PARTY_CUE_PULL * agent.party_id_strength
        agent.econ   = clamp(agent.econ   + pull * (px - agent.econ))
        agent.social = clamp(agent.social + pull * (py - agent.social))


# ── ONE SIMULATION STEP (= ONE DAY) ──────────────────────────────────────────
def step(agents: List[Agent],
         inst_members: Dict[Optional[int], List[Agent]],
         neigh_members: Dict[int, List[Agent]],
         step_n: int, rng: random.Random) -> None:

    for agent in agents:
        n_encounters = max(1, poisson_draw(MEAN_ENCOUNTERS_PER_DAY, rng))
        n_work  = round(n_encounters * WORKPLACE_SHARE)
        n_neigh = round(n_encounters * NEIGHBOURHOOD_SHARE)
        n_open  = n_encounters - n_work - n_neigh

        for _ in range(n_work):
            partner = pick_workplace_partner(agent, inst_members, agents, rng)
            if partner:
                interact(agent, partner, rng)

        for _ in range(n_neigh):
            partner = pick_neighbour_partner(agent, neigh_members, agents, rng)
            if partner:
                interact(agent, partner, rng)

        for _ in range(max(0, n_open)):
            partner = pick_open_partner(agent, agents, rng)
            if partner:
                interact(agent, partner, rng)

        if not agent.is_zealot:
            agent.econ   = clamp(agent.econ   + rng.gauss(0, NOISE))
            agent.social = clamp(agent.social + rng.gauss(0, NOISE))
            social_media_step(agent, rng)

    if step_n % MEDIA_STEP_EVERY == 0:
        media_step(agents, rng)

    religiosity_step(agents)
    party_cue_step(agents)


# ── ELECTION ──────────────────────────────────────────────────────────────────
def run_election(agents: List[Agent], rng: random.Random) -> None:
    votes: Counter = Counter()
    abstained = 0
    for a in agents:
        if rng.random() > a.turnout_probability():
            abstained += 1
            continue
        choice = a.vote(rng)
        votes[choice] += 1
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
    me  = sum(econs)   / n
    ms  = sum(socials) / n
    ve  = (sum((x - me) ** 2 for x in econs)   / n) ** 0.5
    vs  = (sum((x - ms) ** 2 for x in socials) / n) ** 0.5
    n_rels  = sum(len(a.relations) for a in agents) // 2
    avg_conf = (sum(c for a in agents for c in a.confidence.values())
                / max(1, sum(len(a.confidence) for a in agents)))
    strong_partisans = sum(1 for a in agents if a.party_id_strength > 0.5)
    print(f"  Day {step_n:>4} | econ: {me:+.3f} (std {ve:.3f}) | "
          f"auth: {ms:+.3f} (std {vs:.3f}) | "
          f"rels: {n_rels} | conf: {avg_conf:.3f} | "
          f"strong partisans: {strong_partisans}")


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main() -> None:
    rng = random.Random(RANDOM_SEED)
    agents, inst_members, neigh_members = create_agents(N_AGENTS, rng)

    sector_counts = Counter(a.sector for a in agents)
    print(f"Voting Behaviour Model v3 -- {N_AGENTS} agents, {N_STEPS} days/round")
    print(f"  Sectors: {dict(sector_counts)}")
    print(f"  econ axis: -1=left  +1=right  |  social axis: -1=libertarian  +1=authoritarian")
    print(f"  {round(N_AGENTS * ZEALOT_FRACTION)} zealots | "
          f"{N_INSTITUTIONS} institutions | {N_NEIGHBOURHOODS} neighbourhoods")
    print(f"  ~{MEAN_ENCOUNTERS_PER_DAY} encounters/day/agent "
          f"({int(WORKPLACE_SHARE*100)}% work, {int(NEIGHBOURHOOD_SHARE*100)}% neighbourhood, "
          f"{int(OPEN_SHARE*100)}% open)")
    print()

    round_n = 0
    while True:
        print(f"\n--- Round {round_n} ---")
        for s in range(1, N_STEPS + 1):
            step(agents, inst_members, neigh_members, s, rng)
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
