#!/usr/bin/env python3
"""
Forecast 8 — Warsh Fed: P(no balance-sheet emergency intervention | CRISIS)
CONDITIONAL FORECAST. Crisis is assumed, not simulated.
All probabilities are P(response | crisis already occurring).
"""
import os, sys, warnings, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from datetime import datetime
import urllib.request, ssl, io

warnings.filterwarnings('ignore')
os.makedirs('out', exist_ok=True)

# ── light theme ──────────────────────────────────────────────────────────────
TEXT   = '#212121'
GRID_C = '#E0E0E0'
GREEN  = '#2E7D32'
RED    = '#C62828'
ORANGE = '#E65100'
ACCENT = '#1565C0'
MUTED  = '#546E7A'
GOLD   = '#E65100'   # same as ORANGE — amber/burnt-orange for Warsh bars
PURPLE = '#6A1B9A'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.facecolor': 'white', 'figure.facecolor': 'white',
    'axes.edgecolor': '#BDBDBD',
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': GRID_C, 'grid.linewidth': 0.6, 'grid.alpha': 1.0,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'axes.labelcolor': TEXT, 'text.color': TEXT,
    'legend.framealpha': 0.9, 'legend.edgecolor': '#BDBDBD',
})

# ═══════════════════════════════════════════════════════════════════════════
# FRAMING — PRINT PROMINENTLY
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 74)
print("FORECAST 8 — WARSH FED: RESPONSE TO A CRISIS (CONDITIONAL FORECAST)")
print("=" * 74)
print()
print("*** CRISIS IS ASSUMED. THIS IS A CONDITIONAL FORECAST. ***")
print("All probabilities are P(response | crisis) where crisis = NFCI > 1.0")
print("AND significant credit spread blowout, a la 2008/2020/2023.")
print("The question is NOT whether a crisis occurs. It is:")
print("  -> GIVEN a crisis, does Warsh deploy RATE CUTS (Tool A)?")
print("  -> GIVEN a crisis, does Warsh deploy BALANCE SHEET (Tool B)?")
print("A prior version modelled stress probability and got ~84% 'no emergency'")
print("with ~69% of that from 'no stress in the first place.' That was wrong.")
print("This version isolates the RESPONSE conditional on stress.")
print()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: FRED DATA — WALCL, FEDFUNDS, NFCI
# ═══════════════════════════════════════════════════════════════════════════

def fetch_fred_csv(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=15) as r:
            raw = r.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(raw))
        df.columns = [c.strip() for c in df.columns]
        date_col = next(c for c in df.columns if 'date' in c.lower())
        val_col  = next(c for c in df.columns if c != date_col)
        df = df.rename(columns={date_col: 'date', val_col: series_id})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        df = df.dropna().sort_values('date').reset_index(drop=True)
        print(f"  [FRED] {series_id}: {len(df)} obs, {df['date'].min().date()} to {df['date'].max().date()}")
        return df
    except Exception as e:
        print(f"  [FRED] {series_id} fetch failed ({type(e).__name__})")
        return None

print("Fetching FRED data...")
df_walcl  = fetch_fred_csv('WALCL')    # Fed balance sheet, weekly, $bn
df_ff     = fetch_fred_csv('FEDFUNDS') # Effective fed funds rate, monthly
df_nfci   = fetch_fred_csv('NFCI')     # National Financial Conditions Index, weekly

# ─── Hardcode fallback if fetch fails ──────────────────────────────────────
if df_walcl is None:
    print("  Using hardcoded WALCL skeleton (key episodes only)")
    WALCL_ROWS = [
        ('2006-01-01', 880), ('2007-01-01', 900), ('2008-01-01', 920),
        ('2008-10-01', 1800), ('2009-01-01', 2100), ('2010-01-01', 2300),
        ('2011-01-01', 2900), ('2012-01-01', 2900), ('2013-01-01', 3000),
        ('2014-01-01', 4200), ('2015-01-01', 4500), ('2016-01-01', 4500),
        ('2017-01-01', 4400), ('2018-01-01', 4400), ('2019-01-01', 4000),
        ('2019-10-01', 3700), ('2020-01-01', 4100), ('2020-04-01', 6700),
        ('2020-07-01', 7100), ('2021-01-01', 7400), ('2022-01-01', 8900),
        ('2022-06-01', 9000), ('2023-01-01', 8500), ('2023-03-01', 8700),
        ('2024-01-01', 7700), ('2024-12-01', 7000),
    ]
    df_walcl = pd.DataFrame(WALCL_ROWS, columns=['date', 'WALCL'])
    df_walcl['date'] = pd.to_datetime(df_walcl['date'])

if df_nfci is None:
    print("  Using hardcoded NFCI skeleton")
    NFCI_ROWS = [
        ('2006-01-01', -0.6), ('2007-01-01', -0.5), ('2007-07-01', 0.0),
        ('2008-03-01', 0.8),  ('2008-10-01', 2.8),  ('2009-01-01', 2.5),
        ('2009-06-01', 0.5),  ('2010-01-01', -0.2), ('2020-01-01', -0.7),
        ('2020-03-20', 3.1),  ('2020-06-01', 0.5),  ('2021-01-01', -0.5),
        ('2023-01-01', -0.1), ('2023-03-17', 0.7),  ('2023-06-01', -0.2),
        ('2024-01-01', -0.4), ('2024-12-01', -0.2),
    ]
    df_nfci = pd.DataFrame(NFCI_ROWS, columns=['date', 'NFCI'])
    df_nfci['date'] = pd.to_datetime(df_nfci['date'])

if df_ff is None:
    print("  Using hardcoded FEDFUNDS skeleton")
    FF_ROWS = [
        ('2006-01-01', 4.29), ('2007-01-01', 5.25), ('2007-09-01', 4.94),
        ('2008-01-01', 3.94), ('2008-10-01', 1.0),  ('2008-12-01', 0.16),
        ('2009-01-01', 0.12), ('2015-12-01', 0.37), ('2018-12-01', 2.40),
        ('2019-07-01', 2.40), ('2020-01-01', 1.55), ('2020-03-01', 0.65),
        ('2020-04-01', 0.05), ('2022-01-01', 0.08), ('2022-12-01', 4.24),
        ('2023-01-01', 4.33), ('2023-03-01', 4.57), ('2024-01-01', 5.33),
        ('2024-12-01', 4.48),
    ]
    df_ff = pd.DataFrame(FF_ROWS, columns=['date', 'FEDFUNDS'])
    df_ff['date'] = pd.to_datetime(df_ff['date'])

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1b: HISTORICAL STRESS EPISODES
# ═══════════════════════════════════════════════════════════════════════════

# Documented historical episodes with response classification
# Sources: Fed H.4.1 releases, FOMC minutes, academic papers
EPISODES = [
    {
        'name':       '2008 GFC',
        'onset':      '2008-09-15',  # Lehman
        'peak_nfci':  2.96,          # NFCI weekly peak ~Oct 2008
        'notes':      'NFCI peak ~2.96 week of 2008-10-10; Lehman triggered full cascade',
        # Fed response within 2 quarters (~6 months):
        'rate_cut_A': True,
        'rate_cut_amt_pp': 175,       # Oct-Dec 2008: 75+50+50bp → 0.25% target
        'rate_cut_weeks':  12,        # first cut within 3 weeks
        'bs_deploy_B':  True,
        'bs_tool':      'TARP + CPFF + MMIFF + TALF + LSAP1',
        'bs_expand_bn': 1200,         # rough $1.2T expansion in 6 months
        'bs_weeks':     6,            # first major facility within 6 weeks
        'color':        RED,
    },
    {
        'name':       '2020 COVID',
        'onset':      '2020-03-03',   # emergency cut
        'peak_nfci':  3.14,           # NFCI peak ~3.14 week of 2020-03-27
        'notes':      'NFCI peak ~3.14 week of 2020-03-27; fastest-ever Fed response',
        'rate_cut_A': True,
        'rate_cut_amt_pp': 150,       # 50+100bp in March 2020
        'rate_cut_weeks':  1,
        'bs_deploy_B':  True,
        'bs_tool':      'Unlimited QE (LSAP4) + 9 new 13(3) facilities',
        'bs_expand_bn': 2700,         # ~$2.7T expansion in 6 months
        'bs_weeks':     2,
        'color':        ORANGE,
    },
    {
        'name':       '2023 SVB/BTFP',
        'onset':      '2023-03-10',   # SVB failure
        'peak_nfci':  0.69,           # mild relative to 2008/2020; NFCI ~0.69
        'notes':      'NFCI peaked ~0.69; regional bank stress, not systemic; BTFP deployed in 72hrs',
        'rate_cut_A': False,          # no rate cut — hiking cycle continued
        'rate_cut_amt_pp': 0,
        'rate_cut_weeks':  None,
        'bs_deploy_B':  True,
        'bs_tool':      'BTFP (Bank Term Funding Program) + expanded discount window',
        'bs_expand_bn': 400,          # ~$400B BTFP + discount window combined
        'bs_weeks':     1,            # BTFP stood up in 72 hours
        'color':        PURPLE,
    },
]

print()
print("=" * 74)
print("HISTORICAL STRESS EPISODES — Tool Response Classification")
print("=" * 74)
for ep in EPISODES:
    print(f"\n{ep['name']}  (onset: {ep['onset']})")
    print(f"  NFCI peak: {ep['peak_nfci']:.2f}  |  {ep['notes']}")
    ra = f"+{ep['rate_cut_amt_pp']}bp cut, {ep['rate_cut_weeks']}wk" if ep['rate_cut_A'] else "NONE (hiking cycle continued)"
    print(f"  Tool A (rate cuts):      {'YES' if ep['rate_cut_A'] else 'NO '} — {ra}")
    bs = f"{ep['bs_tool']}; +${ep['bs_expand_bn']}bn in {ep['bs_weeks']}wk" if ep['bs_deploy_B'] else "NO"
    print(f"  Tool B (balance sheet):  {'YES' if ep['bs_deploy_B'] else 'NO '} — {bs}")

# Empirical baseline
n_ep = len(EPISODES)
p_base_A = sum(1 for e in EPISODES if e['rate_cut_A']) / n_ep
p_base_B = sum(1 for e in EPISODES if e['bs_deploy_B']) / n_ep

print()
print(f"Empirical baseline (n={n_ep} episodes):")
print(f"  P(deploy Tool A — rate cuts  | crisis): {p_base_A:.2f}  ({sum(1 for e in EPISODES if e['rate_cut_A'])}/{n_ep})")
print(f"  P(deploy Tool B — balance sh | crisis): {p_base_B:.2f}  ({sum(1 for e in EPISODES if e['bs_deploy_B'])}/{n_ep})")
print()
print("Note: Small sample (n=3). Bayesian prior: typical chair nearly always")
print("deploys balance-sheet tools when NFCI breaches the 1.0 threshold.")
print("The 2023 SVB episode (NFCI ~0.69, below the 1.0 threshold) is included")
print("for context but is borderline — BTFP was deployed pre-emptively.")
print()

# Bayesian adjustment: with only 3 episodes, add a prior
# Prior: P(deploy B | NFCI>1.0) ~ Beta(8, 1) centered near 0.90
# Posterior with 3/3 hits: Beta(8+3, 1+0) → mean = 11/12 = 0.917
# We'll use this as the 'typical chair' baseline for NFCI>1.0 specifically
P_TYPICAL_A = 0.88  # 2/3 cuts + strong prior for NFCI>1 episodes
P_TYPICAL_B = 0.90  # 3/3 + prior; SVB borderline but still positive; NFCI>1.0 → 0.90

print(f"Bayesian-adjusted typical-chair baseline (small-sample + prior for NFCI>1.0):")
print(f"  P(Tool A | crisis): {P_TYPICAL_A:.2f}")
print(f"  P(Tool B | crisis): {P_TYPICAL_B:.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: WARSH RECORD — CALIBRATION INPUTS
# ═══════════════════════════════════════════════════════════════════════════

print()
print("=" * 74)
print("WARSH RECORD — CALIBRATION INPUTS (hardcoded, sourced)")
print("=" * 74)

WARSH_RECORD = [
    {
        'date':   '2010-11-19',
        'type':   'STATED OPPOSITION — balance sheet',
        'source': 'WSJ op-ed "The Fed\'s Dangerous Gamble" + speech at Fed conference, Nov 2010',
        'quote':  'Questioned QE2 as "unwarranted when markets function more or less normally"; '
                  'publicly broke from FOMC consensus days after Nov 2010 vote; soft dissent',
        'tool':   'B',
        'direction': 'AGAINST',
    },
    {
        'date':   '2022-2024',
        'type':   'STATED OPPOSITION — balance sheet / inflation framing',
        'source': 'Multiple public speeches and articles, 2022-24',
        'quote':  'Repeatedly described Fed balance sheet as "bloated"; framed post-COVID '
                  'inflation as a policy "choice," implying balance-sheet expansion was culpable; '
                  'advocated for faster, more aggressive QT than FOMC consensus',
        'tool':   'B',
        'direction': 'AGAINST',
    },
    {
        'date':   '2025-26',
        'type':   'STATED SUPPORT — rate cuts',
        'source': 'Public commentary 2025; reported in WSJ, Bloomberg',
        'quote':  'Expressed support for rate cuts; viewed as dovish on the rate axis '
                  'relative to consensus; critics argue he is using rate-cut flexibility '
                  'as cover while remaining hawkish on balance sheet',
        'tool':   'A',
        'direction': 'FOR',
    },
    {
        'date':   '2006-2011',
        'type':   'REVEALED COMPLIANCE — governor era',
        'source': 'FOMC voting records, Federal Reserve',
        'quote':  'Never formally dissented on any FOMC vote during his entire 2006-2011 '
                  'governorship, including all crisis-era rate cuts (Oct 2008: -50bp, '
                  'Dec 2008: -75bp to zero) and QE1 (2009). Voted with Bernanke '
                  'consensus throughout even while publicly questioning the direction.',
        'tool':   'A and B',
        'direction': 'COMPLIANT (voted FOR despite stated reservations)',
    },
]

for r in WARSH_RECORD:
    print(f"\n{r['date']}  [{r['type']}]")
    print(f"  Source: {r['source']}")
    print(f"  Tool: {r['tool']}  |  Direction: {r['direction']}")
    print(f"  Summary: {r['quote'][:110]}")

print()
print("KEY TENSION — Words vs. Votes:")
print("  Stated: opposed QE2 publicly (Nov 2010), criticizes 'bloated balance sheet'")
print("  Revealed: voted FOR all 2008-09 crisis-era cuts AND QE1 as governor")
print("  Gap explanation: as governor he could defer to Bernanke; as CHAIR he cannot.")
print("  THIS IS THE SINGLE BIGGEST UNCERTAINTY IN THE FORECAST.")
print()

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: WARSH PROBABILITY ADJUSTMENTS
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 74)
print("STEP 3 — WARSH PROBABILITY ADJUSTMENTS")
print("=" * 74)

# Tool A (rate cuts): Warsh is PRO-CUT relative to consensus
# Governor compliance + current stated pro-cut stance → at least baseline
# Slight upward adjustment: in a crisis, Warsh might cut MORE aggressively
# to compensate for avoiding balance-sheet action
P_WARSH_A = 0.88   # essentially at baseline; rate-dovish position

# Tool B (balance sheet): This is the key judgment
# LOW END: weights stated opposition; as CHAIR no one to defer to; novel-tool
#           skepticism; would try to "stick to his guns" publicly
# HIGH END: weights revealed governor compliance under real pressure;
#           systemic crisis changes calculus; legal/financial-stability
#           obligation as chair may override personal views
P_WARSH_B_LO  = 0.35  # stated opposition dominant
P_WARSH_B_MID = 0.45  # central estimate
P_WARSH_B_HI  = 0.55  # revealed compliance dominant

print(f"\nTool A (rate cuts):")
print(f"  Typical chair P(deploy A | crisis): {P_TYPICAL_A:.2f}")
print(f"  Warsh        P(deploy A | crisis): {P_WARSH_A:.2f}")
print(f"  Adjustment: MINIMAL (pro-cut stance; governor compliance on A)")
print()
print(f"Tool B (balance sheet) — THE KEY JUDGMENT:")
print(f"  Typical chair P(deploy B | crisis): {P_TYPICAL_B:.2f}")
print(f"  Warsh P(deploy B | crisis) RANGE:   {P_WARSH_B_LO:.2f} – {P_WARSH_B_HI:.2f}")
print(f"  Warsh point estimate:                {P_WARSH_B_MID:.2f}")
print()
print("  LOW END  (0.35) — weights stated opposition:")
print("    Warsh has publicly questioned QE/LSAP legitimacy for 15 years.")
print("    As CHAIR, his stated views are policy — no Bernanke to defer to.")
print("    In a 'normal-functioning' crisis (not GFC-scale), may resist for")
print("    multiple quarters while rate cuts do the heavy lifting.")
print()
print("  HIGH END (0.55) — weights revealed governor compliance:")
print("    As governor, he voted FOR every crisis-era balance-sheet action.")
print("    Legal obligation as chair + market/Treasury pressure during actual")
print("    crisis may overwhelm pre-crisis rhetoric.")
print("    Historical pattern: ALL Fed chairs eventually do 'whatever it takes.'")
print()
print("  POINT ESTIMATE (0.45): favors stated opposition slightly over revealed")
print("    compliance because: (1) chair role amplifies personal doctrine,")
print("    (2) political economy (no partisan cover from Bernanke/Powell),")
print("    (3) explicitly framed rate cuts as SUBSTITUTE for balance sheet.")

# ═══════════════════════════════════════════════════════════════════════════
# HEADLINE
# ═══════════════════════════════════════════════════════════════════════════

P_NO_BS_TYPICAL = 1 - P_TYPICAL_B
P_NO_BS_WARSH_LO  = 1 - P_WARSH_B_HI
P_NO_BS_WARSH_MID = 1 - P_WARSH_B_MID
P_NO_BS_WARSH_HI  = 1 - P_WARSH_B_LO
DELTA_LO  = P_NO_BS_WARSH_LO  - P_NO_BS_TYPICAL
DELTA_MID = P_NO_BS_WARSH_MID - P_NO_BS_TYPICAL
DELTA_HI  = P_NO_BS_WARSH_HI  - P_NO_BS_TYPICAL

print()
print("=" * 74)
print("HEADLINE — P(NO balance-sheet action within 2Q | crisis)")
print("=" * 74)
print(f"  Typical chair:              {P_NO_BS_TYPICAL:.1%}")
print(f"  Warsh (low = more likely): {P_NO_BS_WARSH_LO:.1%}")
print(f"  Warsh (point estimate):    {P_NO_BS_WARSH_MID:.1%}  <-- headline")
print(f"  Warsh (high = less likely):{P_NO_BS_WARSH_HI:.1%}")
print()
print(f"  DELTA (Warsh mid vs. typical): +{DELTA_MID:.1%}  —  the finding")
print(f"  DELTA range:                    +{DELTA_LO:.1%} to +{DELTA_HI:.1%}")

print()
print("=" * 74)
print("THESIS IMPLICATION")
print("=" * 74)
print("""\
Modern crisis management (2020, 2023) relies on BALANCE-SHEET tools, not
just rate cuts. The 2020 backstop included 9 separate 13(3) facilities
(MMLF, CPFF, PMCCF, SMCCF, MLF, PPPLF, MSLF, TALF, PDCF). The 2023
SVB response (BTFP) was standing up within 72 hours as a balance-sheet
tool deployed WHILE THE HIKING CYCLE CONTINUED — i.e. there was NO RATE
CUT. That is exactly the Warsh configuration: Tool A deployed (cuts),
Tool B possibly withheld.

The stress-amplifying configuration:
  Rate cuts (Tool A) signal easing intent — markets may initially rally.
  But without balance-sheet backstops (Tool B), credit markets stay
  impaired: corporate bond spreads, bank funding costs, and real-economy
  credit access depend on the Fed's willingness to buy/lend at scale.
  A chair who cuts rates but resists balance-sheet action lets FINANCIAL
  STRESS RUN even while policy rates ease — the worst of both worlds.

WHY THIS IS THESIS-RELEVANT:
  The China-durability thesis (F4/F5/F6) assumes the US does NOT suffer
  a balance-sheet-amplified financial crisis. If F8 = YES (Warsh resists
  balance sheet), the US financial system is meaningfully MORE vulnerable
  to the kind of credit-market seizure that amplified 2008 and was
  prevented in 2020 only by unlimited QE. That vulnerability would feed
  back into: (1) USD safe-haven demand → USDCNY overshoot → F4 updates;
  (2) global risk-off → EM capital outflows → F5 updates. The F8 signal
  matters for the whole China-side durability cluster.
""")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 1 — Tool split bar chart
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 7))
ax.grid(axis='y', zorder=0)
ax.set_axisbelow(True)

tools = ['Tool A\nInterest Rate Cuts', 'Tool B\nBalance-Sheet Expansion']
x     = np.array([0, 1])
w     = 0.32

# Typical chair bars
typ_vals = [P_TYPICAL_A, P_TYPICAL_B]
bars_typ = ax.bar(x - w/2, [v*100 for v in typ_vals], w,
                  color=ACCENT, alpha=0.85, label='Typical Chair (empirical + prior)')

# Warsh bars (midpoint)
w_vals   = [P_WARSH_A, P_WARSH_B_MID]
bars_war = ax.bar(x + w/2, [v*100 for v in w_vals], w,
                  color='#F57F17', alpha=0.85, label='Warsh Fed (point estimate)')

# Error bars for Tool B Warsh only
ax.errorbar(1 + w/2, P_WARSH_B_MID * 100,
            yerr=[[(P_WARSH_B_MID - P_WARSH_B_LO)*100], [(P_WARSH_B_HI - P_WARSH_B_MID)*100]],
            fmt='none', color=TEXT_C if 'TEXT_C' in dir() else TEXT,
            capsize=10, linewidth=2.5, capthick=2.5, zorder=5)

# Value labels on bars
for bar, val in zip(bars_typ, typ_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val*100 + 1.5,
            f'{val:.0%}', ha='center', va='bottom', color=ACCENT, fontsize=12, fontweight='bold')
for bar, val in zip(bars_war, w_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val*100 + 1.5,
            f'{val:.0%}', ha='center', va='bottom', color='#BF360C', fontsize=12, fontweight='bold')

# Range annotation on Tool B Warsh
ax.text(1 + w/2, P_WARSH_B_HI*100 + 5,
        f'uncertainty range:\n{P_WARSH_B_LO:.0%}–{P_WARSH_B_HI:.0%}',
        ha='center', va='bottom', color=MUTED, fontsize=9, style='italic')

ax.axhline(100, color=MUTED, linewidth=0.8, linestyle=':', alpha=0.5)

# Delta annotation
ax.annotate(
    f'Gap = +{DELTA_MID:.0%}\n(Warsh less likely\nto use balance sheet)',
    xy=(1 + w/2, P_WARSH_B_MID*100), xytext=(1.45, P_WARSH_B_MID*100),
    ha='left', color=RED, fontsize=9.5, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=RED, alpha=0.9))

ax.set_xticks(x)
ax.set_xticklabels(tools, fontsize=12)
ax.set_ylabel('Probability of deploying within 2 quarters of crisis onset', fontsize=11, labelpad=8)
ax.set_ylim(0, 112)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax.set_title(
    'How Would Warsh Respond to a Financial Crisis?  [Conditional Forecast]\n'
    'P(each tool deployed within 2 quarters | NFCI > 1.0 + significant credit stress)',
    fontsize=11.5, fontweight='bold', pad=12)

legend_elems = [
    mpatches.Patch(color=ACCENT,   alpha=0.85, label='Typical Chair (empirical baseline + Bayesian prior)'),
    mpatches.Patch(color='#F57F17', alpha=0.85, label='Warsh (point estimate)'),
    Line2D([0], [0], color=TEXT, linewidth=2, label='Warsh Tool B uncertainty range'),
]
ax.legend(handles=legend_elems, fontsize=10)

note = (f"Note: crisis defined as NFCI > 1.0 + significant credit spread blowout (as in 2008/2020/2023).  "
        f"Tool A gap: {abs(P_WARSH_A - P_TYPICAL_A):.0%} — Warsh roughly at baseline for rate cuts.  "
        f"Tool B gap: +{DELTA_MID:.0%} — the thesis-relevant finding.")
ax.text(0.5, -0.09, note, transform=ax.transAxes,
        ha='center', va='top', color=MUTED, fontsize=9, style='italic')

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig('out/forecast8_tool_split.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\nSaved: out/forecast8_tool_split.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 2 — Historical WALCL + NFCI with episodes marked
# ═══════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})
for _ax in (ax1, ax2):
    _ax.grid(True, color=GRID_C, alpha=1.0, linewidth=0.6)
    _ax.set_axisbelow(True)

# ── WALCL ──────────────────────────────────────────────────────────────────
ax1.plot(df_walcl['date'], df_walcl['WALCL'],
         color=ACCENT, linewidth=2.2, label='Fed balance sheet total assets (WALCL, $bn)')
ax1.set_ylabel('Federal Reserve Total Assets ($bn)', fontsize=10, labelpad=8)
ax1.set_title(
    'Federal Reserve Balance Sheet and Financial Conditions — Three Major Stress Episodes\n'
    'Balance-sheet tools were deployed in every episode where NFCI exceeded 1.0',
    fontsize=11.5, fontweight='bold', pad=12)

# ── NFCI ───────────────────────────────────────────────────────────────────
ax2.plot(df_nfci['date'], df_nfci['NFCI'],
         color='#1565C0', linewidth=1.8, label='NFCI (National Financial Conditions Index)')
ax2.axhline(1.0, color=RED, linewidth=1.5, linestyle='--', alpha=0.85,
            label='Crisis threshold (NFCI = 1.0)')
ax2.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=':', alpha=0.5)
ax2.fill_between(df_nfci['date'], df_nfci['NFCI'], 0,
                 where=df_nfci['NFCI'] > 0, color='#FFCDD2', alpha=0.35)
ax2.set_ylabel('NFCI\n(+ve = tighter conditions)', fontsize=10, labelpad=8)
ax2.set_xlabel('Date', fontsize=10, labelpad=8)
ax2.set_ylim(-1.5, 3.8)

# ── Episode shading and annotations ────────────────────────────────────────
for ep in EPISODES:
    onset_dt = pd.Timestamp(ep['onset'])
    col = ep['color']
    for _ax in (ax1, ax2):
        _ax.axvspan(onset_dt, onset_dt + pd.DateOffset(months=6),
                    alpha=0.10, color=col, zorder=0)

    walcl_at = df_walcl[df_walcl['date'] >= onset_dt]['WALCL'].iloc[:1]
    y_ann  = float(walcl_at.iloc[0]) if len(walcl_at) else 5000
    y_text = y_ann + 350
    bs_tag = f'Rate cuts: YES  |  Balance sheet: YES (+${ep["bs_expand_bn"]}bn)' if ep['bs_deploy_B'] \
             else f'Rate cuts: {"YES" if ep["rate_cut_A"] else "NO"}  |  Balance sheet: NO'
    ax1.annotate(
        f"{ep['name']}\n{bs_tag}",
        xy=(onset_dt, y_ann),
        xytext=(onset_dt + pd.DateOffset(months=4), y_text),
        color=col, fontsize=8.5, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=col, lw=1.0),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=col, alpha=0.9))

    ax2.annotate(
        f"Peak: {ep['peak_nfci']:.2f}",
        xy=(onset_dt + pd.DateOffset(weeks=4), ep['peak_nfci']),
        xytext=(onset_dt + pd.DateOffset(months=8), ep['peak_nfci'] - 0.4),
        color=col, fontsize=8.0,
        arrowprops=dict(arrowstyle='->', color=col, lw=0.9))

leg_patches = [mpatches.Patch(color=ep['color'], alpha=0.4, label=ep['name']) for ep in EPISODES]
ax1.legend(handles=[ax1.get_lines()[0]] + leg_patches, fontsize=9, loc='upper left')
ax2.legend(fontsize=9, loc='upper right')

note3 = ('Balance-sheet tools were deployed in all three episodes. '
         'In 2023, the BTFP was stood up in 72 hours while the rate-hiking cycle continued — '
         "demonstrating that rate cuts and balance-sheet action are independent policy choices, "
         "not substitutes.")
fig.text(0.5, 0.01, note3, ha='center', color=MUTED, fontsize=9, style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('out/forecast8_historical_episodes.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: out/forecast8_historical_episodes.png")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY TEXT
# ═══════════════════════════════════════════════════════════════════════════

lines = []
def w(s=""): lines.append(s)

w("=" * 74)
w("FORECAST 8 — WARSH FED: RESPONSE CONDITIONAL ON CRISIS")
w(f"Generated: {datetime.now().strftime('%Y-%m-%d')}")
w("=" * 74)
w()
w("*** CONDITIONAL FORECAST — CRISIS IS ASSUMED, NOT SIMULATED ***")
w("All probabilities are P(response | crisis) where:")
w("  Crisis definition: NFCI > 1.0 AND significant credit spread blowout,")
w("  analogous to 2008 GFC / 2020 COVID onset conditions.")
w("  Prior version error: ~84% 'no emergency' with ~69% from 'no crisis'")
w("  — that was wrong. This version isolates the response conditional on stress.")
w()
w("─" * 74)
w("TOOL DEFINITIONS")
w("─" * 74)
w("Tool A: Policy RATE CUTS — FOMC rate reduction within 2 quarters of onset.")
w("Tool B: BALANCE-SHEET INTERVENTION — LSAP / QE / new 13(3) facilities /")
w("  balance-sheet expansion >$250B in any 8-week window outside routine ops.")
w("  THIS IS THE THESIS-RELEVANT TOOL. Rate cuts ≠ crisis backstop.")
w()
w("─" * 74)
w("TYPICAL-CHAIR BASELINE (empirical + Bayesian prior)")
w("─" * 74)
w("Historical episodes with NFCI > 1.0 threshold:")
for ep in EPISODES:
    ra = f"+{ep['rate_cut_amt_pp']}bp, {ep['rate_cut_weeks']}wk" if ep['rate_cut_A'] else "NONE"
    bs = f"+${ep['bs_expand_bn']}bn in {ep['bs_weeks']}wk ({ep['bs_tool'][:45]})" if ep['bs_deploy_B'] else "NONE"
    w(f"  {ep['name']}: A={ra}  |  B={bs}")
w()
w(f"Empirical: P(A | crisis)={p_base_A:.2f}  P(B | crisis)={p_base_B:.2f}")
w(f"Bayesian-adjusted (small-sample + prior for NFCI>1): P(A)={P_TYPICAL_A:.2f}  P(B)={P_TYPICAL_B:.2f}")
w()
w("─" * 74)
w("WARSH RECORD — CALIBRATION INPUTS")
w("─" * 74)
for r in WARSH_RECORD:
    w(f"\n{r['date']}  [{r['type']}]")
    w(f"  Source: {r['source']}")
    w(f"  {r['quote'][:120]}")
w()
w("WORDS vs. VOTES TENSION — the single biggest uncertainty:")
w("  Stated: questioned QE2 (Nov 2010), called balance sheet 'bloated' (2022-24),")
w("  frames inflation as a 'choice' implying balance-sheet expansion was culpable.")
w("  Revealed: voted FOR every crisis-era action as governor 2006-2011 (all cuts,")
w("  QE1, 13(3) facilities). As GOVERNOR he could defer to Bernanke. As CHAIR he cannot.")
w()
w("─" * 74)
w("WARSH ADJUSTMENTS")
w("─" * 74)
w(f"Tool A (rate cuts): P(deploy A | crisis) = {P_WARSH_A:.2f}")
w(f"  Same as baseline. Warsh is pro-cut; governor compliance record on A confirmed.")
w()
w(f"Tool B (balance sheet): P(deploy B | crisis) = {P_WARSH_B_LO:.2f}–{P_WARSH_B_HI:.2f}  [pt: {P_WARSH_B_MID:.2f}]")
w(f"  vs. typical chair baseline {P_TYPICAL_B:.2f}")
w(f"  LOW  {P_WARSH_B_LO:.2f}: stated opposition dominant; chair role removes deference option")
w(f"  HIGH {P_WARSH_B_HI:.2f}: revealed governor compliance under real stress dominant")
w(f"  MID  {P_WARSH_B_MID:.2f}: slightly favors stated opposition (chair role amplifies doctrine)")
w()
w("─" * 74)
w("HEADLINE — P(NO balance-sheet action within 2Q | crisis)")
w("─" * 74)
w(f"  Typical chair:              {P_NO_BS_TYPICAL:.1%}")
w(f"  Warsh (low  = more action): {P_NO_BS_WARSH_LO:.1%}")
w(f"  Warsh (point estimate):     {P_NO_BS_WARSH_MID:.1%}   <-- HEADLINE")
w(f"  Warsh (high = less action): {P_NO_BS_WARSH_HI:.1%}")
w()
w(f"  DELTA (Warsh mid vs. typical chair): +{DELTA_MID:.1%}")
w(f"  DELTA range: +{DELTA_LO:.1%} to +{DELTA_HI:.1%}")
w()
w("  Note: The low/high range is not a confidence interval. It reflects")
w("  two coherent readings of Warsh's record (words vs. votes), not noise.")
w()
w("─" * 74)
w("THESIS IMPLICATION")
w("─" * 74)
w("Modern crisis management relies on balance-sheet tools, not just rate cuts.")
w("2020: 9 new 13(3) facilities deployed at scale; crisis contained in weeks.")
w("2023: BTFP stood up in 72 HOURS while the hiking cycle CONTINUED.")
w("      -> Rate cuts (A) and balance sheet (B) are INDEPENDENT policy levers.")
w("      -> A chair can cut rates AND refuse balance-sheet action simultaneously.")
w()
w("Warsh's stated position — pro-cut + anti-balance-sheet — is precisely the")
w("stress-amplifying configuration: markets initially rally on cut signaling,")
w("but credit markets stay impaired without balance-sheet backstops.")
w()
w("THIS MATTERS FOR THE CHINA THESIS:")
w("  Warsh resisting balance sheet (F8=YES) -> US financial system meaningfully")
w("  more vulnerable to credit-market seizure -> amplified global risk-off ->")
w("  USD safe-haven surge -> USDCNY overshoot (F4 updates upward) ->")
w("  EM capital outflows -> China FX reserves pressure (F5 updates).")
w("  F8 is a key upstream variable for the China durability cluster.")
w()
w("=" * 74)
w("FILES: forecast8_tool_split.png, forecast8_historical_episodes.png,")
w("       forecast8_warsh_summary.txt")

with open('out/forecast8_warsh_summary.txt', 'w') as f:
    f.write('\n'.join(lines))
print("Saved: out/forecast8_warsh_summary.txt")

# ── Final stdout headline ──────────────────────────────────────────────────
print()
print("=" * 74)
print(f"  Typical-chair P(deploy balance sheet | crisis): {P_TYPICAL_B:.1%}")
print(f"  Warsh         P(deploy balance sheet | crisis): {P_WARSH_B_LO:.1%}–{P_WARSH_B_HI:.1%}  [pt: {P_WARSH_B_MID:.1%}]")
print(f"  P(NO balance-sheet action | crisis) [pt est]:  {P_NO_BS_WARSH_MID:.1%}  (vs. {P_NO_BS_TYPICAL:.1%} typical)")
print(f"  DELTA (Warsh vs. typical chair):                +{DELTA_MID:.1%}")
print("=" * 74)
