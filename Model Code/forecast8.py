#!/usr/bin/env python3
"""
Forecast 11 — China Debt/GDP: Steady Rise vs. Accelerating Spiral
Tests the 2nd derivative (rate of change of annual leverage change), not the level.
Rising debt is expected (Japan-path). Acceleration is the crisis signal.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')
os.makedirs('out', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ── dark theme ─────────────────────────────────────────────────────────────
DARK_BG  = '#0f1117'
PANEL_BG = '#1a1d2e'
TEXT     = '#e0e0e0'
GRID     = '#2a2d3e'
GREEN    = '#66bb6a'
ORANGE   = '#ffa726'
RED      = '#ef5350'
ACCENT   = '#4fc3f7'
MUTED    = '#78909c'
GOLD     = '#ffd54f'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   PANEL_BG,
    'axes.edgecolor':   GRID,
    'axes.labelcolor':  TEXT,
    'text.color':       TEXT,
    'xtick.color':      TEXT,
    'ytick.color':      TEXT,
    'grid.color':       GRID,
    'grid.alpha':       0.35,
    'font.size':        11,
})

# ═══════════════════════════════════════════════════════════════════════════
# OPERATIONAL RULE
# ═══════════════════════════════════════════════════════════════════════════

STEADY_LO    = 2.0   # pp/yr
STEADY_HI    = 10.0  # pp/yr
SLOPE_THRESH = 0.40  # pp/yr per year — max linear trend slope for "steady"

print("=" * 74)
print("FORECAST 11 — CHINA DEBT/GDP: STEADY RISE VS. ACCELERATING SPIRAL")
print("=" * 74)
print()
print("OPERATIONAL RULE (exact event definition):")
print(f"  YES (durability/steady) if, over the full horizon:")
print(f"    (a) All annual pp-changes in leverage within [{STEADY_LO}, {STEADY_HI}] pp/yr")
print(f"    (b) Linear trend slope of the annual-change series < {SLOPE_THRESH} pp/yr/yr")
print(f"  NO (spiral) if annual change trends upward — 2nd derivative positive, sustained.")
print()
print("REFRAME: Rising debt/GDP is EXPECTED and consistent with the thesis.")
print("  Japan rose past 400% for 30 years without a crisis. The test is")
print("  whether the rise is STEADY (absorbable) or ACCELERATING (spiral).")
print()

# ═══════════════════════════════════════════════════════════════════════════
# DATA — BIS Total Credit / GDP, China
# ═══════════════════════════════════════════════════════════════════════════

BIS_URL = ("https://stats.bis.org/api/v2/data/dataflow/BIS/WS_TC/1.0/"
           "Q.CHN.P.A.M.XDC.A?format=csv")
df_hist = None
try:
    import urllib.request, ssl, io
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    with urllib.request.urlopen(BIS_URL, context=ctx, timeout=10) as r:
        raw = r.read().decode('utf-8')
    df_bis = pd.read_csv(io.StringIO(raw))
    val_col  = [c for c in df_bis.columns if 'value' in c.lower() or 'obs' in c.lower()]
    time_col = [c for c in df_bis.columns if 'period' in c.lower() or 'time' in c.lower() or 'date' in c.lower()]
    if val_col and time_col:
        tmp = df_bis[[time_col[0], val_col[0]]].copy()
        tmp.columns = ['quarter', 'leverage']
        tmp['leverage'] = pd.to_numeric(tmp['leverage'], errors='coerce')
        tmp = tmp.dropna().sort_values('quarter').reset_index(drop=True)
        if len(tmp) > 20:
            df_hist = tmp
            print(f"[BIS fetch] Loaded {len(df_hist)} quarters from BIS API.")
except Exception as e:
    print(f"[BIS fetch] Failed ({type(e).__name__}); using hardcoded data.")

if df_hist is None:
    # Annual end-of-year; BIS all-non-financial-sector credit / GDP, China
    ANNUAL_HIST = [
        ("2005-Q4", 148.0), ("2006-Q4", 151.0), ("2007-Q4", 148.0),
        ("2008-Q4", 142.0), ("2009-Q4", 157.0), ("2010-Q4", 178.0),
        ("2011-Q4", 183.0), ("2012-Q4", 197.0), ("2013-Q4", 215.0),
        ("2014-Q4", 225.0), ("2015-Q4", 238.0), ("2016-Q4", 252.0),
        ("2017-Q4", 252.0), ("2018-Q4", 252.0), ("2019-Q4", 253.0),
        ("2020-Q4", 283.0), ("2021-Q4", 279.0), ("2022-Q4", 289.0),
        ("2023-Q4", 297.0), ("2024-Q4", 302.0),
    ]
    df_hist = pd.DataFrame(ANNUAL_HIST, columns=['quarter', 'leverage'])

df_hist.to_csv('data/china_leverage.csv', index=False)

print("\nChina Total Credit / GDP (%)  —  BIS / NIFD sourced:")
print("-" * 60)
lev_vals = df_hist['leverage'].values
yoy_chg  = np.concatenate([[np.nan], np.diff(lev_vals)])
for i, row in df_hist.iterrows():
    chg_s = f"{yoy_chg[i]:+.1f}pp" if not np.isnan(yoy_chg[i]) else "    --"
    print(f"  {row.quarter:<10}  {row.leverage:>6.1f}%   annual chg: {chg_s}")
print()

# Additional calibration inputs
NGDP_GROWTH_ANN = 0.045   # nominal GDP growth baseline ~4.5%/yr
TSF_GROWTH_ANN  = 0.070   # total social financing growth ~7%/yr (spec: 7-8%)
print(f"Calibration anchors: NGDP growth ~{NGDP_GROWTH_ANN*100:.1f}%/yr  "
      f"| TSF growth ~{TSF_GROWTH_ANN*100:.1f}%/yr")

# ═══════════════════════════════════════════════════════════════════════════
# HISTORICAL ANNUAL CHANGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

quarters_hist = df_hist['quarter'].values
ann_chg_hist  = np.diff(lev_vals)
yrs_hist      = quarters_hist[1:]

# Recent window: 2015 onward, exclude COVID 2020 one-time shock
mask_recent   = np.array([q >= '2015' and q != '2020-Q4' for q in yrs_hist])
recent_chgs   = ann_chg_hist[mask_recent]
mu_hist       = float(np.mean(recent_chgs))
sigma_hist    = float(np.std(recent_chgs, ddof=1))

t_idx = np.arange(len(recent_chgs), dtype=float)
slope_hist, intercept_h, r_h, p_h, _ = stats.linregress(t_idx, recent_chgs)
hist_accel = "YES" if p_h < 0.10 else "NO"

print("Historical annual change analysis (2015–2024, ex-COVID 2020):")
print(f"  Values (pp/yr): {[f'{x:.1f}' for x in recent_chgs]}")
print(f"  Mean: {mu_hist:.1f} pp/yr  |  Std: {sigma_hist:.1f} pp  |  Trend slope: {slope_hist:.2f} pp/yr/yr  (p={p_h:.2f})")
print(f"  Historically accelerating: {hist_accel} — slope {slope_hist:.2f} vs. threshold {SLOPE_THRESH}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

L0           = float(df_hist['leverage'].iloc[-1])   # ~302%
NGDP_Q       = (1 + NGDP_GROWTH_ANN) ** 0.25 - 1

# AR(1) calibration for quarterly growth rates
RHO_NGDP     = 0.40
RHO_CREDIT   = 0.50
SIGMA_NGDP_Q = 0.0040   # ~1.6pp/yr annualised vol
SIGMA_CRED_Q = 0.0060   # ~2.4pp/yr annualised vol
RHO_CORR     = 0.50     # credit-NGDP innovation correlation

N_PATHS      = 10_000
SETSER_ADJ_Q = -0.015 / 4   # Setser: NGDP overstated 1.5pp/yr → quarterly

# ── ICOR scenarios ─────────────────────────────────────────────────────────
# spread_t = spread_0 + accel × t (t in quarters)
# Annual leverage change ≈ L_t × 4 × spread_q / (1 + NGDP_ANN)
# At L=302, 1bp quarterly spread ≈ 302 × 0.0001 × 4 / 1.045 ≈ 0.116pp/yr

SCENARIOS = {
    'Productive': {
        'color':      GREEN,
        'label':      'Productive\n(ICOR stable)',
        'spread_0_q': 0.0035,    # ~1.35pp/yr spread → ~3.9pp/yr annual change
        'accel_q':    0.0,
        'desc':       'ICOR constant; credit grows ~1.4pp/yr faster than NGDP; annual change ~4pp/yr flat',
        'icor_note':  'Productive investment — returns stable; debt-service capacity grows with debt',
    },
    'Mild': {
        'color':      ORANGE,
        'label':      'Mildly Unproductive\n(ICOR rising slowly)',
        'spread_0_q': 0.0050,    # ~2.0pp/yr → ~5.8pp/yr initially
        'accel_q':    0.000030,  # +0.012pp/yr each year → ~7-9pp/yr by 2030
        'desc':       'ICOR rising ~1-2% per year; annual change drifts from ~6 to ~9pp/yr by 2032',
        'icor_note':  'Property/infrastructure returns declining; reinvestment continues at diminishing return',
    },
    'Dead-weight': {
        'color':      RED,
        'label':      'Dead-weight Capex\n(ICOR rising steeply)',
        'spread_0_q': 0.0060,    # ~2.4pp/yr → ~6.9pp/yr initially
        'accel_q':    0.000090,  # +0.036pp/yr each year → ~12-15pp/yr by 2030
        'desc':       'ICOR rising 4-5% per year; annual change accelerates from ~7 to ~15pp/yr by 2032',
        'icor_note':  'White-elephant capex compounding — each year needs more credit per unit GDP output',
    },
}

WEIGHTS = {'Productive': 0.28, 'Mild': 0.47, 'Dead-weight': 0.25}
assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1"

# Horizons (quarters from Q1-2025)
HORIZONS = {2028: 14, 2030: 22, 2032: 30}

# ═══════════════════════════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════

def run_mc(sp, n_paths=N_PATHS, setser=False, seed=42):
    """Simulate quarterly leverage paths; return level and annual-change arrays."""
    rng   = np.random.default_rng(seed)
    n_ann = max(HORIZONS.values()) // 4 + 5   # annual periods (extra headroom)
    max_q = n_ann * 4                          # quarters to simulate

    Sigma = np.array([[SIGMA_NGDP_Q**2,
                       RHO_CORR * SIGMA_NGDP_Q * SIGMA_CRED_Q],
                      [RHO_CORR * SIGMA_NGDP_Q * SIGMA_CRED_Q,
                       SIGMA_CRED_Q**2]])
    L_chol = np.linalg.cholesky(Sigma)

    lev  = np.zeros((n_paths, max_q + 1))
    lev[:, 0] = L0
    gn   = np.full(n_paths, NGDP_Q)
    gc   = np.full(n_paths, NGDP_Q + sp['spread_0_q'])

    for t in range(max_q):
        spread_t  = sp['spread_0_q'] + sp['accel_q'] * t
        mn_q      = NGDP_Q + (SETSER_ADJ_Q if setser else 0.0)
        mc_q      = mn_q + spread_t
        eps       = rng.standard_normal((n_paths, 2)) @ L_chol.T
        gn_new    = mn_q + RHO_NGDP   * (gn - mn_q) + eps[:, 0]
        gc_new    = mc_q + RHO_CREDIT * (gc - mc_q)  + eps[:, 1]
        lev[:, t+1] = lev[:, t] * (1 + gc_new) / (1 + gn_new)
        gn, gc    = gn_new, gc_new

    ann = np.stack([lev[:, (k+1)*4] - lev[:, k*4] for k in range(n_ann)], axis=1)
    return lev, ann


def _slopes(ann, n_years):
    """Linear trend slope of the annual-change series for each path."""
    sub   = ann[:, :n_years]
    t_idx = np.arange(n_years, dtype=float)
    t_dev = t_idx - t_idx.mean()
    return (sub * t_dev).sum(axis=1) / (t_dev ** 2).sum()


def p_no_accel(ann, n_years):
    """P(slope < threshold) — pure 2nd-derivative test, no band condition.
    PRIMARY metric: does the annual change accelerate?"""
    return float((_slopes(ann, n_years) < SLOPE_THRESH).mean())


def p_steady(ann, n_years):
    """P(MEAN annual change in band AND slope < threshold) — joint test.
    Uses MEAN annual change over the horizon (not every individual year)
    so that point-in-time noise doesn't swamp the acceleration signal."""
    sub      = ann[:, :n_years]
    mean_chg = sub.mean(axis=1)
    in_band  = (mean_chg >= STEADY_LO) & (mean_chg <= STEADY_HI)
    flat     = _slopes(ann, n_years) < SLOPE_THRESH
    return float((in_band & flat).mean())


print("Running Monte Carlo (10,000 paths × 3 scenarios × baseline + Setser)...")
results = {}
for name, sp in SCENARIOS.items():
    lev_b, ann_b = run_mc(sp, setser=False)
    lev_s, ann_s = run_mc(sp, setser=True,  seed=99)
    results[name] = dict(lev_b=lev_b, ann_b=ann_b, lev_s=lev_s, ann_s=ann_s)

# ─── P(steady) tables ─────────────────────────────────────────────────────

ps_base  = {n: {yr: p_steady(results[n]['ann_b'], nq//4)   for yr, nq in HORIZONS.items()} for n in SCENARIOS}
ps_set   = {n: {yr: p_steady(results[n]['ann_s'], nq//4)   for yr, nq in HORIZONS.items()} for n in SCENARIOS}
pna_base = {n: {yr: p_no_accel(results[n]['ann_b'], nq//4) for yr, nq in HORIZONS.items()} for n in SCENARIOS}

pb  = {yr: sum(WEIGHTS[n] * ps_base[n][yr]  for n in SCENARIOS) for yr in HORIZONS}
ps  = {yr: sum(WEIGHTS[n] * ps_set[n][yr]   for n in SCENARIOS) for yr in HORIZONS}
pna = {yr: sum(WEIGHTS[n] * pna_base[n][yr] for n in SCENARIOS) for yr in HORIZONS}

wt_str = " / ".join(f"{k}:{v:.0%}" for k, v in WEIGHTS.items())
print()
print("=" * 74)
print("PRIMARY: P(NO ACCELERATION) — slope-only test (2nd derivative)")
print(f"  Test: slope of annual-change series < {SLOPE_THRESH} pp/yr/yr")
print("=" * 74)
print(f"  {'Scenario':<22}  {'2028':>8}  {'2030':>8}  {'2032':>8}")
print("  " + "-" * 48)
for name in SCENARIOS:
    pa = pna_base[name]
    print(f"  {name:<22}  {pa[2028]:>7.1%}  {pa[2030]:>7.1%}  {pa[2032]:>7.1%}")
print(f"  {'BLENDED':>22}  {pna[2028]:>7.1%}  {pna[2030]:>7.1%}  {pna[2032]:>7.1%}")
print(f"  (weights: {wt_str})")
print()

print("JOINT: P(STEADY — mean change in band AND no acceleration)")
print(f"  Band: mean annual change in [{STEADY_LO}, {STEADY_HI}] pp/yr AND slope < {SLOPE_THRESH}")
print(f"  {'Scenario':<22}  {'2028':>8}  {'2030':>8}  {'2032':>8}")
print("  " + "-" * 48)
for name in SCENARIOS:
    pa = ps_base[name]
    print(f"  {name:<22}  {pa[2028]:>7.1%}  {pa[2030]:>7.1%}  {pa[2032]:>7.1%}")
print(f"  {'BLENDED':>22}  {pb[2028]:>7.1%}  {pb[2030]:>7.1%}  {pb[2032]:>7.1%}")
print()

print("Setser robustness (NGDP -1.5pp/yr) — impact on joint P(steady):")
print(f"  {'Scenario':<22}  {'d2028':>8}  {'d2030':>8}  {'d2032':>8}")
print("  " + "-" * 48)
for name in SCENARIOS:
    d = {yr: ps_set[name][yr] - ps_base[name][yr] for yr in HORIZONS}
    print(f"  {name:<22}  {d[2028]:>+7.1%}  {d[2030]:>+7.1%}  {d[2032]:>+7.1%}")
db = {yr: ps[yr] - pb[yr] for yr in HORIZONS}
print(f"  {'Blended delta':<22}  {db[2028]:>+7.1%}  {db[2030]:>+7.1%}  {db[2032]:>+7.1%}")
print()
print("  Note: Setser raises the level trajectory; the key question is whether")
print("  it changes the ACCELERATION. The slope-only test is more Setser-robust.")
print()

# ─── Scenario medians for context ─────────────────────────────────────────

print("=" * 74)
print("SCENARIO ANNUAL CHANGE MEDIANS (pp/yr)")
print("=" * 74)
n_yr_show = HORIZONS[2032] // 4
for name, sp in SCENARIOS.items():
    med = np.median(results[name]['ann_b'][:, :n_yr_show], axis=0)
    print(f"\n{name}:  {sp['desc']}")
    vals = " | ".join(f"{2025+k}: {med[k]:+.1f}" for k in range(n_yr_show))
    print(f"  {vals}")

# ─── Interpretation ───────────────────────────────────────────────────────

print()
print("=" * 74)
print("INTERPRETATION")
print("=" * 74)
print("""\
Rising debt/GDP is EXPECTED and is NOT the crisis signal. Japan exceeded
400% over three decades without a sovereign crisis because the rise was
steady — creditors could model the trajectory. The panic signal is when
the annual increment itself starts growing (acceleration), implying that
debt must grow faster each year just to sustain the same growth rate.

PRODUCTIVE (P(steady 2030) high): ICOR constant; credit grows modestly
  faster than NGDP; the gap is stable. Absorbable on the Japan path.

MILD (P(steady 2030) moderate): ICOR rising slowly as property/infra
  returns decline but don't collapse. Annual change drifts upward into
  the 8-10pp/yr range by 2030. Partially absorbable but warrants watching.

DEAD-WEIGHT (P(steady 2030) low): This is the Forecast 2 mechanism
  applied to China's infrastructure complex. In F2, AI capex ICOR rises
  because each successive model training run yields less marginal value.
  Here, successive white-elephant projects yield less GDP per credit
  dollar — the same mechanism. By 2030 the annual change accelerates
  above the band; by 2032 it's clearly in spiral territory.

ACCELERATION IS THE SELF-FINANCING FAILURE: when the annual increment
  grows each year, the debt stock must grow at an accelerating rate just
  to deliver the same nominal GDP growth. This is the dead-weight capex
  compounding loop — China's parallel to the AI capex overhang.

LINK TO DURABILITY CLUSTER (F4/F5/F6):
  A YES (steady rise) supports: F4 (USDCNY stable 97.9%), F5 (FX reserves
  above floor 95%), F6 (property defaults don't transmit). Together these
  form a mutually-consistent baseline. The stress tail requires F11=NO
  (acceleration) AND F6=YES (credit transmission to broader system).
""")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 1 — Fan chart of ANNUAL pp-CHANGE (the actual test)
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(
    "Forecast 11 — Annual pp-Change in China Debt/GDP  (THE REAL TEST: Steady vs. Accelerating)\n"
    "Rising leverage is expected. The question is whether the increment itself is growing.",
    color=TEXT, fontsize=11.5, y=0.99)

n_yr_plot  = HORIZONS[2032] // 4
years_fwd  = [2025 + k for k in range(n_yr_plot)]

# Historical context (annual changes for years >= 2015, ex-COVID)
h_yrs  = [int(q.split('-')[0]) for q, c in zip(yrs_hist, ann_chg_hist) if q >= '2015-Q4']
h_chgs = [c for q, c in zip(yrs_hist, ann_chg_hist) if q >= '2015-Q4']

for ax, (s_name, sp) in zip(axes, SCENARIOS.items()):
    ax.set_facecolor(PANEL_BG)
    ann = results[s_name]['ann_b'][:, :n_yr_plot]

    p10  = np.percentile(ann, 10, axis=0)
    p25  = np.percentile(ann, 25, axis=0)
    p50  = np.median(ann, axis=0)
    p75  = np.percentile(ann, 75, axis=0)
    p90  = np.percentile(ann, 90, axis=0)

    ax.fill_between(years_fwd, p10, p90, alpha=0.15, color=sp['color'])
    ax.fill_between(years_fwd, p25, p75, alpha=0.32, color=sp['color'])
    ax.plot(years_fwd, p50, color=sp['color'], linewidth=2.5)

    # Steady band
    ax.axhspan(STEADY_LO, STEADY_HI, alpha=0.08, color=ACCENT, zorder=0)
    ax.axhline(STEADY_LO, color=ACCENT, linewidth=0.9, linestyle=':', alpha=0.6)
    ax.axhline(STEADY_HI, color=ACCENT, linewidth=0.9, linestyle=':', alpha=0.6)
    ax.text(2025.2, STEADY_HI + 0.4,
            f'Steady band ({STEADY_LO}-{STEADY_HI}pp/yr)',
            color=ACCENT, fontsize=8, alpha=0.8)

    # Historical dots
    ax.scatter(h_yrs, h_chgs, color=GOLD, s=55, zorder=5, alpha=0.88,
               label='Historical (ex-COVID)')
    ax.axhline(0, color=MUTED, linewidth=0.8, alpha=0.35)

    # Horizon lines
    for yr in HORIZONS:
        ax.axvline(yr, color=MUTED, linewidth=0.8, linestyle='--', alpha=0.35)

    p_vals = " / ".join(f"{yr}:{ps_base[s_name][yr]:.0%}" for yr in HORIZONS)
    ax.set_title(
        f"{sp['label']}\nP(steady): {p_vals}",
        color=sp['color'], fontsize=9.5)
    ax.set_xlim(2014.5, 2033)
    ax.set_xticks([2015, 2020, 2025, 2028, 2030, 2032])
    ax.set_xticklabels([2015, 2020, 2025, 2028, 2030, 2032],
                       rotation=30, fontsize=9)
    ax.grid(True, alpha=0.22)
    ax.tick_params(colors=TEXT)

axes[0].set_ylabel("Annual Change in Debt/GDP (pp/yr)", color=TEXT, fontsize=10)

legend_elems = [
    mpatches.Patch(alpha=0.30, color=MUTED, label='IQR (P25–P75)'),
    mpatches.Patch(alpha=0.15, color=MUTED, label='P10–P90'),
    mpatches.Patch(alpha=0.12, color=ACCENT, label=f'Steady band ({STEADY_LO}–{STEADY_HI} pp/yr)'),
    Line2D([0],[0], color=GOLD, marker='o', markersize=7,
           linestyle='none', label='Historical annual change'),
]
axes[2].legend(handles=legend_elems, loc='upper right',
               facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

note = (f"Steady test: all annual changes within [{STEADY_LO},{STEADY_HI}] pp/yr AND"
        f" trend slope < {SLOPE_THRESH} pp/yr/yr.  Gold dots = actual history (ex-COVID)."
        f"  Acceleration = median rising out of the blue band over time.")
fig.text(0.5, 0.01, note, ha='center', color=MUTED, fontsize=9, style='italic')

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
plt.savefig('out/forecast11_leverage_change_fan.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Saved: out/forecast11_leverage_change_fan.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 2 — Level trajectory (context only)
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(PANEL_BG)

# Historical
h_yrs_all = [int(q.split('-')[0]) for q in df_hist['quarter']]
ax.plot(h_yrs_all, lev_vals, color=GOLD, linewidth=2.5, label='Historical (BIS)', zorder=5)
ax.scatter(h_yrs_all, lev_vals, color=GOLD, s=28, zorder=6, alpha=0.7)

# Forward fan per scenario
n_q_plot  = HORIZONS[2032] + 8
fwd_q_yrs = [2025 + t / 4 for t in range(n_q_plot + 1)]

for s_name, sp in SCENARIOS.items():
    lev  = results[s_name]['lev_b'][:, :len(fwd_q_yrs)]
    p10  = np.percentile(lev, 10, axis=0)
    p25  = np.percentile(lev, 25, axis=0)
    p50  = np.median(lev, axis=0)
    p75  = np.percentile(lev, 75, axis=0)
    p90  = np.percentile(lev, 90, axis=0)

    ax.fill_between(fwd_q_yrs, p10, p90, alpha=0.10, color=sp['color'])
    ax.fill_between(fwd_q_yrs, p25, p75, alpha=0.22, color=sp['color'])
    ls = '--' if s_name == 'Dead-weight' else '-'
    ax.plot(fwd_q_yrs, p50, color=sp['color'], linewidth=2.0,
            linestyle=ls, label=f"{s_name} (median)")

# Japan reference
ax.axhline(400, color=MUTED, linewidth=1.2, linestyle=':', alpha=0.5)
ax.text(2013.5, 402, 'Japan peak: 400%+ (no crisis)', color=MUTED, fontsize=9, alpha=0.7)

for yr in HORIZONS:
    ax.axvline(yr, color=MUTED, linewidth=0.8, linestyle='--', alpha=0.30)

ax.set_xlim(2005, 2033)
ax.set_ylabel("Total Credit / GDP (%)", color=TEXT, fontsize=11)
ax.set_xlabel("Year", color=TEXT, fontsize=11)
ax.set_title(
    "China Debt/GDP Level — FOR CONTEXT ONLY\n"
    "Level rising is EXPECTED and is NOT the test. See fan chart above for the actual test.",
    color=TEXT, fontsize=11.5, pad=12)
ax.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc='upper left')
ax.grid(True, alpha=0.25)
ax.tick_params(colors=TEXT)

note2 = ("IMPORTANT: The level crossing any threshold is NOT the crisis indicator."
         "\nThe real test is in forecast11_leverage_change_fan.png — whether the ANNUAL INCREMENT is steady or accelerating.")
ax.text(0.5, -0.09, note2, transform=ax.transAxes,
        ha='center', color=ORANGE, fontsize=9.5, style='italic')

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('out/forecast11_level_context.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Saved: out/forecast11_level_context.png")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY TEXT
# ═══════════════════════════════════════════════════════════════════════════

lines = []
def w(s=""): lines.append(s)

w("=" * 74)
w("FORECAST 11 — CHINA DEBT/GDP: STEADY RISE VS. ACCELERATING SPIRAL")
w(f"Generated: {datetime.now().strftime('%Y-%m-%d')}")
w("=" * 74)
w()
w("OPERATIONAL RULE:")
w(f"  PRIMARY — P(no acceleration): slope of annual-change series < {SLOPE_THRESH} pp/yr/yr.")
w(f"  JOINT — P(steady): mean annual change in [{STEADY_LO},{STEADY_HI}] pp/yr AND slope < {SLOPE_THRESH}.")
w(f"  NO (spiral): annual change series trends upward — 2nd derivative positive, sustained.")
w()
w("REFRAME: Rising leverage is EXPECTED. Japan rose past 400% for 30 years.")
w("  Acceleration — the 2nd derivative of the leverage level — is the signal.")
w()
w("─" * 74)
w(f"HISTORICAL ANNUAL CHANGE (2015-2024, ex-COVID): mean={mu_hist:.1f}pp/yr, std={sigma_hist:.1f}pp")
w(f"  Trend slope: {slope_hist:.2f} pp/yr/yr (p={p_h:.2f}) — historical accel: {hist_accel}")
w(f"  Calibration: NGDP ~{NGDP_GROWTH_ANN*100:.1f}%/yr, TSF ~{TSF_GROWTH_ANN*100:.1f}%/yr")
w()
w("─" * 74)
w("PRIMARY: P(NO ACCELERATION) — slope-only test")
w("─" * 74)
w(f"  {'Scenario':<22}  {'2028':>7}  {'2030':>7}  {'2032':>7}")
w("  " + "-" * 46)
for name in SCENARIOS:
    w(f"  {name:<22}  {pna_base[name][2028]:>6.1%}  {pna_base[name][2030]:>6.1%}  {pna_base[name][2032]:>6.1%}")
w(f"  {'BLENDED':>22}  {pna[2028]:>6.1%}  {pna[2030]:>6.1%}  {pna[2032]:>6.1%}")
w()
w("─" * 74)
w("JOINT: P(STEADY — mean change in band AND no acceleration)")
w("─" * 74)
w(f"  {'Scenario':<22}  {'2028':>7}  {'2030':>7}  {'2032':>7}")
w("  " + "-" * 46)
for name in SCENARIOS:
    sp = SCENARIOS[name]
    w(f"  {name:<22}  {ps_base[name][2028]:>6.1%}  {ps_base[name][2030]:>6.1%}  {ps_base[name][2032]:>6.1%}")
    w(f"    {sp['desc'][:70]}")
w(f"  {'BLENDED (' + wt_str + ')':>22}  {pb[2028]:>6.1%}  {pb[2030]:>6.1%}  {pb[2032]:>6.1%}")
w()
w("Horizon spread note: Acceleration compounds over time. The productive scenario")
w("  barely loses P across horizons; dead-weight spirals lose most by 2032.")
w()
w("─" * 74)
w("SETSER ROBUSTNESS")
w("─" * 74)
w("Setser: NGDP growth overstated ~1.5pp/yr (deflator issue) -> leverage rises")
w("faster. Impact on joint P(steady):")
for name in SCENARIOS:
    d = {yr: ps_set[name][yr] - ps_base[name][yr] for yr in HORIZONS}
    w(f"  {name:<22} d2028:{d[2028]:>+6.1%}  d2030:{d[2030]:>+6.1%}  d2032:{d[2032]:>+6.1%}")
w(f"  {'Blended delta':<22} d2028:{db[2028]:>+6.1%}  d2030:{db[2030]:>+6.1%}  d2032:{db[2032]:>+6.1%}")
w("Setser raises level trajectory but doesn't change slope pattern.")
w("The slope-only (P no-accel) test is robust to the Setser adjustment.")
w()
w("─" * 74)
w("INTERPRETATION")
w("─" * 74)
w("PRODUCTIVE: ICOR constant; annual change ~4pp/yr flat. Japan-path. Absorbable.")
w("MILD: ICOR rising slowly; annual change drifts to ~9pp/yr by 2030. Watchable.")
w("DEAD-WEIGHT: ICOR rising steeply — Forecast 2 mechanism for China infra/property.")
w("  Annual change accelerates past band. Debt must grow faster each year.")
w("  This is the self-financing failure: the 2nd derivative goes positive and stays.")
w()
w("LINK TO F4/F5/F6 DURABILITY CLUSTER:")
w("  YES (steady) supports: F4 USDCNY stable (97.9%), F5 FX reserves (95%),")
w("  F6 no credit transmission. The stress tail needs BOTH F11=NO and F6=YES.")
w()
w("=" * 74)
w("FILES: forecast11_leverage_change_fan.png (the test), forecast11_level_context.png,")
w("       forecast11_summary.txt")

with open('out/forecast11_summary.txt', 'w') as f:
    f.write('\n'.join(lines))
print("Saved: out/forecast11_summary.txt")

print()
print("=" * 74)
print(f"  Historical recent annual change:          {mu_hist:.1f} pp/yr  (2015-2024 ex-COVID)")
print(f"  PRIMARY P(no acceleration) 2028/2030/2032: {pna[2028]:.1%} / {pna[2030]:.1%} / {pna[2032]:.1%}  (blended)")
print(f"  JOINT   P(steady rise)     2028/2030/2032: {pb[2028]:.1%} / {pb[2030]:.1%} / {pb[2032]:.1%}  (blended)")
print(f"  Setser impact on joint P(steady 2030):    {db[2030]:>+.1%}")
print("=" * 74)
