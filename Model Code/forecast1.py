"""
FORECAST 3 (REVISED): US PBS EMPLOYMENT — SUSTAINED BELOW-TREND GROWTH
P(below trend for >=4 consecutive quarters at any point before end-2035)
Event: AI-driven erosion of PBS hiring trend, NOT a disaster-only collapse.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import ndtr as normcdf
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from io import StringIO

OUT_DIR = os.path.join(os.path.dirname(__file__), 'out')
os.makedirs(OUT_DIR, exist_ok=True)

# ── light style ───────────────────────────────────────────────────────────────
C_MED    = '#1565C0'          # dark blue  — median projected path
C_FAN    = '#42A5F5'          # mid blue   — simulation band fill
C_TRD_A  = '#E65100'          # burnt orange — pre-2020 benchmark line
C_TRD_B  = '#6A1B9A'          # purple     — rolling recent average line
C_HIST   = '#455A64'          # slate      — historical data line
C_SHADE  = '#EF9A9A'          # pale red   — below-benchmark shading
GRID_C   = '#E0E0E0'          # light gray grid
TEXT_C   = '#212121'          # near-black text

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        10,
    'axes.facecolor':   'white',
    'figure.facecolor': 'white',
    'axes.edgecolor':   '#BDBDBD',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'grid.color':       GRID_C,
    'grid.linewidth':   0.6,
    'grid.alpha':       1.0,
    'xtick.color':      TEXT_C,
    'ytick.color':      TEXT_C,
    'axes.labelcolor':  TEXT_C,
    'text.color':       TEXT_C,
    'legend.framealpha':0.9,
    'legend.edgecolor': '#BDBDBD',
    'legend.fontsize':  9,
})

def light_fig(w=13, h=6.5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(True, axis='y', zorder=0)
    ax.set_axisbelow(True)
    return fig, ax

# ── parameters ────────────────────────────────────────────────────────────────
N_PATHS       = 10_000
TREND_A_START = '2010-01'
TREND_A_END   = '2019-12'
ROLL_WINDOW   = 60        # Trend-B: 60-month rolling
HORIZON_MO    = 12        # 12 consecutive months = ~4 quarters

# Annual drift scenarios in pp/yr — converted to per-month const drift AFTER model fit
# (because conversion requires fitted AR coefficients)
ANNUAL_PP = {'zero': 0.0, 'mild': 0.05, 'moderate': 0.20, 'aggressive': 0.40}

print("=" * 79)
print("FORECAST 3 (REVISED): PBS EMPLOYMENT — SUSTAINED BELOW-TREND GROWTH")
print("=" * 79)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DOWNLOAD FRED DATA
# ─────────────────────────────────────────────────────────────────────────────
SERIES = {
    'USPBS':    'PBS Employment (000s, SA)',
    'TEMPHELPS':'Temporary Help Services (000s, SA)',
    'JTSJOL':  'Job Openings Total (000s, SA)',
}
raw = {}
for sid, label in SERIES.items():
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}'
    r = requests.get(url, timeout=30, verify=False)
    r.raise_for_status()
    df_s = pd.read_csv(StringIO(r.text))
    df_s.columns = [c.strip() for c in df_s.columns]
    date_col = next(c for c in df_s.columns if 'date' in c.lower())
    val_col  = next(c for c in df_s.columns if c != date_col)
    df_s = df_s.rename(columns={date_col: 'date', val_col: 'value'})
    df_s['date']  = pd.to_datetime(df_s['date'])
    df_s['value'] = pd.to_numeric(df_s['value'], errors='coerce')
    df_s = df_s.set_index('date').dropna().sort_index()
    raw[sid] = df_s
    print(f"\n{sid} — {label}")
    print(f"  First 5:\n{df_s.head().to_string()}")
    print(f"  Last  5:\n{df_s.tail().to_string()}")

pbs      = raw['USPBS']['value']
temph    = raw['TEMPHELPS']['value']
openings = raw['JTSJOL']['value']

pbs_log    = np.log(pbs)
pbs_growth = pbs_log.diff().dropna()

last_date  = pbs_growth.index[-1]
last_month = last_date.strftime('%Y-%m')
print(f"\n*** LAST ACTUAL DATA MONTH: {last_month} ***")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  TREND DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
mask_a     = (pbs_growth.index >= TREND_A_START) & (pbs_growth.index <= TREND_A_END)
mu_trend_A = float(pbs_growth[mask_a].mean())
print(f"\nTREND-A fixed mean (2010-2019): {mu_trend_A*100:.4f}%/mo "
      f"({mu_trend_A*1200:.3f}%/yr)")

trend_B_hist = pbs_growth.rolling(ROLL_WINDOW).mean()
print(f"TREND-B latest rolling mean   : {trend_B_hist.iloc[-1]*100:.4f}%/mo")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  MARKOV-SWITCHING AR(2) FIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 79)
print("MARKOV-SWITCHING AR(2) FIT")
print("=" * 79)

fit_data = pbs_growth['2000-01':]
mod = MarkovAutoregression(fit_data, k_regimes=2, order=2, switching_variance=True)
res = mod.fit(disp=False, maxiter=500)

k_mu  = [res.params[f'const[{k}]']   for k in range(2)]
k_s2  = [res.params[f'sigma2[{k}]']  for k in range(2)]
k_ar1 = [res.params[f'ar.L1[{k}]']  for k in range(2)]
k_ar2 = [res.params[f'ar.L2[{k}]']  for k in range(2)]

expand_regime   = int(np.argmax(k_mu))
contract_regime = 1 - expand_regime

print(f"  Expansion regime index  : {expand_regime}")
print(f"  Contraction regime index: {contract_regime}")
print(f"  Regime means    : {k_mu[0]:.6f}, {k_mu[1]:.6f}")
print(f"  Regime variances: {k_s2[0]:.8f}, {k_s2[1]:.8f}")
print(f"  Regime AR(L1)   : {k_ar1[0]:.4f}, {k_ar1[1]:.4f}")
print(f"  Regime AR(L2)   : {k_ar2[0]:.4f}, {k_ar2[1]:.4f}")

p00 = res.params['p[0->0]']
p10 = res.params['p[1->0]']   # prob FROM regime 1 TO regime 0
p01 = 1 - p00
p11 = 1 - p10

print(f"  Transition: P[0->0]={p00:.4f} P[0->1]={p01:.4f} | P[1->0]={p10:.4f} P[1->1]={p11:.4f}")

# Regime-labelled transition probs (0=expansion, 1=contraction in our sim)
if expand_regime == 0:
    p_exp_to_con = p01
    p_con_to_exp = p10
else:
    p_exp_to_con = p10
    p_con_to_exp = p01

dur_exp = 1.0 / p_exp_to_con
dur_con = 1.0 / p_con_to_exp
print(f"  p(exp->con): {p_exp_to_con:.4f}  dur_exp: {dur_exp:.1f} mo")
print(f"  p(con->exp): {p_con_to_exp:.4f}  dur_con: {dur_con:.1f} mo")

mu0  = k_mu[expand_regime]
mu1  = k_mu[contract_regime]
s0   = np.sqrt(k_s2[expand_regime])
s1   = np.sqrt(k_s2[contract_regime])
ar1_0 = k_ar1[expand_regime];  ar2_0 = k_ar2[expand_regime]
ar1_1 = k_ar1[contract_regime]; ar2_1 = k_ar2[contract_regime]

p_start_exp = float(res.smoothed_marginal_probabilities[expand_regime].iloc[-1])
current_regime = 'expansion' if p_start_exp > 0.5 else 'contraction'
print(f"  Current P(expansion): {p_start_exp:.4f}  → MAP: {current_regime}")

# ── compute DELTAS now that we have AR coefficients ───────────────────────────
# The expansion unconditional mean = mu0 / (1 - ar1_0 - ar2_0).
# To reduce the unconditional ANNUAL growth by delta_annual pp each year
# (linearly accumulated), we need to reduce the AR const by:
#   delta_const = (delta_annual / 100 / 12 / 12) * (1 - ar1_0 - ar2_0)
#                 ──────────────────────────────────
#                 annual→monthly growth, then /12 again because
#                 delta_annual is a RATE per year applied each month
ar_sum = ar1_0 + ar2_0
unc_mean_0_annual = (mu0 / (1 - ar_sum)) * 1200   # %/yr, for printing
DELTAS = {name: (d / 100 / 144) * (1 - ar_sum) for name, d in ANNUAL_PP.items()}

print(f"\n  Expansion unconditional mean : {unc_mean_0_annual:.2f}%/yr")
print(f"  AR sum (ar1+ar2)             : {ar_sum:.4f}")
print(f"  1 - ar_sum                   : {1-ar_sum:.4f}")
print(f"  Trend-A threshold            : {mu_trend_A*1200:.2f}%/yr")
print(f"  Gap (unc_mean - Trend-A)     : {unc_mean_0_annual - mu_trend_A*1200:.2f}pp")
print(f"\n  delta_const per month (const drift):")
for name, d in DELTAS.items():
    ann = ANNUAL_PP[name]
    unc_reduction_10yr = ann * 10  # pp of annual growth rate by 2035
    print(f"    {name:10s}: {d:.3e}/mo  "
          f"(uncond mean at 2035: {unc_mean_0_annual - unc_reduction_10yr:.2f}%/yr)")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  DELTA CALIBRATION OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 79)
print("DELTA CALIBRATION — Eloundou et al. 2023 (GPTs are GPTs)")
print("=" * 79)
print(f"""
  PBS is a high-exposure sector (professional, scientific, technical, admin).
  Eloundou et al.: ~50% of workers have >=10% of tasks LLM-exposed;
                   ~19% have >=50% exposed. PBS roles cluster at the top.

  Conversion to annual pp / yr trend reduction:
    MILD       0.05 pp/yr — 5-10% of exposed tasks displace hiring;
                             firms absorb via attrition.
    MODERATE   0.20 pp/yr — 15-25% task conversion over 10 years;
                             AI augments standard white-collar workflows.
    AGGRESSIVE 0.40 pp/yr — 35-50% conversion; AI agents substitute
                             for hiring across PBS sectors.

  Pre-2020 trend (Trend-A): {mu_trend_A*100:.4f}%/mo = {mu_trend_A*1200:.3f}%/yr.
  Expansion regime unconditional mean: {unc_mean_0_annual:.2f}%/yr.
  (The unconditional mean exceeds Trend-A because the Markov model
  captures post-crisis rebounds; the structural trend is the right
  threshold for "sustained below-trend" in the labour market sense.)

  Under AGGRESSIVE (0.40pp/yr), the expansion unconditional mean falls
  from {unc_mean_0_annual:.2f}%/yr to {unc_mean_0_annual - 4.0:.2f}%/yr by 2035
  vs Trend-A = {mu_trend_A*1200:.2f}%/yr. This is what makes the drift
  mechanistically meaningful.

  NOTE: delta is NOT estimated from history. The series contains no
  AI-displacement episode. These are scenario conversions of Eloundou
  exposure shares — deliberately transparent and adjustable.
""")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 79)
print("MONTE CARLO SIMULATION")
print("=" * 79)

sim_start = last_date + pd.DateOffset(months=1)
dl35 = pd.Timestamp('2035-12-31')
dl30 = pd.Timestamp('2030-12-31')

n35 = int(round((dl35.year - sim_start.year) * 12
                + (dl35.month - sim_start.month) + 1))
n30 = int(round((dl30.year - sim_start.year) * 12
                + (dl30.month - sim_start.month) + 1))

print(f"  Sim start : {sim_start.strftime('%Y-%m')}")
print(f"  n35={n35} months  n30={n30} months")

lag1_seed = float(pbs_growth.iloc[-1])
lag2_seed = float(pbs_growth.iloc[-2])
trend_b_seed = list(pbs_growth.values[-ROLL_WINDOW:])


def simulate_paths(delta, n_months, seed=42):
    rng     = np.random.default_rng(seed)
    u_init  = rng.random(N_PATHS)
    reg_cur = np.where(u_init < p_start_exp, 0, 1).astype(np.int8)
    lag1    = np.full(N_PATHS, lag1_seed)
    lag2    = np.full(N_PATHS, lag2_seed)
    out     = np.empty((N_PATHS, n_months), dtype=np.float64)
    for t in range(n_months):
        mu0_t    = mu0 - delta * t
        mu_path  = np.where(reg_cur == 0,
                            mu0_t + ar1_0 * lag1 + ar2_0 * lag2,
                            mu1   + ar1_1 * lag1 + ar2_1 * lag2)
        sig_path = np.where(reg_cur == 0, s0, s1)
        eps      = rng.standard_normal(N_PATHS)
        g        = mu_path + sig_path * eps
        out[:, t] = g
        u_tr    = rng.random(N_PATHS)
        new_con = np.where(reg_cur == 0,
                           u_tr < p_exp_to_con,
                           u_tr >= p_con_to_exp)
        reg_cur = new_con.astype(np.int8)
        lag2 = lag1.copy()
        lag1 = g.copy()
    return out


def p_event(paths, trend_choice, n_cap):
    N, T = paths.shape
    T    = min(T, n_cap)
    sub  = paths[:, :T]
    hit  = np.zeros(N, dtype=bool)
    for i in range(N):
        path  = sub[i]
        buf   = list(trend_b_seed)
        consec = 0
        for g in path:
            if trend_choice == 'A':
                thr = mu_trend_A
            else:
                buf.append(g)
                if len(buf) > ROLL_WINDOW:
                    buf.pop(0)
                thr = np.mean(buf)
            if g < thr:
                consec += 1
                if consec >= HORIZON_MO:
                    hit[i] = True
                    break
            else:
                consec = 0
    return float(hit.sum()) / N


results = {}
for dname, dval in DELTAS.items():
    print(f"  Simulating delta={dname} ({ANNUAL_PP[dname]:.2f} pp/yr) …", flush=True)
    paths35 = simulate_paths(dval, n35)
    for tr in ('A', 'B'):
        p35 = p_event(paths35, tr, n35)
        p30 = p_event(paths35, tr, n30)
        results[(dname, tr)] = (p35, p30)
        print(f"    Trend-{tr}: P(2035)={p35:.4f}  P(2030)={p30:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PROBIT CROSS-CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 79)
print("PROBIT CROSS-CHECK")
print("=" * 79)

pbs_q   = pbs.resample('QS').last()
temph_q = temph.resample('QS').last()
open_q  = openings.resample('QS').last()
common  = pbs_q.index.intersection(temph_q.index).intersection(open_q.index)
pbs_q   = pbs_q.reindex(common)
temph_q = temph_q.reindex(common)
open_q  = open_q.reindex(common)

pbs_yoy_log = np.log(pbs_q).diff(4)
mu_A_yoy    = mu_trend_A * 12
below_A     = (pbs_yoy_log < mu_A_yoy).astype(int)

df_prob = pd.DataFrame({
    'below': below_A,
    'th_yoy': temph_q.pct_change(4),
    'jol_yoy': open_q.pct_change(4),
}).dropna()
df_prob['target'] = df_prob['below'].shift(-4)
df_prob = df_prob.dropna()

X_const = sm.add_constant(df_prob[['th_yoy', 'jol_yoy']].values)
probit_p = float('nan')
try:
    pr = sm.Probit(df_prob['target'].values, X_const).fit(disp=False, maxiter=200)
    beta = pr.params
    temph_yoy_cur = float((temph_q.iloc[-1] - temph_q.iloc[-5]) / temph_q.iloc[-5])
    jol_yoy_cur   = float((open_q.iloc[-1]  - open_q.iloc[-5])  / open_q.iloc[-5])
    x_cur = np.array([1.0, temph_yoy_cur, jol_yoy_cur])
    probit_p = float(normcdf(float(x_cur @ beta)))
    print(f"  TEMPHELPS YoY current: {temph_yoy_cur:.3f}")
    print(f"  JTSJOL YoY current   : {jol_yoy_cur:.3f}")
    print(f"  Probit P(below-trend next 4Q): {probit_p:.3f}")
except Exception as e:
    print(f"  Probit failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PROBABILITY SURFACE TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 79)
print("PROBABILITY SURFACE")
print("=" * 79)
print(f"\n{'Delta':12s}  {'Ann.Δ':7s}  "
      f"{'TrA-2030':10s}  {'TrA-2035':10s}  {'TrB-2030':10s}  {'TrB-2035':10s}")
print("-" * 68)

ORDER = ['zero', 'mild', 'moderate', 'aggressive']
rows = []
for dname in ORDER:
    p35a, p30a = results[(dname, 'A')]
    p35b, p30b = results[(dname, 'B')]
    ann = ANNUAL_PP[dname]
    print(f"{dname:12s}  {ann:+.2f}pp/yr  "
          f"{p30a:10.4f}  {p35a:10.4f}  {p30b:10.4f}  {p35b:10.4f}")
    rows.append({'delta_scenario': dname, 'annual_delta_pp': ann,
                 'P_TrendA_2030': round(p30a, 4), 'P_TrendA_2035': round(p35a, 4),
                 'P_TrendB_2030': round(p30b, 4), 'P_TrendB_2035': round(p35b, 4)})

pd.DataFrame(rows).to_csv(
    os.path.join(OUT_DIR, 'forecast3_trend_probability_surface.csv'), index=False)
print("Saved: out/forecast3_trend_probability_surface.csv")

p_zero_a35 = results[('zero',     'A')][0]
p_zero_b35 = results[('zero',     'B')][0]
p_mod_a35  = results[('moderate', 'A')][0]
p_mod_b35  = results[('moderate', 'B')][0]
p_agg_a35  = results[('aggressive','A')][0]
p_mild_a35 = results[('mild',     'A')][0]

# ─────────────────────────────────────────────────────────────────────────────
# 8.  FAN CHART
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating fan chart …")
paths_fan  = simulate_paths(DELTAS['moderate'], n35, seed=99)
sim_dates  = pd.date_range(sim_start, periods=n35, freq='MS')
pct        = np.percentile(paths_fan, [5, 25, 50, 75, 95], axis=0)

hist_g      = pbs_growth['2000-01':]
trend_b_h   = trend_B_hist['2000-01':]

fig, ax = light_fig(14, 7)

# ── historical data ──
ax.plot(hist_g.index, hist_g.values * 100,
        color=C_HIST, lw=0.9, alpha=0.65, zorder=3,
        label='Historical monthly growth (actual)')

# ── benchmark lines ──
tA_label = (f'Pre-2020 benchmark  '
            f'({mu_trend_A*100:.2f}%/mo, 2010–2019 avg)')
ax.axhline(mu_trend_A * 100, color=C_TRD_A, lw=2.0, ls='--',
           zorder=4, label=tA_label)
ax.plot(trend_b_h.index, trend_b_h.values * 100,
        color=C_TRD_B, lw=1.4, ls='-.', alpha=0.85, zorder=4,
        label='Recent rolling average (5-year window)')

# ── simulation fan: moderate AI-impact scenario ──
ax.fill_between(sim_dates, pct[0]*100, pct[4]*100,
                color=C_FAN, alpha=0.12, zorder=2,
                label='Simulated range — 5th to 95th percentile')
ax.fill_between(sim_dates, pct[1]*100, pct[3]*100,
                color=C_FAN, alpha=0.26, zorder=2,
                label='Simulated range — 25th to 75th percentile')
ax.plot(sim_dates, pct[2]*100,
        color=C_MED, lw=2.2, zorder=5,
        label='Median simulated path')

# ── shade region where median dips below benchmark ──
below = pct[2] < mu_trend_A
if below.any():
    ax.fill_between(sim_dates, -1.2, pct[2]*100,
                    where=below, color=C_SHADE, alpha=0.30, zorder=1,
                    label='Median falls below pre-2020 benchmark')

# ── "data ends here" divider ──
ax.axvline(last_date, color='#757575', lw=1.2, ls=':', zorder=6)
ax.text(last_date + pd.DateOffset(months=2),  0.72,
        f'Data through\n{last_month}',
        color='#757575', fontsize=8.5, va='top',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#BDBDBD', alpha=0.9))

# ── zero line ──
ax.axhline(0, color='#9E9E9E', lw=0.8, ls='-', alpha=0.5, zorder=1)

ax.set_xlabel('Year', fontsize=11, labelpad=6)
ax.set_ylabel('Monthly growth rate  (%)', fontsize=11, labelpad=6)
ax.set_title(
    'PBS Professional Services Employment — Monthly Growth Rate\n'
    'Simulated paths under moderate AI-displacement assumption  '
    '(trend drift = 0.20 pp/yr)',
    fontsize=12, fontweight='bold', pad=12, color=TEXT_C)
ax.legend(loc='upper left', ncol=2, fontsize=8.5,
          facecolor='white', edgecolor='#BDBDBD')
ax.set_xlim(pd.Timestamp('2000-01-01'), pd.Timestamp('2036-06-01'))
ax.set_ylim(-1.2, 0.9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+.1f}%'))
fig.text(0.99, 0.01,
         'Simulation: Markov-switching AR(2) + structural AI drift  |  '
         'Benchmark = pre-2020 monthly average  |  12-month window for event trigger',
         ha='right', va='bottom', fontsize=7.5, color='#757575', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'forecast3_fan_chart.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: out/forecast3_fan_chart.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  DELTA SENSITIVITY CHART
# ─────────────────────────────────────────────────────────────────────────────
print("Generating delta-sensitivity chart …")
n_sweep = 16
delta_sweep = np.linspace(0, 0.50, n_sweep)
p_A35, p_B35, p_A30, p_B30 = [], [], [], []

for d_ann in delta_sweep:
    # FIX: use the same conversion as DELTAS dict — must include (1-ar_sum)
    # factor that maps unconditional-mean drift to AR intercept drift.
    # Prior code used d_ann/12/100, which was ~53x too large (missing the
    # (1-ar_sum)≈0.228 factor AND dividing by 12 instead of 144).
    d_mo = (d_ann / 100 / 144) * (1 - ar_sum)
    pts  = simulate_paths(d_mo, n35)
    p_A35.append(p_event(pts, 'A', n35))
    p_B35.append(p_event(pts, 'B', n35))
    p_A30.append(p_event(pts, 'A', n30))
    p_B30.append(p_event(pts, 'B', n30))
    print(f"  δ={d_ann:.2f}pp/yr  A35={p_A35[-1]:.3f}  B35={p_B35[-1]:.3f}", flush=True)

fig2, ax2 = light_fig(12, 6.5)

# ── curves ──
ax2.plot(delta_sweep, [v*100 for v in p_A35],
         color=C_TRD_A, lw=2.5, zorder=4,
         label='vs. pre-2020 benchmark — by end-2035')
ax2.plot(delta_sweep, [v*100 for v in p_B35],
         color=C_TRD_B, lw=2.5, zorder=4,
         label='vs. recent rolling average — by end-2035')
ax2.plot(delta_sweep, [v*100 for v in p_A30],
         color=C_TRD_A, lw=1.8, ls='--', zorder=3,
         label='vs. pre-2020 benchmark — by end-2030')
ax2.plot(delta_sweep, [v*100 for v in p_B30],
         color=C_TRD_B, lw=1.8, ls='--', zorder=3,
         label='vs. recent rolling average — by end-2030')

# ── named scenario markers ──
scenarios_ref = [
    ('Mild\n(0.05 pp/yr)', 0.05,  '#4CAF50'),
    ('Moderate\n(0.20 pp/yr)', 0.20, '#FF9800'),
    ('Aggressive\n(0.40 pp/yr)', 0.40, '#F44336'),
]
for lbl, xv, col in scenarios_ref:
    ax2.axvline(xv, color=col, lw=1.2, ls=':', alpha=0.7, zorder=2)
    ax2.text(xv, 97, lbl,
             ha='center', va='top', fontsize=8.5, color=col,
             fontweight='semibold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=col, alpha=0.85))

# ── headline dot: moderate δ, Trend-A, 2035 ──
mod_idx = np.argmin(np.abs(delta_sweep - 0.20))
hl_y    = p_A35[mod_idx] * 100
ax2.scatter([0.20], [hl_y], color=C_TRD_A, s=90, zorder=6)
ax2.annotate(f'Headline: {hl_y:.0f}%\n(moderate, 2035)',
             xy=(0.20, hl_y), xytext=(0.27, hl_y + 12),
             arrowprops=dict(arrowstyle='->', color=C_TRD_A, lw=1.2),
             fontsize=8.5, color=C_TRD_A,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_TRD_A, alpha=0.9))

ax2.set_xlabel(
    'Assumed annual AI impact on hiring trend  (percentage points per year reduction)',
    fontsize=11, labelpad=8)
ax2.set_ylabel(
    'Probability of event occurring  (%)',
    fontsize=11, labelpad=8)
ax2.set_title(
    'How Strong AI Adoption Needs to Be to Cause Sustained PBS Underperformance\n'
    'P(growth stays below benchmark for 12+ consecutive months, at any point before horizon)',
    fontsize=11, fontweight='bold', pad=12, color=TEXT_C)
ax2.legend(loc='upper left', fontsize=9, facecolor='white', edgecolor='#BDBDBD')
ax2.set_xlim(0, 0.50)
ax2.set_ylim(0, 100)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
fig2.text(0.99, 0.01,
          'Benchmark (solid lines) = pre-2020 fixed avg  |  '
          'Rolling avg (dashed lines) adapts with recent data  |  '
          'Solid = 2035 horizon, dashed = 2030',
          ha='right', va='bottom', fontsize=7.5, color='#757575', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'forecast3_delta_sensitivity.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: out/forecast3_delta_sensitivity.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  SUMMARY TEXT
# ─────────────────────────────────────────────────────────────────────────────
trend_b_dir = 'lower' if p_mod_b35 < p_mod_a35 else 'higher'

summary = f"""FORECAST 3 (REVISED): PBS EMPLOYMENT — SUSTAINED BELOW-TREND GROWTH
Generated: {pd.Timestamp('today').strftime('%Y-%m-%d')}
=======================================================================

EVENT DEFINITION
=======================================================================
  P(US PBS employment MoM log-growth falls BELOW TREND for >=12
  consecutive months at any point before end-2035, in eligible
  future months only)

  Last actual data month: {last_month}
  First eligible month  : {sim_start.strftime('%Y-%m')}

WHY THIS IS NOT JUST DISASTER FREQUENCY
=======================================================================
  A standard Markov-switching model extended to 2035 measures only how
  often the disaster regime fires by chance. That is NOT the AI thesis.
  The thesis is structural: PBS expansion-regime trend growth gradually
  DECLINES as AI agents absorb white-collar tasks. We encode this as:

    mu_expansion(t) = mu0 - delta * t

  where t is months from last actual data and delta is calibrated from
  Eloundou et al. (2023) occupation-level LLM-exposure data.

DELTA CALIBRATION (Eloundou et al. 2023, "GPTs are GPTs")
=======================================================================
  PBS is a high-exposure sector (professional/scientific/technical/admin).
  ~50% of US workers have >=10% of tasks LLM-exposed; PBS clusters high.

  MILD       0.05 pp/yr — 5-10% of exposed tasks displace hiring;
                           firms absorb via attrition. Minimal disruption.
  MODERATE   0.20 pp/yr — 15-25% task conversion over 10 years.
                           AI augments standard white-collar workflows.
  AGGRESSIVE 0.40 pp/yr — 35-50% conversion. AI agents materially
                           substitute for hiring across PBS sectors.

  Pre-2020 trend (Trend-A): {mu_trend_A*100:.4f}%/mo = {mu_trend_A*1200:.3f}%/yr.
  Under AGGRESSIVE delta, trend declines 4.0pp total by 2035.

  NOTE: delta cannot be estimated from history — no AI-displacement
  episode has appeared in the series. These are SCENARIO CONVERSIONS
  of Eloundou exposure shares, not data-estimated parameters.

MARKOV-SWITCHING AR(2) PARAMETERS
=======================================================================
  Expansion : const={mu0:.5f}, AR1={ar1_0:.4f}, AR2={ar2_0:.4f}, σ={s0:.5f}
  Contraction: const={mu1:.5f}, AR1={ar1_1:.4f}, AR2={ar2_1:.4f}, σ={s1:.5f}
  p(expansion→contraction): {p_exp_to_con:.4f}  (expected duration: {dur_exp:.1f} mo)
  p(contraction→expansion): {p_con_to_exp:.4f}  (expected duration: {dur_con:.1f} mo)
  Current regime (MAP): {current_regime}  [P(expansion)={p_start_exp:.4f}]

TREND DEFINITIONS
=======================================================================
  TREND-A (fixed): 2010-01 to 2019-12 mean MoM log-growth
    = {mu_trend_A*100:.5f}%/mo  ({mu_trend_A*1200:.3f}%/yr)
  TREND-B (rolling): trailing 60-month moving average (latest: {trend_B_hist.iloc[-1]*100:.5f}%/mo)

PROBABILITY SURFACE — P(event before deadline)
=======================================================================
  Delta         Ann.Δ   TrendA-2030  TrendA-2035  TrendB-2030  TrendB-2035
  ─────────────────────────────────────────────────────────────────────────"""

for dname in ORDER:
    p35a, p30a = results[(dname, 'A')]
    p35b, p30b = results[(dname, 'B')]
    ann = ANNUAL_PP[dname]
    summary += (f"\n  {dname:12s}  {ann:+.2f}pp/yr  "
                f"{p30a:.4f}       {p35a:.4f}       {p30b:.4f}       {p35b:.4f}")

p_zero_30a = results[('zero', 'A')][1]
summary += f"""

HEADLINE INTERPRETATION
=======================================================================
  DELTA=0 BASELINE (pure disaster luck, NOT the AI thesis):
    P(2035, Trend-A) = {p_zero_a35:.3f}  |  P(2035, Trend-B) = {p_zero_b35:.3f}
    This measures only how often the Markov disaster regime fires for
    >=12 consecutive months by chance over a 10-year window.

  MODERATE DELTA (core AI-displacement case):
    P(2035, Trend-A) = {p_mod_a35:.3f}  |  P(2035, Trend-B) = {p_mod_b35:.3f}
    Increment over delta=0: +{p_mod_a35-p_zero_a35:.3f} pp (Trend-A).
    This increment is STRUCTURAL DRIFT — probability from AI trend
    erosion, not disaster-regime luck.

  Trend-B probabilities are {trend_b_dir} than Trend-A because the
  rolling threshold also declines with the actual series, making it
  {'harder' if trend_b_dir == 'lower' else 'easier'} to trigger sustained below-trend growth.

PROBIT CROSS-CHECK
=======================================================================
  Probit of P(below-trend next 4Q) on TEMPHELPS YoY + JTSJOL YoY:
  Current-data-implied P = {probit_p:.3f}
  (A low value confirms current PBS hiring strength.)

WHAT DRIVES THE NUMBER
=======================================================================
  1. BASELINE DISASTER FREQUENCY: The contraction regime fires
     ~{dur_con:.0f}-month episodes; over 10 years, cumulative P ≈ {p_zero_a35:.0%}.

  2. STRUCTURAL DRIFT: MODERATE delta adds ~+{p_mod_a35-p_zero_a35:.0%} by 2035.
     This is the probability that comes from AI compression of the
     expansion-regime trend — even without a classic recession.

  3. HORIZON SENSITIVITY: The 2030 vs. 2035 gap reflects when drift
     becomes large enough to push the expansion mean below the threshold.
     Under mild delta, the gap is small; under aggressive delta, it is
     substantial because 10yr of drift has fully depressed the mean.
"""

with open(os.path.join(OUT_DIR, 'forecast3_summary.txt'), 'w') as f:
    f.write(summary)
print("Saved: out/forecast3_summary.txt")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 79)
print("FINAL STDOUT SUMMARY")
print("=" * 79)
print(f"  LAST ACTUAL DATA MONTH              : {last_month}")
print(f"  DELTA=0 BASELINE  P(event,2035)     : {p_zero_a35:.3f}  (Trend-A, pure disaster luck)")
print(f"  MODERATE DELTA    P(event,2035)     : {p_mod_a35:.3f}  (Trend-A, HEADLINE)")
print(f"  MODERATE DELTA    P(event,2035)     : {p_mod_b35:.3f}  (Trend-B)")
print(f"  PROBIT CROSS-CHECK P(below-trend 4Q): {probit_p:.3f}")
print("=" * 79)
print("All outputs saved to ./out/")
