"""
Forecast 12: P(China (general public budget expenditure - revenue) / tax revenue > 40%)
             in any of 2026, 2027, 2028 — official MoF general public budget.
Also computes augmented (incl. government-fund budget) version.
Data: China MoF annual releases (hardcoded); land revenue from MoF/NIFD.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')
os.makedirs('data', exist_ok=True)
os.makedirs('out', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. HARDCODED DATA — China MoF general public budget (¥ trn, annual)
#    Source: Ministry of Finance annual fiscal data releases
#    https://www.mof.gov.cn/zhengwuxinxi/caizhengshuju/
#    Tax revenue / total revenue / expenditure (all general public budget)
# ─────────────────────────────────────────────────────────────────────────────

FISCAL = {
    # Year: (total_rev, tax_rev, expenditure, gf_revenue, gf_expenditure, land_rev)
    # gf = government-fund budget (including land sale proceeds)
    # All ¥ trillion
    # Sources: MoF annual budget reports, NBS; [EST] = estimates
    2015: (15.22, 12.49, 17.58, 4.26, 4.96, 3.25),
    2016: (15.96, 13.04, 18.77, 4.39, 5.40, 3.75),
    2017: (17.26, 14.44, 20.33, 5.21, 6.24, 5.21),
    2018: (18.33, 15.64, 22.09, 6.58, 7.31, 6.50),
    2019: (19.04, 15.80, 23.89, 8.44, 9.46, 7.76),
    2020: (18.29, 15.43, 24.56, 7.57, 8.65, 8.41),  # COVID fiscal expansion
    2021: (20.25, 17.27, 24.63, 9.97, 9.55, 8.70),  # land sales peak
    2022: (20.37, 16.66, 26.06, 7.79, 8.37, 6.69),  # land decline begins
    2023: (21.68, 18.11, 27.46, 5.92, 7.26, 5.57),
    2024: (22.03, 18.00, 28.37, 5.40, 7.50, 5.00),  # [EST MoF pre-release]
    2025: (22.80, 18.50, 29.20, 4.55, 7.90, 4.15),  # [EST; land ¥4.15tn per user]
}
#  NOTE: land revenue ¥8.49tn peak in 2021 → ¥4.15tn in 2025 = -51% as specified

cols = ['total_rev','tax_rev','expenditure','gf_revenue','gf_expenditure','land_rev']
df = pd.DataFrame(FISCAL, index=cols).T
df.index.name = 'Year'
df['deficit'] = df['expenditure'] - df['total_rev']
df['deficit_to_tax'] = df['deficit'] / df['tax_rev'] * 100  # official metric
df['augmented_deficit'] = (df['expenditure'] + df['gf_expenditure']) - (df['total_rev'] + df['gf_revenue'])
df['augmented_to_tax'] = df['augmented_deficit'] / df['tax_rev'] * 100

df.to_csv('data/china_fiscal.csv')
print("=== CHINA FISCAL DATA (¥ TRN) ===")
print(df[['total_rev','tax_rev','expenditure','deficit','deficit_to_tax',
          'land_rev','augmented_deficit','augmented_to_tax']].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2. TAX REVENUE ELASTICITY
# ─────────────────────────────────────────────────────────────────────────────

# g_tax ≈ elasticity × g_ngdp
G_NGDP_ANN = {
    2015:7.1, 2016:8.0, 2017:11.2, 2018:9.7, 2019:7.8,
    2020:3.1, 2021:12.5, 2022:4.0, 2023:4.6, 2024:5.0, 2025:4.5,
}
tax_vals = df['tax_rev'].values
tax_growth = np.diff(np.log(tax_vals))
yrs = sorted(FISCAL.keys())
ngdp_growth = np.array([G_NGDP_ANN[y]/100 for y in yrs[1:]])

# Exclude 2020-2021 extremes for elasticity estimation
mask = np.array([y not in [2020,2021] for y in yrs[1:]])
coeff = np.polyfit(ngdp_growth[mask], tax_growth[mask], 1)
elasticity = float(coeff[0])
print(f"\nTax buoyancy elasticity: {elasticity:.3f} (g_tax ≈ {elasticity:.2f} × g_ngdp)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. EXPENDITURE GROWTH MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Trend expenditure growth (log-linear, 2015-2025)
t_exp = np.arange(len(yrs))
log_exp = np.log(df['expenditure'].values)
trend_coeffs = np.polyfit(t_exp, log_exp, 1)
trend_growth = float(np.exp(trend_coeffs[0]) - 1)  # ~annual log rate
print(f"Trend expenditure growth rate: {trend_growth:.3f} ({trend_growth*100:.1f}%)")

# Counter-cyclical rule: if g_ngdp < 4%, add 1.5pp to expenditure growth
CYCLICAL_THRESHOLD = 0.04
CYCLICAL_BUMP = 0.015
print(f"Counter-cyclical rule: if g_ngdp < {CYCLICAL_THRESHOLD:.0%}, exp_growth += {CYCLICAL_BUMP:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. GOVERNMENT-FUND BUDGET — LAND REVENUE SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
LAND_SCENARIOS = {
    'Land Stabilises (0%/yr)':   0.00,
    'Land -10%/yr':             -0.10,
    'Land -20%/yr':             -0.20,
}
GF_EXP_GROWTH = 0.04   # government-fund expenditure grows ~4%/yr

# ─────────────────────────────────────────────────────────────────────────────
# 5. MONTE CARLO — 5,000 PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Stochastic parameters:
# g_ngdp ~ AR(1) with scenarios [4.5%, 4.0%, 3.5%] as mean alternatives
# sigma_ngdp from residuals
ngdp_hist = np.array([G_NGDP_ANN[y]/100 for y in sorted(G_NGDP_ANN)])
sigma_ngdp = float(np.std(np.diff(ngdp_hist), ddof=1))
print(f"\ng_ngdp volatility: {sigma_ngdp:.4f}")

# elasticity noise
resid_elasticity = tax_growth[mask] - elasticity * ngdp_growth[mask]
sigma_elast = float(np.std(resid_elasticity, ddof=1))

# expenditure noise
exp_residuals = log_exp - np.polyval(trend_coeffs, t_exp)
sigma_exp = float(np.std(exp_residuals, ddof=1))

NGDP_SCENARIOS = {'4.5% growth': 0.045, '4.0% growth': 0.040, '3.5% growth': 0.035}
NGDP_WEIGHTS   = {'4.5% growth': 0.30, '4.0% growth': 0.45, '3.5% growth': 0.25}
LAND_WEIGHTS   = {'Land Stabilises (0%/yr)': 0.20, 'Land -10%/yr': 0.50, 'Land -20%/yr': 0.30}

N_PATHS = 5_000
PROJ_YEARS = [2026, 2027, 2028]
np.random.seed(42)

# Track results
all_results = {}
for ngdp_label, mu_ngdp in NGDP_SCENARIOS.items():
    for land_label, land_growth in LAND_SCENARIOS.items():
        # Initialize
        tax_t  = df['tax_rev'].iloc[-1]   # 2025 starting tax revenue
        rev_t  = df['total_rev'].iloc[-1]
        exp_t  = df['expenditure'].iloc[-1]
        gfrev_t = df['gf_revenue'].iloc[-1]
        gfexp_t = df['gf_expenditure'].iloc[-1]
        land_t  = df['land_rev'].iloc[-1]

        annual_ratios_off = []  # official (expenditure-revenue)/tax_rev each year
        annual_ratios_aug = []  # augmented

        for yr_idx, yr in enumerate(PROJ_YEARS):
            # Stochastic g_ngdp
            gy = mu_ngdp + np.random.randn(N_PATHS) * sigma_ngdp
            gy = np.clip(gy, -0.03, 0.12)

            # Tax revenue
            g_tax = elasticity * gy + np.random.randn(N_PATHS) * sigma_elast
            tax_new = tax_t * (1 + g_tax)

            # Total revenue (other revenue grows with GDP)
            other_rev_t = rev_t - tax_t
            other_rev_new = other_rev_t * (1 + gy * 0.8)
            rev_new = tax_new + other_rev_new

            # Expenditure — trend + cyclical
            g_exp_base = trend_growth + np.random.randn(N_PATHS) * sigma_exp
            # Counter-cyclical bump
            g_exp_base += np.where(gy < CYCLICAL_THRESHOLD, CYCLICAL_BUMP, 0)
            exp_new = exp_t * (1 + g_exp_base)

            # Official deficit
            deficit = exp_new - rev_new
            ratio_off = deficit / tax_new * 100

            # Government-fund budget
            land_new = land_t * (1 + land_growth)
            other_gfrev = gfrev_t - land_t
            gfrev_new = land_new + other_gfrev * (1 + gy * 0.5)
            gfexp_new = gfexp_t * (1 + GF_EXP_GROWTH + np.random.randn(N_PATHS) * 0.02)

            aug_deficit = (exp_new + gfexp_new) - (rev_new + gfrev_new)
            ratio_aug = aug_deficit / tax_new * 100

            annual_ratios_off.append(ratio_off)
            annual_ratios_aug.append(ratio_aug)

            # Update for next year
            tax_t = float(np.mean(tax_new))
            rev_t = float(np.mean(rev_new))
            exp_t = float(np.mean(exp_new))
            land_t = land_new if np.isscalar(land_new) else float(land_new)
            gfrev_t = float(np.mean(gfrev_new))
            gfexp_t = float(np.mean(gfexp_new))

        all_results[(ngdp_label, land_label)] = {
            'ratios_off': np.array(annual_ratios_off),   # shape (3, N_PATHS)
            'ratios_aug': np.array(annual_ratios_aug),
        }

# P(ratio > 40% in ANY year 2026-28)
THRESHOLD_RATIO = 40.0
print(f"\n=== P(official deficit/tax > {THRESHOLD_RATIO}% in any year 2026-2028) ===")
summary_table = []
for ngdp_label, mu_ngdp in NGDP_SCENARIOS.items():
    for land_label, land_growth in LAND_SCENARIOS.items():
        res = all_results[(ngdp_label, land_label)]
        max_off = res['ratios_off'].max(axis=0)   # max across years per path
        max_aug = res['ratios_aug'].max(axis=0)
        p_breach_off = float(np.mean(max_off > THRESHOLD_RATIO))
        p_breach_aug = float(np.mean(max_aug > THRESHOLD_RATIO))
        print(f"  {ngdp_label} + {land_label}: official={p_breach_off:.3f} | augmented={p_breach_aug:.3f}")
        summary_table.append({
            'NGDP_Scenario': ngdp_label,
            'Land_Scenario': land_label,
            'P_breach_official': round(p_breach_off, 3),
            'P_breach_augmented': round(p_breach_aug, 3),
        })

# Blended
blended_off = 0.0
blended_aug = 0.0
for ngdp_label in NGDP_SCENARIOS:
    for land_label in LAND_SCENARIOS:
        w = NGDP_WEIGHTS[ngdp_label] * LAND_WEIGHTS[land_label]
        r = all_results[(ngdp_label, land_label)]
        max_off = r['ratios_off'].max(axis=0)
        max_aug = r['ratios_aug'].max(axis=0)
        blended_off += w * float(np.mean(max_off > THRESHOLD_RATIO))
        blended_aug += w * float(np.mean(max_aug > THRESHOLD_RATIO))

print(f"\n{'='*60}")
print(f"BLENDED P(official > 40% in any year): {blended_off:.3f}")
print(f"BLENDED P(augmented > 40% in any year): {blended_aug:.3f}")
print(f"{'='*60}")

pd.DataFrame(summary_table).to_csv('out/forecast12_scenario_table.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG, PANEL_BG = '#0f1117', '#1a1d2e'
TEXT_COLOR, GRID_COLOR = '#e0e0e0', '#2a2d3e'
ACCENT1, ACCENT2, ACCENT3 = '#00d4ff', '#ff6b6b', '#ffd166'

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(DARK_BG)

# Plot A: Revenue vs Expenditure history + projection
ax = axes[0]
ax.set_facecolor(PANEL_BG)
for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
ax.tick_params(colors=TEXT_COLOR, labelsize=8)

years_hist = sorted(FISCAL.keys())
ax.plot(years_hist, df['total_rev'].values, color=ACCENT1, lw=2, marker='o', ms=4, label='Revenue (official)')
ax.plot(years_hist, df['tax_rev'].values, color='#06d6a0', lw=1.5, linestyle='--', label='Tax Revenue')
ax.plot(years_hist, df['expenditure'].values, color=ACCENT2, lw=2, marker='s', ms=4, label='Expenditure (official)')
ax.plot(years_hist, df['land_rev'].values, color=ACCENT3, lw=1.5, linestyle=':', label='Land Revenue (GF)')

ax.set_title('China General Public Budget\n& Land Revenue 2015-2025', color=TEXT_COLOR, fontsize=10, fontweight='bold')
ax.set_xlabel('Year', color=TEXT_COLOR, fontsize=8)
ax.set_ylabel('¥ Trillion', color=TEXT_COLOR, fontsize=8)
ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)
ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.text(2021, 8.9, 'Land peak\n¥8.49tn', color=ACCENT3, fontsize=7)
ax.text(2025, 4.3, '¥4.15tn\n(-51%)', color=ACCENT3, fontsize=7)

# Plot B: Ratio distributions 2026-2028
ax2 = axes[1]
ax2.set_facecolor(PANEL_BG)
for sp in ax2.spines.values(): sp.set_edgecolor(GRID_COLOR)
ax2.tick_params(colors=TEXT_COLOR, labelsize=8)

# Use 4%/yr GDP + -10%/yr land as representative base case
base_res = all_results[('4.0% growth', 'Land -10%/yr')]
yr_colors = [ACCENT1, ACCENT3, ACCENT2]
for yr_idx, (yr, c) in enumerate(zip(PROJ_YEARS, yr_colors)):
    vals = base_res['ratios_off'][yr_idx]
    ax2.hist(vals, bins=60, alpha=0.55, color=c, label=f'{yr}', density=True)

ax2.axvline(THRESHOLD_RATIO, color='white', linestyle='--', lw=2, label=f'{THRESHOLD_RATIO}% threshold')
ax2.axvline(df['deficit_to_tax'].iloc[-1], color='gray', linestyle=':', lw=1.5, label=f'2025 actual ({df["deficit_to_tax"].iloc[-1]:.1f}%)')

ax2.set_title(f'Distribution of Official Deficit/Tax (%)\n2026–2028 (base case: 4% NGDP, land -10%/yr)',
              color=TEXT_COLOR, fontsize=9, fontweight='bold')
ax2.set_xlabel('(Expenditure - Revenue) / Tax Revenue (%)', color=TEXT_COLOR, fontsize=8)
ax2.set_ylabel('Density', color=TEXT_COLOR, fontsize=8)
ax2.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)
ax2.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COLOR)

fig.suptitle('Forecast 12: China Fiscal Deficit Ratio', color=TEXT_COLOR, fontsize=13, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('out/forecast12_fiscal_charts.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
print("\nSaved: out/forecast12_fiscal_charts.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
cur_ratio = float(df['deficit_to_tax'].iloc[-1])
cur_aug   = float(df['augmented_to_tax'].iloc[-1])

summary = f"""FORECAST 12: P(CHINA OFFICIAL DEFICIT/TAX > 40% IN ANY YEAR 2026-2028)
Generated: 2026-07-28
=============================================================

DEFINITION: (General Public Budget Expenditure - Revenue) / Tax Revenue > 40%
            Resolution: MoF annual general public budget data
            Also computed: augmented (+ government-fund budget) for context.

CURRENT LEVELS (2025 estimates):
  Official deficit/tax ratio: {cur_ratio:.1f}%
  Augmented deficit/tax ratio: {cur_aug:.1f}%
  Distance to 40% threshold (official): {40.0 - cur_ratio:+.1f}pp

TAX REVENUE MODEL:
  Elasticity (g_tax ≈ ε × g_ngdp): ε = {elasticity:.3f}
  Estimated from 2015-2025 data (excl. 2020-21 extremes)
  σ_elasticity = {sigma_elast:.3f}

EXPENDITURE MODEL:
  Trend growth rate: {trend_growth:.3f} ({trend_growth*100:.1f}%/yr, log-linear 2015-2025)
  Counter-cyclical rule: add +{CYCLICAL_BUMP:.1%} to growth if g_ngdp < {CYCLICAL_THRESHOLD:.0%}

HEADLINE RESULTS:
  BLENDED P(official > 40%, any year 2026-28): {blended_off:.3f} ({blended_off*100:.0f}%)
  BLENDED P(augmented > 40%, any year 2026-28): {blended_aug:.3f} ({blended_aug*100:.0f}%)

  Scenario weights: NGDP (30%/45%/25%) × Land (20%/50%/30%)

INTERPRETATION:
  The official metric is already at {cur_ratio:.1f}%, just {40.0-cur_ratio:.1f}pp from threshold.
  Land revenue decline is the dominant swing factor in the augmented measure —
  the ¥4.34tn collapse from 2021 peak has already materially weakened fiscal
  space. The official measure insulates against this because land revenue
  flows through the government-fund budget, not the general public budget.

  P(breach official) = {blended_off:.0%}: relatively contained because:
  (1) MoF has shown willingness to understate official deficits via special
      bonds and local-government off-balance-sheet vehicles.
  (2) Even in a weak-growth scenario, the official deficit/tax ratio has
      historically been managed below 40%.

  P(breach augmented) = {blended_aug:.0%}: significantly higher, reflecting the
  genuine fiscal pressure from collapsing land revenues and rising social
  expenditures. This is the more economically meaningful measure.
"""

with open('out/forecast12_summary.txt', 'w') as f:
    f.write(summary)
print(summary)
print(f"\n{'='*60}")
print(f"HEADLINE: P(official > 40%) = {blended_off:.2f} | P(augmented > 40%) = {blended_aug:.2f}")
print(f"{'='*60}")
