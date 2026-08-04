#!/usr/bin/env python3
"""
Forecast 10 — China Indigenous Advanced-Node Production by end-2028
JUDGMENT FORECAST with explicit uncertainty framing.
"""
import os, sys, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from datetime import datetime

os.makedirs('out', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ── dark theme ─────────────────────────────────────────────────────────────
DARK_BG  = '#0f1117'
PANEL_BG = '#1a1d2e'
TEXT     = '#e0e0e0'
GRID     = '#2a2d3e'
ACCENT   = '#4fc3f7'
GOLD     = '#ffd54f'
RED      = '#ef5350'
GREEN    = '#66bb6a'
ORANGE   = '#ffa726'
PURPLE   = '#ab47bc'
MUTED    = '#78909c'

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
# 1. DATA — China advanced-node trajectory (hardcoded, labelled by source)
# ═══════════════════════════════════════════════════════════════════════════

RECORDS = [
    # (year_num, year_label, milestone, node_nm, source, status, notes)
    (2019.0, "2019",
     "SMIC 14nm (N) mass production", "14",
     "SMIC annual report 2019",
     "CONFIRMED",
     "First volume below 14nm; DUV-only lithography; roughly TSMC 14nm equivalent"),

    (2020.5, "2020",
     "SMIC N+1 (~7nm) process development", "7",
     "SMIC investor disclosure Oct 2020",
     "CONFIRMED-prototype",
     "Process defined; yield unviable for commercial volume; no product shipped on N+1"),

    (2022.5, "2022",
     "SMIC N+2 (~7nm-class) first limited production", "7",
     "Inferred from TechInsights teardown (Sep 2023); Bloomberg Sep 2023",
     "CONFIRMED-limited",
     "DUV multi-patterning; transistor density ~7nm-class; yield and cost materially worse than TSMC 7nm (EUV); SMIC has never publicly named this node"),

    (2023.6, "2023",
     "Kirin 9000s (Huawei Mate 60 Pro) confirmed on N+2", "7",
     "TechInsights physical teardown Sep 2023; Bloomberg Sep 2023",
     "CONFIRMED",
     "Only confirmed indigenous 7nm-class product; Mate 60 Pro launched without public announcement; volumes reportedly constrained by SMIC yield"),

    (2023.9, "2023",
     "Huawei Ascend 910B AI chip on SMIC N+2", "7",
     "SemiAnalysis 2024; Reuters 2024; analyst inference",
     "CONFIRMED-limited",
     "AI accelerator on N+2 process; shipping volumes far below stated Huawei targets; yield-constrained supply chain"),

    (2024.3, "2024",
     "CXMT HBM2E pilot production", "HBM2E",
     "TrendForce Q1 2024; Yole Developpement 2024",
     "CONFIRMED-early",
     "Approx 3+ years behind HBM3 (the AI-critical spec); pilot not volume; not a viable substitute for SK Hynix/Samsung HBM3 in AI workloads"),

    (2024.7, "2024",
     "SMIC N+3 (~5nm-class) R&D program (contested)", "5",
     "Anonymous industry sources; analyst inference 2024 (no primary source)",
     "DISPUTED — unconfirmed",
     "No public teardown; no SMIC disclosure; existence of N+3 program is analyst inference only; may be optimized N+2 with minor improvements"),

    (2024.9, "2024",
     "Huawei Ascend 910C — node classification unclear", "5?",
     "SemiAnalysis; Reuters 2024; Huawei not disclosing",
     "DISPUTED — unconfirmed",
     "Analysts divided: may be SMIC N+2 optimized, or N+3 if N+3 exists; no physical teardown as of 2024-25; Huawei silent on node"),

    (2024.95, "2024",
     "SMEE domestic EUV prototype (claimed)", "N/A",
     "SMEE company statements 2024; SCMP reporting 2024",
     "PARTIAL — contested",
     "Enormous gap between prototype and production-grade EUV: mirror polish, laser source power, photoresist ecosystem, optics precision — all years behind ASML; NOT a near-term ASML substitute"),

    (2025.5, "2025-26",
     "Next-gen Huawei chip on uncertain node", "5?",
     "Forward projection from Mate 60 Pro / Kirin product cadence",
     "PROJECTED — uncertain",
     "If SMIC N+3 exists and yields improve; may still be relabeled optimized-7nm; no teardown data to anchor projection"),

    (2028.0, "2028",
     "HORIZON: volume indigenous 5nm-class?", "5",
     "THIS FORECAST — see model below",
     "TARGET",
     "P(BACKFIRE) = 10-22% indigenous-only; 16-35% incl. foreign fabs. USER SETS FINAL ESTIMATE."),
]

COLS = ['year_num', 'year_label', 'milestone', 'node_nm', 'source', 'status', 'notes']
df = pd.DataFrame(RECORDS, columns=COLS)
df.to_csv('data/china_advanced_node.csv', index=False)

print("=" * 74)
print("FORECAST 10 — CHINA INDIGENOUS ADVANCED-NODE PRODUCTION BY 2028")
print("=" * 74)
print()
print("!!! WARNING: JUDGMENT FORECAST — NOT A DATA-DRIVEN MODEL !!!")
print("    Evidence base is murky. Error bars are genuinely wide.")
print("    User sets the final point estimate.")
print()
print("China Advanced-Node Milestone Table (hardcoded; sourced):")
print("-" * 74)
for _, r in df.iterrows():
    disp = "!" if any(k in r.status for k in ("DISPUTED", "PROJECTED", "PARTIAL")) else " "
    print(f" {disp} {r.year_label:<7} [{r.status:<26}] ~{r.node_nm:<6} {r.milestone[:44]}")
    print(f"         Source: {r.source[:68]}")
    print(f"         Note:   {r.notes[:70]}")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# 2. DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 74)
print("DEFINITIONS (explicit — forecast resolution depends on these choices)")
print("=" * 74)
print("""\
NODE THRESHOLD
  "5nm-class" = logic density comparable to TSMC N5 / Samsung 5LPE
  (~170 MTr/mm2 transistor density) AND commercial yield sufficient for
  multi-product line shipment. Specifically EXCLUDES relabeled/optimized
  7nm chips that meet density spec only by SMIC's own label convention.
  "5nm-class" is CONTESTABLE — if SMIC self-labels N+3 as 5nm but
  density/yield falls short of TSMC N5, it does NOT meet this bar.
  Resolution depends entirely on teardown data becoming available.

VOLUME PRODUCTION
  Operationally: >= 100k wafer-starts/month sustained for >= 3
  consecutive months, or equivalent supply sufficient to power multiple
  Huawei product lines simultaneously. EXCLUDES: lab demo, single chip
  teardown without repeat volume, limited engineering samples.
  NOTE: SMIC N+2 as of 2023-24 likely does NOT yet meet this bar.

INDIGENOUS-ONLY (primary metric)
  SMIC/Huawei-ecosystem domestic capability only. EXCLUDES foreign-owned
  fabs on Chinese soil (TSMC Nanjing = 16nm only; Samsung Xi'an = legacy
  DRAM). These are the restricted entities; they are NOT the story here.

INCLUDE-FOREIGN VARIANT (looser upper bound)
  Adds: could a foreign fab on Chinese soil achieve 5nm-class under
  Chinese-directed production? Different policy question; provides
  upper bound on China's total advanced-node supply access.

EVIDENCE BASIS
  TechInsights physical teardown reports (primary); SMIC annual reports
  (sparse, no node names disclosed); Bloomberg/Reuters industry reporting;
  SemiAnalysis node-density analysis; Yole/TrendForce supply-chain data.
  SMIC NEVER discloses node names or yield statistics publicly.
  All "7nm" / "5nm" attributions are analyst inferences from transistor
  density measurements — introducing irreducible definitional uncertainty.
""")

# ═══════════════════════════════════════════════════════════════════════════
# 3. KEY UNCERTAINTIES / ERROR BARS — PROMINENT
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 74)
print("*** KEY UNCERTAINTIES / ERROR BARS — READ BEFORE CITING ***")
print("=" * 74)
print("""\
  1. DUV YIELD WALL — can DUV multi-patterning reach economic 5nm yield?
     Each extra DUV patterning step adds ~10-20% compounding yield loss.
     Analyst consensus: "economically unviable" at 5nm-class via DUV only.
     But: SMIC 7nm was also widely called "impossible" before TechInsights
     confirmed it in the Mate 60 teardown. Consensus can be wrong.
     Error contribution to P(BACKFIRE): +/- 8-10 percentage points.

  2. EUV ACCESS — smuggling or domestic SMEE EUV?
     SMEE EUV prototype reportedly tested 2024 but gap to production-grade
     EUV is enormous (mirror polish, laser source power, resists, optics).
     ASML EUV = 457 distinct components from controlled Western suppliers.
     Smuggling historically precedented but each component is monitored.
     Error contribution: +/- 5-8 pp.

  3. DEFINITION RISK — single biggest definitional threat to this forecast.
     If bar = "SMIC self-labeled N+3": P(BACKFIRE) rises ~8-15pp.
     If bar = "TSMC N5 density-equivalent": P(BACKFIRE) falls ~5-10pp.
     This forecast uses the stricter TSMC-equivalent density bar.
     Error contribution: +/- 6-12 pp depending on definition chosen.

  4. SUBSIDY OVERRIDE — $143bn+ Big Fund 3 + provincial programs.
     Economic infeasibility arguments may not apply to state priorities.
     China can sustain uneconomic processes for strategic decades.
     But money does not buy physics — yield losses are not just cost.
     Error contribution: +/- 5-7 pp.

  5. INFORMATION GAP — we have ONE confirmed teardown (Mate 60 Pro, Sep
     2023). Unknown: SMIC actual wafer-starts at N+2 and any N+3; N+3
     existence; current yield rates; Ascend 910C actual node.
     Any new teardown data substantially updates P.
     This forecast should be revisited after any new teardown confirmation.

  AGGREGATE ERROR BAR: +/- 8-12 pp on P(BACKFIRE) indigenous-only.
  That is NOT model noise — it reflects genuine epistemic uncertainty.
  The range IS the finding. A point estimate without the range is false precision.
""")

# ═══════════════════════════════════════════════════════════════════════════
# 4. SCENARIO MODEL
# ═══════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    'CONTAINED': {
        'label': 'CONTAINED\n(stalls <= 7nm)',
        'color': GREEN,
        # indigenous-only
        'ind_pt': 0.47, 'ind_lo': 0.38, 'ind_hi': 0.56,
        # include-foreign
        'for_pt': 0.38, 'for_lo': 0.30, 'for_hi': 0.46,
        'assumption': 'DUV multi-patterning cannot reach economic volume 5nm yield by 2028 — EUV restriction is binding',
        'confidence':  'MODERATE',
        'for_ev':   'No EUV; compounding yield penalty per DUV step makes 5nm cost-prohibitive even with subsidies; SMIC N+2 volumes already constrained; no confirmed N+3 program as of 2024',
        'against':  'China has surprised analysts before; $143bn+ subsidies absorb yield losses; domestic EUV progressing; national security framing overrides economics',
    },
    'PARTIAL': {
        'label': 'PARTIAL\n(limited 5nm demo)',
        'color': ORANGE,
        'ind_pt': 0.38, 'ind_lo': 0.28, 'ind_hi': 0.46,
        'for_pt': 0.37, 'for_lo': 0.27, 'for_hi': 0.45,
        'assumption': 'Technical 5nm-class milestone achieved but yield does not reach "volume production" bar before 2028 deadline',
        'confidence':  'MODERATE',
        'for_ev':   'Consistent with China semiconductor pattern: technical milestones arrive ~2-3 years after claims, but volume lags further; SMIC N+2 achieved technically before volume',
        'against':  'Partial outcome still FAILS the event condition; 2028 deadline is tight; DUV 5nm yield may be fundamentally non-viable, not just delayed',
    },
    'BACKFIRE': {
        'label': 'BACKFIRE\n(volume 5nm-class)',
        'color': RED,
        'ind_pt': 0.15, 'ind_lo': 0.10, 'ind_hi': 0.22,
        'for_pt': 0.25, 'for_lo': 0.16, 'for_hi': 0.35,
        'assumption': 'DUV workarounds + subsidies achieve volume 5nm-class, OR domestic/smuggled EUV matures faster than consensus expects',
        'confidence':  'LOW-MODERATE — contrarian/tail scenario; possible but requires multiple developments aligning',
        'for_ev':   '2022-23 SMIC 7nm surprise (achieved vs analyst consensus of impossible); unprecedented subsidy scale; SMEE domestic EUV reported progress; Huawei national security imperative',
        'against':  'EUV gap is much larger than 7nm-to-5nm gap implies; no confirmed N+3 program; yield economics at 5nm via DUV are analyst consensus "fundamentally unviable"; time is short',
    },
}

# Sanity check: point estimates should sum to 1.0 for each variant
ind_sum = sum(s['ind_pt'] for s in SCENARIOS.values())
for_sum = sum(s['for_pt'] for s in SCENARIOS.values())
assert abs(ind_sum - 1.0) < 0.01, f"Indigenous sum {ind_sum:.3f} != 1.0"
assert abs(for_sum - 1.0) < 0.01, f"Foreign sum {for_sum:.3f} != 1.0"
print(f"[Check] Scenario sums: indigenous={ind_sum:.2f}, foreign={for_sum:.2f}  (both == 1.00)")
print()

print("=" * 74)
print("SCENARIO MODEL — THREE BRANCHES")
print("=" * 74)
for name, s in SCENARIOS.items():
    print(f"\n{'─'*62}")
    print(f"SCENARIO: {name}  [{s['confidence']}]")
    print(f"  P(indigenous-only):   {s['ind_lo']*100:.0f}%-{s['ind_hi']*100:.0f}%  (pt: {s['ind_pt']*100:.0f}%)")
    print(f"  P(include-foreign):   {s['for_lo']*100:.0f}%-{s['for_hi']*100:.0f}%  (pt: {s['for_pt']*100:.0f}%)")
    print(f"  KEY ASSUMPTION: {s['assumption']}")
    print(f"  FOR:    {s['for_ev'][:76]}")
    print(f"  AGAINST:{s['against'][:76]}")

bf = SCENARIOS['BACKFIRE']
print()
print("=" * 74)
print("HEADLINE — P(BACKFIRE): volume indigenous 5nm-class by end-2028")
print("=" * 74)
print(f"\n  INDIGENOUS-ONLY:    RANGE  {bf['ind_lo']*100:.0f}%-{bf['ind_hi']*100:.0f}%")
print(f"  SUGGESTED POINT:    {bf['ind_pt']*100:.0f}%   <-- USER SETS THE FINAL NUMBER")
print()
print(f"  INCLUDE-FOREIGN:    RANGE  {bf['for_lo']*100:.0f}%-{bf['for_hi']*100:.0f}%")
print(f"  SUGGESTED POINT:    {bf['for_pt']*100:.0f}%   (looser definition, upper bound)")
print()
print("  *** THE RANGE IS THE FINDING. Point estimate is a suggestion only. ***")
print("  *** This is a reasoning-driven forecast, not a statistical output.  ***")

# ═══════════════════════════════════════════════════════════════════════════
# 5. PAIRING WITH FORECAST 9
# ═══════════════════════════════════════════════════════════════════════════

print()
print("=" * 74)
print("PAIRING WITH FORECAST 9 — THE ESCALATION-LOOP LOGIC")
print("=" * 74)
print("""\
Forecast 9:  P(US keeps tightening advanced-node controls through 2027)
             = ~93%.  The US DOES continue restricting EUV, advanced DUV,
             chip design tools (EDA), and HBM to China. Near-certain.

Forecast 10: GIVEN that tightening, P(China achieves volume indigenous
             5nm-class by end-2028) = 10-22% [pt: 15%].
             This asks whether the controls CONTAIN or ACCELERATE.

THE ESCALATION-LOOP THESIS:
  P(F9=true AND F10=BACKFIRE) = ~93% x 15% = ~14%
  Interpretation: ~14% chance controls accelerate rather than contain
  domestic substitution. This is the escalation-loop tail risk.

MODAL OUTCOME (the base case):
  P(F9=true AND F10=CONTAINED) = ~93% x 47% = ~44%
  Controls tighten AND China stays stuck at 7nm through 2028.
  This is the most probable single scenario by a wide margin.

THE CONTRARIAN BET (the BACKFIRE scenario):
  Consensus = "tight controls -> China stalls."
  Contrarian = "tight controls -> national emergency -> mobilization ->
  accelerated breakthrough, as with Soviet nuclear / SMIC 7nm surprise."
  The Mate 60 Pro teardown (Sep 2023) IS the precedent. Analysts
  said SMIC 7nm was impossible. SMIC shipped it anyway.

CONTRAST AS THE FINDING:
  F9 high (93%) x F10 mid (15%) = escalation-loop risk is real but
  not the base case. The key MONITORING VARIABLE is new teardown data
  on SMIC N+3 or Ascend 910C node. If confirmed 5nm-class, P(BACKFIRE)
  updates upward substantially, and so does the joint probability.
""")

# ═══════════════════════════════════════════════════════════════════════════
# 6. CHART 1 — Scenario probabilities with error bars
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(PANEL_BG)

names  = list(SCENARIOS.keys())
x      = np.arange(len(names))
width  = 0.32

ind_pts = np.array([SCENARIOS[s]['ind_pt'] for s in names]) * 100
ind_lo  = np.array([SCENARIOS[s]['ind_pt'] - SCENARIOS[s]['ind_lo'] for s in names]) * 100
ind_hi  = np.array([SCENARIOS[s]['ind_hi'] - SCENARIOS[s]['ind_pt'] for s in names]) * 100

for_pts = np.array([SCENARIOS[s]['for_pt'] for s in names]) * 100
for_lo  = np.array([SCENARIOS[s]['for_pt'] - SCENARIOS[s]['for_lo'] for s in names]) * 100
for_hi  = np.array([SCENARIOS[s]['for_hi'] - SCENARIOS[s]['for_pt'] for s in names]) * 100

colors = [SCENARIOS[s]['color'] for s in names]

bars_ind = ax.bar(x - width/2, ind_pts, width, color=colors,
                  alpha=0.88, label='Indigenous-only (SMIC/Huawei)')
bars_for = ax.bar(x + width/2, for_pts, width, color=colors,
                  alpha=0.44, label='Include-foreign fabs', hatch='//')

ax.errorbar(x - width/2, ind_pts, yerr=[ind_lo, ind_hi],
            fmt='none', color=TEXT, capsize=8, linewidth=2, capthick=2)
ax.errorbar(x + width/2, for_pts, yerr=[for_lo, for_hi],
            fmt='none', color=TEXT, capsize=8, linewidth=2, capthick=2, linestyle='--')

# Value labels above error bar tops
for i, s_name in enumerate(names):
    s = SCENARIOS[s_name]
    top_ind = s['ind_hi'] * 100 + 3.5
    top_for = s['for_hi'] * 100 + 3.5
    ax.text(i - width/2, top_ind, f"{s['ind_pt']*100:.0f}%",
            ha='center', va='bottom', color=TEXT, fontsize=11, fontweight='bold')
    ax.text(i + width/2, top_for, f"{s['for_pt']*100:.0f}%",
            ha='center', va='bottom', color=TEXT, fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels([SCENARIOS[s]['label'] for s in names], fontsize=12)
ax.set_ylabel("Probability (%)", color=TEXT, fontsize=12)
ax.set_title(
    "Forecast 10 — China 5nm-Class Volume Production by end-2028\n"
    "Scenario Probabilities with Error Bars  |  JUDGMENT FORECAST — Wide uncertainty",
    color=TEXT, fontsize=12, pad=14)
ax.set_ylim(0, 72)
ax.grid(axis='y', alpha=0.3)

legend_elems = [
    mpatches.Patch(color=MUTED, alpha=0.88, label='Indigenous-only (SMIC/Huawei ecosystem)'),
    mpatches.Patch(color=MUTED, alpha=0.44, hatch='//', label='Include foreign fabs on Chinese soil'),
    Line2D([0],[0], color=TEXT, linewidth=2,
           label='Error bar = judgment range (not model noise)'),
]
ax.legend(handles=legend_elems, loc='upper right',
          facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)

note = ("Error bars reflect genuine expert disagreement, not statistical noise.\n"
        "P(BACKFIRE) indigenous: 10-22%  |  suggested point: 15%  "
        "|  USER SETS FINAL NUMBER")
ax.text(0.5, -0.12, note, transform=ax.transAxes,
        ha='center', va='top', color=MUTED, fontsize=9, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('out/forecast10_scenario_probabilities.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("\nSaved: out/forecast10_scenario_probabilities.png")

# ═══════════════════════════════════════════════════════════════════════════
# 7. CHART 2 — China node timeline
# ═══════════════════════════════════════════════════════════════════════════

# Data points for the timeline (cleaner grouping than full RECORDS table)
TL = [
    # (yr, node_nm_for_plot, short_label, status_key, label_above)
    # status_key maps to color and marker
    (2019.0, 14.0, "SMIC 14nm (N)\nMass Production",        "confirmed",  True),
    (2020.5,  7.0, "SMIC N+1\nprototype (2020)",            "proto",      False),
    (2022.5,  7.0, "SMIC N+2\nKirin 9000s [2022]",          "confirmed",  True),
    (2023.8,  7.0, "Ascend 910B\non N+2 [2023]",            "proto",      False),
    (2024.8,  5.0, "SMIC N+3 / Ascend 910C\n[DISPUTED — no teardown]", "disputed", False),
    (2025.7,  5.0, "Next-gen Huawei chip\n[PROJECTED]",     "projected",  True),
    (2028.0,  5.0, "HORIZON\n2028",                         "target",     False),
]

STATUS_META = {
    'confirmed': {'color': GREEN,  'marker': 'o', 'size': 130, 'lw': 1.0},
    'proto':     {'color': ACCENT, 'marker': 'D', 'size':  90, 'lw': 0.8},
    'disputed':  {'color': ORANGE, 'marker': 's', 'size': 120, 'lw': 1.0},
    'projected': {'color': PURPLE, 'marker': 's', 'size': 100, 'lw': 0.8},
    'target':    {'color': RED,    'marker': '^', 'size': 180, 'lw': 1.2},
}

fig, ax = plt.subplots(figsize=(14, 7.5))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(PANEL_BG)

# TSMC 5nm EUV reference band (commercially available ~2020)
ax.axhspan(4.7, 5.3, alpha=0.07, color=RED, label='5nm-class threshold band')
ax.text(2021.5, 5.0, 'TARGET: 5nm-class', color=RED, fontsize=9, alpha=0.7,
        ha='center', va='center', style='italic')

# TSMC 7nm dashed reference
ax.axhline(7.0, color=MUTED, linewidth=1.0, linestyle=':', alpha=0.4)
ax.text(2019.1, 6.65, 'TSMC 7nm EUV (commercial 2019)', color=MUTED,
        fontsize=8, alpha=0.65)

# 2028 horizon
ax.axvline(2028.0, color=RED, alpha=0.65, linewidth=2.5, linestyle='--')

# "DUV wall" annotation
ax.annotate('DUV multi-patterning\nyield wall (consensus)',
            xy=(2026, 5.3), xytext=(2024.5, 3.9),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2),
            color=ORANGE, fontsize=8.5, alpha=0.85)

# Plot each milestone
for (yr, nm, lbl, sk, above) in TL:
    meta = STATUS_META[sk]
    ax.scatter(yr, nm, color=meta['color'], s=meta['size'], zorder=5,
               marker=meta['marker'], edgecolors='white',
               linewidths=meta['lw'], alpha=0.92)
    # Inverted axis: smaller nm = top; label above => smaller y offset (nm - offset)
    if above:
        ax.text(yr, nm - 1.6, lbl, ha='center', va='bottom',
                color=meta['color'], fontsize=8.2, linespacing=1.4)
    else:
        ax.text(yr, nm + 1.4, lbl, ha='center', va='top',
                color=meta['color'], fontsize=8.2, linespacing=1.4)

# Trajectory line through confirmed points
confirmed_pts = [(yr, nm) for (yr, nm, _, sk, _) in TL if sk in ('confirmed', 'proto')]
cx = [p[0] for p in confirmed_pts]
cy = [p[1] for p in confirmed_pts]
ax.plot(cx, cy, color=GREEN, alpha=0.35, linewidth=1.5,
        linestyle='--', zorder=2, label='Confirmed/limited trajectory')

# Axis formatting (inverted: smaller nm = top = more advanced)
ax.set_ylim(3.0, 16.5)
ax.invert_yaxis()
ax.set_yticks([5, 7, 10, 14])
ax.set_yticklabels(['5nm (target)', '7nm', '10nm', '14nm'], color=TEXT, fontsize=10)
ax.set_xlim(2018.5, 2029.2)
ax.set_xticks(range(2019, 2029))
ax.set_xlabel("Year", color=TEXT, fontsize=11)
ax.set_ylabel("Node Tier (nm, approximate)  —  smaller = more advanced  ->", color=TEXT, fontsize=10)
ax.set_title(
    "China Advanced-Node Progression — 2019 to 2028 Horizon\n"
    "Confirmed vs. Disputed vs. Projected  |  EUV-free DUV multi-patterning trajectory",
    color=TEXT, fontsize=12, pad=12)

legend_elems = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=GREEN,  markersize=10, label='CONFIRMED production'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor=ACCENT, markersize=9,  label='Confirmed-prototype (limited)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=ORANGE, markersize=9,  label='DISPUTED — no teardown confirmation'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=PURPLE, markersize=9,  label='PROJECTED (forward inference)'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor=RED,    markersize=11, label='2028 target horizon'),
    mpatches.Patch(color=RED, alpha=0.15,                                         label='5nm-class threshold band'),
]
ax.legend(handles=legend_elems, loc='lower right',
          facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
ax.grid(True, alpha=0.25)

note2 = ("Node labels are analyst attributions from transistor density measurements, NOT SMIC disclosures."
         "  |  DISPUTED entries lack teardown confirmation.")
ax.text(0.5, -0.09, note2, transform=ax.transAxes,
        ha='center', va='top', color=ORANGE, fontsize=8.5, style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('out/forecast10_node_timeline.png', dpi=150,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Saved: out/forecast10_node_timeline.png")

# ═══════════════════════════════════════════════════════════════════════════
# 8. SUMMARY TEXT
# ═══════════════════════════════════════════════════════════════════════════

lines = []
def w(s=""): lines.append(s)

w("=" * 74)
w("FORECAST 10 — CHINA INDIGENOUS ADVANCED-NODE PRODUCTION BY end-2028")
w(f"Generated: {datetime.now().strftime('%Y-%m-%d')}")
w("=" * 74)
w()
w("!!! WARNING: JUDGMENT FORECAST — NOT A DATA-DRIVEN MODEL !!!")
w("Evidence is genuinely murky. Error bars are wide. User sets final estimate.")
w()
w("─" * 74)
w("DEFINITIONS")
w("─" * 74)
w("NODE THRESHOLD:  5nm-class = transistor density >= ~170 MTr/mm2 AND")
w("  commercial yield sufficient for multi-product volume.  CONTESTABLE.")
w("  Strict bar used here: TSMC N5 density-equivalent, NOT SMIC self-label.")
w()
w("VOLUME PRODUCTION: >= 100k wafer-starts/month for >= 3 months. EXCLUDES")
w("  lab/demo, single teardown, limited engineering samples.")
w("  NOTE: SMIC N+2 (~7nm) likely does NOT yet meet this bar as of 2024.")
w()
w("INDIGENOUS-ONLY (primary): SMIC/Huawei-ecosystem domestic fabs only.")
w("  Excludes foreign-owned fabs (TSMC Nanjing, Samsung Xi'an).")
w()
w("INCLUDE-FOREIGN VARIANT: upper-bound sensitivity including foreign fabs")
w("  on Chinese soil — a different policy question.")
w()
w("EVIDENCE BASIS: TechInsights teardowns (primary physical evidence),")
w("  SMIC annual reports (no node/yield disclosure), Bloomberg/Reuters,")
w("  SemiAnalysis, Yole/TrendForce. SMIC never discloses node names.")
w()
w("─" * 74)
w("*** KEY UNCERTAINTIES / ERROR BARS — READ BEFORE CITING ***")
w("─" * 74)
w("1. DUV YIELD WALL         +/- 8-10pp  Each DUV patterning step: 10-20%")
w("   compounding yield loss. Consensus: 5nm via DUV = economically unviable.")
w("   But SMIC 7nm was also 'impossible' before confirmed in Sep 2023.")
w()
w("2. EUV ACCESS             +/- 5-8pp   SMEE EUV prototype reported 2024.")
w("   Enormous gap to production-grade (mirrors, laser, resists, optics).")
w("   Smuggling: 457-component machine from controlled Western suppliers.")
w()
w("3. DEFINITION RISK        +/- 6-12pp  BIGGEST SINGLE RISK. If bar =")
w("   'SMIC self-labeled N+3,' P rises. If bar = 'TSMC N5 equivalent,' P falls.")
w("   Resolves only with teardown data.")
w()
w("4. SUBSIDY OVERRIDE       +/- 5-7pp   $143bn+ Big Fund 3 + provincial.")
w("   State can sustain uneconomic production. But money does not buy physics.")
w()
w("5. INFORMATION GAP        substantial. ONE confirmed teardown (Sep 2023).")
w("   Unknown: SMIC N+2 volumes, N+3 existence, yield rates, 910C node.")
w("   Revisit this forecast after any new teardown.")
w()
w("AGGREGATE ERROR BAR: +/- 8-12pp on P(BACKFIRE) indigenous-only.")
w("THE RANGE IS THE FINDING. Do not cite the point estimate without the range.")
w()
w("─" * 74)
w("SCENARIO MODEL")
w("─" * 74)
for name, s in SCENARIOS.items():
    w(f"\n{name}  [{s['confidence']}]")
    w(f"  P(indigenous-only):  {s['ind_lo']*100:.0f}%-{s['ind_hi']*100:.0f}%  pt={s['ind_pt']*100:.0f}%")
    w(f"  P(include-foreign):  {s['for_lo']*100:.0f}%-{s['for_hi']*100:.0f}%  pt={s['for_pt']*100:.0f}%")
    w(f"  KEY ASSUMPTION: {s['assumption']}")
    w(f"  FOR:     {s['for_ev']}")
    w(f"  AGAINST: {s['against']}")

w()
w("─" * 74)
w("HEADLINE: P(BACKFIRE) — volume indigenous 5nm-class by end-2028")
w("─" * 74)
bf = SCENARIOS['BACKFIRE']
w(f"  INDIGENOUS-ONLY RANGE:    {bf['ind_lo']*100:.0f}%-{bf['ind_hi']*100:.0f}%")
w(f"  SUGGESTED POINT ESTIMATE: {bf['ind_pt']*100:.0f}%   <-- USER SETS THE FINAL NUMBER")
w(f"  INCLUDE-FOREIGN RANGE:    {bf['for_lo']*100:.0f}%-{bf['for_hi']*100:.0f}%")
w(f"  SUGGESTED POINT:          {bf['for_pt']*100:.0f}%  (looser bar, upper bound)")
w()
w("  THIS IS A REASONING-DRIVEN FORECAST, NOT A STATISTICAL MODEL.")
w("  The model surfaces the key tensions; the analyst sets the final weight.")
w()
w("─" * 74)
w("PAIRING WITH FORECAST 9 — ESCALATION-LOOP LOGIC")
w("─" * 74)
w("F9: P(US tightening through 2027) = ~93%.  Near-certain baseline.")
w("F10: P(China achieves volume 5nm-class given tightening) = 10-22%.")
w()
w("JOINT (escalation loop fires): ~93% x 15% = ~14%")
w("  Controls accelerate, not contain, domestic substitution.")
w()
w("MODAL (base case): ~93% x 47% = ~44%")
w("  Controls tighten AND China stays stuck. Most probable single scenario.")
w()
w("CONTRARIAN BET: tight controls -> national emergency -> mobilization ->")
w("  breakthrough. Historical analog: SMIC 7nm vs analyst consensus (2023).")
w()
w("MONITORING VARIABLE: Ascend 910C or SMIC N+3 teardown data.")
w("  Any confirmation of 5nm-class updates P(BACKFIRE) substantially,")
w("  and updates the escalation-loop joint probability proportionally.")
w()
w("=" * 74)
w("Files: forecast10_scenario_probabilities.png, forecast10_node_timeline.png,")
w("       forecast10_summary.txt")

with open('out/forecast10_summary.txt', 'w') as f:
    f.write('\n'.join(lines))
print("Saved: out/forecast10_summary.txt")

print()
print("─" * 74)
print(f"P(BACKFIRE) RANGE : {bf['ind_lo']*100:.0f}%-{bf['ind_hi']*100:.0f}%  (indigenous-only)")
print(f"SUGGESTED POINT   : {bf['ind_pt']*100:.0f}%   <-- USER SETS THE FINAL NUMBER")
print(f"Include-foreign   : {bf['for_lo']*100:.0f}%-{bf['for_hi']*100:.0f}%  (upper bound)")
print("─" * 74)
