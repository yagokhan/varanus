#!/usr/bin/env python3
"""Quick V4 vs V5 fold comparison plot."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.style.use('dark_background')
TEAL='#00ffcc'; RED='#ff4444'; GREEN='#44ff88'; GREY='#888888'
CYAN='#00e5ff'; ORANGE='#ffaa00'; GOLD='#ffd740'; PURPLE='#b388ff'

v5_pf = [1.34, 1.08, 3.67, 1.05, 1.38, 2.07, 1.20, 1.31]
v5_wr = [35.3, 28.4, 45.2, 30.6, 29.0, 35.8, 32.0, 30.7]
v5_dd = [-26.5, -42.7, -14.8, -20.9, -19.0, -20.7, -19.4, -31.7]
v5_tr = [133, 155, 84, 160, 155, 95, 97, 192]
v5_agg = {"pf": 1.54, "wr": 32.4, "dd": -31.4, "sharpe": 5.68}
v4_agg = {"wr": 40.2, "dd": -15.1, "sharpe": 14.19}

fig = plt.figure(figsize=(24, 16))
fig.patch.set_facecolor('#0d0d0d')
fig.suptitle('Varanus V5 HPO — Per-Fold Analysis + V4 Comparison',
             color='white', fontsize=16, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
folds = np.arange(1, 9)

# 1. PF
ax1 = fig.add_subplot(gs[0, 0])
colors_pf = [GREEN if pf >= 1.50 else TEAL if pf >= 1.0 else RED for pf in v5_pf]
ax1.bar(folds, v5_pf, color=colors_pf, edgecolor='#222', width=0.6, alpha=0.85)
ax1.axhline(1.50, color=GOLD, ls='--', lw=1, alpha=0.7, label='Gate: PF >= 1.50')
ax1.axhline(1.0, color=GREY, ls=':', lw=0.8, alpha=0.5)
for i, v in enumerate(v5_pf):
    ax1.text(i+1, v+0.05, f'{v:.2f}', ha='center', color='white', fontsize=8, fontweight='bold')
ax1.set_title('Profit Factor per Fold', color='white', fontsize=13)
ax1.set_xlabel('Fold', color=GREY); ax1.set_xticks(folds)
ax1.legend(facecolor='#1a1a1a', edgecolor=GREY, labelcolor='white', fontsize=9)
ax1.tick_params(colors=GREY); ax1.set_facecolor('#111')
for sp in ax1.spines.values(): sp.set_color('#333')

# 2. WR
ax2 = fig.add_subplot(gs[0, 1])
colors_wr = [GREEN if wr >= 43 else TEAL if wr >= 30 else ORANGE for wr in v5_wr]
ax2.bar(folds, v5_wr, color=colors_wr, edgecolor='#222', width=0.6, alpha=0.85)
ax2.axhline(43, color=GOLD, ls='--', lw=1, alpha=0.7, label='Gate: WR >= 43%')
ax2.axhline(v4_agg['wr'], color=RED, ls=':', lw=1, alpha=0.7, label=f'V4: {v4_agg["wr"]}%')
for i, v in enumerate(v5_wr):
    ax2.text(i+1, v+0.5, f'{v:.1f}', ha='center', color='white', fontsize=8, fontweight='bold')
ax2.set_title('Win Rate (%) per Fold', color='white', fontsize=13)
ax2.set_xlabel('Fold', color=GREY); ax2.set_xticks(folds)
ax2.legend(facecolor='#1a1a1a', edgecolor=GREY, labelcolor='white', fontsize=9)
ax2.tick_params(colors=GREY); ax2.set_facecolor('#111')
for sp in ax2.spines.values(): sp.set_color('#333')

# 3. DD
ax3 = fig.add_subplot(gs[0, 2])
colors_dd = [GREEN if dd >= -25 else ORANGE if dd >= -35 else RED for dd in v5_dd]
ax3.bar(folds, v5_dd, color=colors_dd, edgecolor='#222', width=0.6, alpha=0.85)
ax3.axhline(-25, color=GOLD, ls='--', lw=1, alpha=0.7, label='Gate: DD >= -25%')
ax3.axhline(v4_agg['dd'], color=PURPLE, ls=':', lw=1, alpha=0.7, label=f'V4: {v4_agg["dd"]}%')
for i, v in enumerate(v5_dd):
    ax3.text(i+1, v-1.5, f'{v:.1f}', ha='center', color='white', fontsize=8, fontweight='bold')
ax3.set_title('Max Drawdown (%) per Fold', color='white', fontsize=13)
ax3.set_xlabel('Fold', color=GREY); ax3.set_xticks(folds)
ax3.legend(facecolor='#1a1a1a', edgecolor=GREY, labelcolor='white', fontsize=9)
ax3.tick_params(colors=GREY); ax3.set_facecolor('#111')
for sp in ax3.spines.values(): sp.set_color('#333')

# 4. Trades
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(folds, v5_tr, color=CYAN, edgecolor='#222', width=0.6, alpha=0.85)
for i, v in enumerate(v5_tr):
    ax4.text(i+1, v+2, str(v), ha='center', color='white', fontsize=8, fontweight='bold')
ax4.set_title('Trades per Fold', color='white', fontsize=13)
ax4.set_xlabel('Fold', color=GREY); ax4.set_xticks(folds)
ax4.tick_params(colors=GREY); ax4.set_facecolor('#111')
for sp in ax4.spines.values(): sp.set_color('#333')

# 5. V4 vs V5 aggregate
ax5 = fig.add_subplot(gs[1, 1:])
labels = ['PF', 'Win Rate', 'Sharpe', '|MaxDD|']
v4_vals = [0, v4_agg['wr'], v4_agg['sharpe'], abs(v4_agg['dd'])]
v5_vals = [v5_agg['pf'], v5_agg['wr'], v5_agg['sharpe'], abs(v5_agg['dd'])]
x = np.arange(len(labels))
w = 0.35
ax5.bar(x - w/2, v4_vals, w, label='V4 (in-sample)', color=TEAL, alpha=0.85, edgecolor='#222')
ax5.bar(x + w/2, v5_vals, w, label='V5 HPO (OOS)', color=GOLD, alpha=0.85, edgecolor='#222')
vl = [('n/a', f'{v5_agg["pf"]:.2f}'), (f'{v4_agg["wr"]:.1f}%', f'{v5_agg["wr"]:.1f}%'),
      (f'{v4_agg["sharpe"]:.1f}', f'{v5_agg["sharpe"]:.1f}'), (f'{v4_agg["dd"]:.1f}%', f'{v5_agg["dd"]:.1f}%')]
for i, (a, b) in enumerate(vl):
    ax5.text(i - w/2, v4_vals[i] + 0.5, a, ha='center', color=TEAL, fontsize=10, fontweight='bold')
    ax5.text(i + w/2, v5_vals[i] + 0.5, b, ha='center', color=GOLD, fontsize=10, fontweight='bold')
ax5.set_xticks(x); ax5.set_xticklabels(labels)
ax5.set_title('V4 vs V5 HPO — Aggregate Metrics', color='white', fontsize=13)
ax5.legend(facecolor='#1a1a1a', edgecolor=GREY, labelcolor='white', fontsize=11)
ax5.tick_params(colors=GREY, labelsize=11); ax5.set_facecolor('#111')
for sp in ax5.spines.values(): sp.set_color('#333')
ax5.text(0.98, 0.02, 'V4 is in-sample (overfitted)\nV5 is true out-of-sample',
         transform=ax5.transAxes, ha='right', va='bottom', color=ORANGE, fontsize=10, style='italic',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor=ORANGE, alpha=0.8))

plt.tight_layout()
path = '/home/yagokhan/varanus/v5_final/v5_07_v4_vs_v5_fold_comparison.png'
fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close(fig)
print(f'Saved: {path}')
