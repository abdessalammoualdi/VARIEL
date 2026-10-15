from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .metrics import reliability_bins, risk_curve


def figures(p, y, window_results, output, title, alpha=0.05):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10, 'text.color': 'black',
                         'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300})
    navy, teal, red = '#17365D', '#087F8C', '#C6533C'
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.3), layout='constrained')
    windows = np.arange(1, len(window_results) + 1)
    axes[0, 0].plot(windows, [r['coverage'] for r in window_results], color=navy, marker='o', lw=1.8)
    axes[0, 0].axhline(1 - alpha, color=red, ls='--', lw=1.3, label=f'Target {1-alpha:.0%}')
    axes[0, 0].set(xlabel='Deployment window', ylabel='Empirical coverage', ylim=(0, 1.02))
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(windows, [r['set_size'] for r in window_results], color=teal, marker='s', lw=1.8)
    axes[0, 1].set(xlabel='Deployment window', ylabel='Mean prediction-set size', ylim=(0, p.shape[1] + 0.1))
    bins = reliability_bins(p, y)
    axes[1, 0].plot([0, 1], [0, 1], color='black', ls=':', lw=1.)
    axes[1, 0].plot([b['confidence'] for b in bins], [b['accuracy'] for b in bins], color=teal, marker='o', lw=1.8)
    axes[1, 0].set(xlabel='Mean confidence', ylabel='Observed accuracy', xlim=(0, 1), ylim=(0, 1))
    coverage, risk = risk_curve(p, y)
    axes[1, 1].plot(coverage, risk, color=navy, lw=1.8)
    axes[1, 1].set(xlabel='Accepted fraction by confidence', ylabel='Empirical error rate', xlim=(0, 1), ylim=(0, max(0.1, float(risk.max()) * 1.05)))
    for label, ax in zip(['(a)', '(b)', '(c)', '(d)'], axes.flat):
        ax.set_title(label, loc='left', fontweight='bold')
        ax.grid(alpha=0.15)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(output / f'static_reliability.{extension}', bbox_inches='tight')
    plt.close(fig)
