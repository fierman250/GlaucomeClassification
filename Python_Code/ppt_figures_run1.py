"""
Regenerate performance figures using Run1 results and copy Run1 images to ppt_assets.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shutil, os

RUN1 = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\results\custom_experiments_Run1"
OUT  = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\ppt_assets"
os.makedirs(OUT, exist_ok=True)

C_TEAL  = "#028090"
C_NAVY  = "#065A82"
C_MINT  = "#02C39A"
C_CORAL = "#F96167"
C_LGRAY = "#E2E8F0"
C_DARK  = "#1E293B"
C_GRAY  = "#64748B"

plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})

# ─────────────────────────────────────────────────────────────────────────────
# RUN 1 DATA
# ─────────────────────────────────────────────────────────────────────────────
results = {
    ('Original', 'AlexNet'):      (0.898958, 0.964362, 0.878843, 0.919164),
    ('Original', 'VGG-16'):       (0.917988, 0.972913, 0.896022, 0.940054),
    ('Original', 'DenseNet-121'): (0.915723, 0.972507, 0.924051, 0.907357),
    ('Original', 'ResNet-18'):    (0.913457, 0.971148, 0.904159, 0.922797),
    ('Original', 'MobileNet-V2'): (0.901676, 0.961298, 0.862568, 0.940963),
    ('HE',       'AlexNet'):      (0.886724, 0.952535, 0.870705, 0.902816),
    ('HE',       'VGG-16'):       (0.903036, 0.963320, 0.885172, 0.920981),
    ('HE',       'DenseNet-121'): (0.893068, 0.959838, 0.862568, 0.923706),
    ('HE',       'ResNet-18'):    (0.888083, 0.953172, 0.868897, 0.907357),
    ('HE',       'MobileNet-V2'): (0.891255, 0.955118, 0.854430, 0.928247),
    ('CLAHE',    'AlexNet'):      (0.889443, 0.955501, 0.867089, 0.911898),
    ('CLAHE',    'VGG-16'):       (0.900770, 0.963563, 0.858047, 0.943688),
    ('CLAHE',    'DenseNet-121'): (0.896239, 0.962833, 0.850814, 0.941871),
    ('CLAHE',    'ResNet-18'):    (0.898052, 0.959105, 0.870705, 0.925522),
    ('CLAHE',    'MobileNet-V2'): (0.896239, 0.958094, 0.856239, 0.936421),
}

models = ['AlexNet', 'VGG-16', 'DenseNet-121', 'ResNet-18', 'MobileNet-V2']
preps  = ['Original', 'HE', 'CLAHE']
prep_colors = [C_TEAL, C_NAVY, C_MINT]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Performance Comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
acc_data = {p: [results[(p, m)][0] for m in models] for p in preps}
auc_data = {p: [results[(p, m)][1] for m in models] for p in preps}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')
x = np.arange(len(models))
w = 0.25

for ax, data, metric in zip(axes, [acc_data, auc_data], ['Accuracy', 'AUC']):
    for i, (prep, color) in enumerate(zip(preps, prep_colors)):
        bars = ax.bar(x + (i-1)*w, data[prep], w, color=color, label=prep, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=7, color=C_DARK, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=15)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_ylim(0.80, 1.02)
    ax.set_title(f'Validation {metric} — All Models & Preprocessing', fontsize=12, fontweight='bold', pad=10)
    ax.yaxis.grid(True, color=C_LGRAY, linewidth=0.8, zorder=0)
    ax.set_facecolor('white')
    ax.legend(fontsize=10)

plt.tight_layout(pad=2)
plt.savefig(os.path.join(OUT, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Heatmap
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
fig.patch.set_facecolor('white')

for ax, (metric_name, idx) in zip(axes, [('Accuracy', 0), ('AUC', 1)]):
    matrix = np.array([[results[(p, m)][idx] for m in models] for p in preps])
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.84, vmax=0.98, aspect='auto')
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(preps)))
    ax.set_xticklabels(models, fontsize=10, rotation=20, ha='right')
    ax.set_yticklabels(preps, fontsize=11)
    ax.set_title(f'{metric_name} Heatmap', fontsize=12, fontweight='bold', pad=8)
    for i in range(len(preps)):
        for j in range(len(models)):
            val = matrix[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color='black', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

plt.tight_layout(pad=2)
plt.savefig(os.path.join(OUT, 'heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Sensitivity vs Specificity — best models (VGG-16 per preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
# Best per preprocessing: Original=VGG-16(91.8%), HE=VGG-16(90.3%), CLAHE=VGG-16(90.1%)
best = {
    'Original\nVGG-16':  (0.896022, 0.940054),
    'HE\nVGG-16':        (0.885172, 0.920981),
    'CLAHE\nVGG-16':     (0.858047, 0.943688),
}

labels_b = list(best.keys())
sens = [best[k][0] for k in labels_b]
spec = [best[k][1] for k in labels_b]
x = np.arange(len(labels_b))
w2 = 0.35

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor('white')
b1 = ax.bar(x - w2/2, sens, w2, color=C_TEAL,  label='Sensitivity', zorder=3)
b2 = ax.bar(x + w2/2, spec, w2, color=C_CORAL, label='Specificity', zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(labels_b, fontsize=12)
ax.set_ylim(0.80, 1.0)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Sensitivity vs Specificity — Best Models per Preprocessing', fontsize=12, fontweight='bold', pad=10)
ax.yaxis.grid(True, color=C_LGRAY, linewidth=0.8, zorder=0)
ax.set_facecolor('white')
ax.legend(fontsize=11)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f'{h:.3f}',
            ha='center', va='bottom', fontsize=10, color=C_DARK)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'sens_spec.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved sens_spec.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Copy Run1 images for VGG-16 (best model per preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
copy_map = {
    'original_vgg16/roc_curve.png':       'roc_orig_vgg16.png',
    'original_vgg16/confusion_matrix.png':'cm_orig_vgg16.png',
    'original_vgg16/learning_curves.png': 'lc_orig_vgg16.png',
    'he_vgg16/roc_curve.png':             'roc_he_vgg16.png',
    'he_vgg16/confusion_matrix.png':      'cm_he_vgg16.png',
    'clahe_vgg16/roc_curve.png':          'roc_clahe_vgg16.png',
    'clahe_vgg16/confusion_matrix.png':   'cm_clahe_vgg16.png',
    'model_performance_comparison.png':   'perf_comparison.png',
    'performance_heatmaps.png':           'perf_heatmap_existing.png',
    'sensitivity_specificity_tradeoff.png':'sens_spec_existing.png',
}
for src_rel, dst in copy_map.items():
    src = os.path.join(RUN1, src_rel.replace('/', os.sep))
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, dst))
        print(f"Copied {dst}")
    else:
        print(f"WARNING: not found: {src}")

print("\nDone. Assets updated with Run1 results.")
