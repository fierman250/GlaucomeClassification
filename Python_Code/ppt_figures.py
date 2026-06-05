"""
Generate all figures needed for the presentation.
Run this script before building the PPT.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
from PIL import Image
import os

OUT = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\ppt_assets"
os.makedirs(OUT, exist_ok=True)

TEAL   = "#028090"
NAVY   = "#065A82"
MINT   = "#02C39A"
GRAY   = "#64748B"
LGRAY  = "#E2E8F0"
WHITE  = "#FFFFFF"
DARK   = "#1E293B"
CORAL  = "#F96167"
GOLD   = "#F9E795"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Distribution (Train vs Test, Class Balance)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('white')

# Left: class distribution bar
categories  = ['POSITIVE\n(Glaucoma)', 'NEGATIVE\n(Normal)']
train_counts = [5516, 5515]
test_counts  = [1000, 1000]   # approximate – test split from RefA1

x = np.arange(len(categories))
w = 0.35
b1 = axes[0].bar(x - w/2, train_counts, w, color=TEAL,  label='Train', zorder=3)
b2 = axes[0].bar(x + w/2, test_counts,  w, color=CORAL, label='Test',  zorder=3)
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories, fontsize=12)
axes[0].set_ylabel('Number of Images', fontsize=12)
axes[0].set_title('Class Distribution by Split', fontsize=14, fontweight='bold', pad=12)
axes[0].yaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
axes[0].set_facecolor('white')
axes[0].legend(fontsize=11)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, h + 60, f'{h:,}',
                 ha='center', va='bottom', fontsize=10, color=DARK)

# Right: pie – total dataset composition
labels2 = ['Train\n11,031', 'Test\n~2,000']
sizes2  = [11031, 2000]
colors2 = [TEAL, CORAL]
wedges, texts, autotexts = axes[1].pie(
    sizes2, labels=labels2, colors=colors2, autopct='%1.1f%%',
    startangle=90, wedgeprops={'edgecolor':'white','linewidth':2},
    textprops={'fontsize':12}
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_color('white')
    at.set_fontweight('bold')
axes[1].set_title('Total Dataset Split', fontsize=14, fontweight='bold', pad=12)

plt.tight_layout(pad=2)
plt.savefig(os.path.join(OUT, 'dataset_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved dataset_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Sub-dataset contribution bar chart
# ─────────────────────────────────────────────────────────────────────────────
datasets_info = {
    'BEH':    (325, 289),
    'CRFO':   (230, 180),
    'DRH':    (50,  31),
    'EyePACS':(1542,1711),
    'FIVES':  (200, 600),
    'G1020':  (231, 789),
    'HRF':    (15,  30),
    'JSIEC':  (80,  40),
    'LES':    (22,  18),
    'OIA':    (168, 482),
    'ORIGA':  (73,  482),
    'ORIGA2': (50,  600),
    'PAPILA': (99,  49),
    'REFUGE': (360, 360),
    'SJCHOI': (101, 900),
}

names = list(datasets_info.keys())
pos_vals = [v[0] for v in datasets_info.values()]
neg_vals = [v[1] for v in datasets_info.values()]

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(names))
w = 0.4
ax.bar(x - w/2, pos_vals, w, color=TEAL,  label='POSITIVE', zorder=3)
ax.bar(x + w/2, neg_vals, w, color=CORAL, label='NEGATIVE', zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('Number of Images', fontsize=12)
ax.set_title('Image Count per Source Dataset (Training Split)', fontsize=14, fontweight='bold', pad=12)
ax.yaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
ax.set_facecolor('white')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'subdataset_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved subdataset_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Image preprocessing comparison (Original / HE / CLAHE)
# ─────────────────────────────────────────────────────────────────────────────
sample_pos = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\MATLAB\Demonstration Code\Testing Image\POSITIVE\PTest (1).jpg"
sample_neg = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\MATLAB\Demonstration Code\Testing Image\NEGATIVE\NTest (1).jpg"

def apply_he(img_bgr):
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def apply_clahe(img_bgr, clip=2.0, tile=(8,8)):
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    img_yuv[:,:,0] = cl.apply(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def resize224(img_bgr):
    return cv2.resize(img_bgr, (224, 224))

for label, sample_path in [('Positive (Glaucoma)', sample_pos), ('Negative (Normal)', sample_neg)]:
    img = cv2.imread(sample_path)
    if img is None:
        print(f"Warning: could not read {sample_path}")
        continue
    orig  = resize224(img)
    he    = resize224(apply_he(img))
    clahe = resize224(apply_clahe(img))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('white')
    titles = ['Original', 'Histogram\nEqualization (HE)', 'CLAHE']
    for ax, im, title in zip(axes, [orig, he, clahe], titles):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
        ax.axis('off')
    fig.suptitle(f'Preprocessing Comparison — {label}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = 'preproc_positive.png' if 'Positive' in label else 'preproc_negative.png'
    plt.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. K-Fold Cross Validation Diagram
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 11)
ax.set_ylim(-0.5, 5.5)
ax.axis('off')

fold_colors = [TEAL, TEAL, TEAL, TEAL, TEAL]
val_color   = CORAL

for fold_i in range(5):
    for block in range(5):
        color = val_color if block == fold_i else TEAL
        rect = mpatches.FancyBboxPatch(
            (block * 2 + 0.1, 4 - fold_i + 0.1), 1.8, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='white', linewidth=2
        )
        ax.add_patch(rect)
        label = 'Val' if block == fold_i else 'Train'
        ax.text(block * 2 + 1, 4 - fold_i + 0.5, label,
                ha='center', va='center', color='white', fontsize=10, fontweight='bold')

    ax.text(10.2, 4 - fold_i + 0.5, f'Fold {fold_i+1}',
            ha='left', va='center', fontsize=11, color=DARK)

# Column headers
for i in range(5):
    ax.text(i * 2 + 1, 4.7, f'Subset {i+1}', ha='center', va='center', fontsize=10, color=GRAY)

ax.text(5.0, 5.2, '5-Fold Stratified Cross-Validation  (11,031 Train Images)',
        ha='center', va='center', fontsize=13, fontweight='bold', color=DARK)

# Legend
train_p = mpatches.Patch(color=TEAL, label='Training Fold')
val_p   = mpatches.Patch(color=CORAL, label='Validation Fold')
ax.legend(handles=[train_p, val_p], loc='lower center', fontsize=10, frameon=False,
          bbox_to_anchor=(0.5, -0.18))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'kfold_diagram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved kfold_diagram.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Model Performance Comparison — grouped bar chart (Accuracy & AUC)
# ─────────────────────────────────────────────────────────────────────────────
results = {
    # (preprocessing, model): (accuracy, auc, sensitivity, specificity)
    ('Original', 'AlexNet'):      (0.7594, 0.8448, 0.7821, 0.7366),
    ('Original', 'VGG-16'):       (0.8564, 0.9258, 0.8427, 0.8701),
    ('Original', 'DenseNet-121'): (0.8237, 0.8988, 0.7649, 0.8828),
    ('Original', 'ResNet-18'):    (0.5392, 0.6202, 0.9295, 0.1471),
    ('Original', 'MobileNet-V2'): (0.7943, 0.8731, 0.7776, 0.8111),
    ('HE',       'AlexNet'):      (0.7970, 0.8743, 0.7342, 0.8601),
    ('HE',       'VGG-16'):       (0.8559, 0.9270, 0.7830, 0.9292),
    ('HE',       'DenseNet-121'): (0.8464, 0.9237, 0.8345, 0.8583),
    ('HE',       'ResNet-18'):    (0.6865, 0.7336, 0.6808, 0.6921),
    ('HE',       'MobileNet-V2'): (0.7503, 0.8355, 0.6275, 0.8738),
    ('CLAHE',    'AlexNet'):      (0.7879, 0.8689, 0.8020, 0.7738),
    ('CLAHE',    'VGG-16'):       (0.8278, 0.9150, 0.8734, 0.7820),
    ('CLAHE',    'DenseNet-121'): (0.5918, 0.6331, 0.8137, 0.3688),
    ('CLAHE',    'ResNet-18'):    (0.5723, 0.6331, 0.2794, 0.8665),
    ('CLAHE',    'MobileNet-V2'): (0.7667, 0.8484, 0.6483, 0.8856),
}

models = ['AlexNet', 'VGG-16', 'DenseNet-121', 'ResNet-18', 'MobileNet-V2']
preps  = ['Original', 'HE', 'CLAHE']
prep_colors = [TEAL, NAVY, MINT]

# Accuracy grouped by model
acc_data = {p: [results[(p, m)][0] for m in models] for p in preps}
auc_data = {p: [results[(p, m)][1] for m in models] for p in preps}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

x = np.arange(len(models))
w = 0.25

for ax, data, metric in zip(axes, [acc_data, auc_data], ['Accuracy', 'AUC']):
    for i, (prep, color) in enumerate(zip(preps, prep_colors)):
        bars = ax.bar(x + (i-1)*w, data[prep], w, color=color, label=prep, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=15)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f'Validation {metric} — All Models & Preprocessing', fontsize=12, fontweight='bold', pad=10)
    ax.yaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
    ax.set_facecolor('white')
    ax.legend(fontsize=10)
    ax.axhline(0.5, color='lightgray', linewidth=0.8, linestyle='--')

plt.tight_layout(pad=2)
plt.savefig(os.path.join(OUT, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Heatmap — Accuracy across Preprocessing × Model
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib.colors as mcolors

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
fig.patch.set_facecolor('white')

metrics_list = [
    ('Accuracy',    0),
    ('AUC',         1),
]

for ax, (metric_name, idx) in zip(axes, metrics_list):
    matrix = np.array([[results[(p, m)][idx] for m in models] for p in preps])
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(preps)))
    ax.set_xticklabels(models, fontsize=10, rotation=20, ha='right')
    ax.set_yticklabels(preps, fontsize=11)
    ax.set_title(f'{metric_name} Heatmap', fontsize=12, fontweight='bold', pad=8)
    for i in range(len(preps)):
        for j in range(len(models)):
            val = matrix[i, j]
            c = 'white' if val < 0.65 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, color=c, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

plt.tight_layout(pad=2)
plt.savefig(os.path.join(OUT, 'heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Sensitivity / Specificity grouped bar — best model per preprocessing
# ─────────────────────────────────────────────────────────────────────────────
best = {
    'Original\nVGG-16':       (0.8427, 0.8701),
    'HE\nVGG-16':             (0.7830, 0.9292),
    'CLAHE\nVGG-16':          (0.8734, 0.7820),
}

labels_b = list(best.keys())
sens  = [best[k][0] for k in labels_b]
spec  = [best[k][1] for k in labels_b]

x = np.arange(len(labels_b))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor('white')
b1 = ax.bar(x - w/2, sens, w, color=TEAL, label='Sensitivity', zorder=3)
b2 = ax.bar(x + w/2, spec, w, color=CORAL, label='Specificity', zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(labels_b, fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Sensitivity vs Specificity — Best Models per Preprocessing', fontsize=12, fontweight='bold', pad=10)
ax.yaxis.grid(True, color=LGRAY, linewidth=0.8, zorder=0)
ax.set_facecolor('white')
ax.legend(fontsize=11)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.3f}',
            ha='center', va='bottom', fontsize=9, color=DARK)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'sens_spec.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved sens_spec.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Confusion matrix formula / visual explanation
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 7)
ax.set_ylim(0, 5)
ax.axis('off')

cm_colors = [[TEAL, CORAL], [CORAL, TEAL]]
cm_labels = [['TP', 'FN'], ['FP', 'TN']]
cm_desc   = [['True Positive', 'False Negative'],
             ['False Positive', 'True Negative']]

for i in range(2):
    for j in range(2):
        rect = mpatches.FancyBboxPatch(
            (1.0 + j*2.2, 1.0 + (1-i)*1.5), 2.0, 1.4,
            boxstyle="round,pad=0.05",
            facecolor=cm_colors[i][j], edgecolor='white', linewidth=2, alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(2.0 + j*2.2, 1.85 + (1-i)*1.5, cm_labels[i][j],
                ha='center', va='center', fontsize=18, fontweight='bold', color='white')
        ax.text(2.0 + j*2.2, 1.35 + (1-i)*1.5, cm_desc[i][j],
                ha='center', va='center', fontsize=8, color='white', style='italic')

ax.text(3.2, 4.5, 'Confusion Matrix', ha='center', va='center',
        fontsize=14, fontweight='bold', color=DARK)
ax.text(3.2, 4.1, 'Predicted', ha='center', va='center', fontsize=11, color=GRAY)
ax.text(0.4, 2.5, 'Actual', ha='center', va='center', fontsize=11, color=GRAY, rotation=90)

# Axes labels
ax.text(2.0, 0.75, 'Predicted\nPositive', ha='center', va='center', fontsize=9, color=DARK)
ax.text(4.2, 0.75, 'Predicted\nNegative', ha='center', va='center', fontsize=9, color=DARK)
ax.text(0.65, 2.7, 'Actual\nPositive', ha='center', va='center', fontsize=9, color=DARK)
ax.text(0.65, 1.2, 'Actual\nNegative', ha='center', va='center', fontsize=9, color=DARK)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'cm_diagram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved cm_diagram.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Evaluation metrics formula card
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3.5))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.5)
ax.axis('off')

metrics_def = [
    ('Accuracy',    'TP + TN',        'TP + TN + FP + FN', TEAL),
    ('Sensitivity', 'TP',             'TP + FN',            NAVY),
    ('Specificity', 'TN',             'TN + FP',            MINT),
    ('AUC',         'Area under',     'ROC Curve',          CORAL),
]

for idx, (name, num, den, color) in enumerate(metrics_def):
    x0 = idx * 2.4 + 0.3
    rect = mpatches.FancyBboxPatch((x0, 0.2), 2.1, 3.0,
        boxstyle="round,pad=0.06",
        facecolor=color, edgecolor='white', linewidth=2, alpha=0.15)
    ax.add_patch(rect)
    ax.text(x0 + 1.05, 2.85, name, ha='center', va='center',
            fontsize=11, fontweight='bold', color=color)
    ax.text(x0 + 1.05, 2.1, num,  ha='center', va='center', fontsize=9.5, color=DARK)
    ax.plot([x0+0.2, x0+1.9], [1.7, 1.7], color=DARK, linewidth=1.2)
    ax.text(x0 + 1.05, 1.2, den,  ha='center', va='center', fontsize=9.5, color=DARK)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'metrics_formula.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved metrics_formula.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. CNN Model Architecture comparison table figure
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor('white')
ax.axis('off')

col_labels = ['Model', 'Year', 'Depth', 'Params (M)', 'Key Feature', 'Input Size']
table_data = [
    ['AlexNet',      '2012', '8',  '60.9',  'ReLU + Dropout',      '224×224'],
    ['VGG-16',       '2014', '16', '138.4', '3×3 Conv Stacks',     '224×224'],
    ['ResNet-18',    '2015', '18', '11.7',  'Residual Connections', '224×224'],
    ['DenseNet-121', '2017', '121','7.98',  'Dense Connections',   '224×224'],
    ['MobileNet-V2', '2018', '53', '3.4',   'Inverted Residuals',  '224×224'],
]

header_colors = [[TEAL]*6]
row_colors = [['#E8F5F7']*6, [WHITE]*6] * 3

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(11)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    if r == 0:
        cell.set_facecolor(TEAL)
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 1:
        cell.set_facecolor('#E8F5F7')
    else:
        cell.set_facecolor(WHITE)

plt.savefig(os.path.join(OUT, 'model_table.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved model_table.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11. ROC curve overlay for best models (Original preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
# Use existing roc_curve.png for original_vgg16 — copy it to assets
import shutil
src_roc_orig = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\results\custom_experiments_Run2\original_vgg16\roc_curve.png"
src_roc_he   = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\results\custom_experiments_Run2\he_vgg16\roc_curve.png"
src_roc_cl   = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\results\custom_experiments_Run2\clahe_vgg16\roc_curve.png"
src_cm_orig  = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\results\custom_experiments_Run2\original_vgg16\confusion_matrix.png"
src_cm_he    = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\results\custom_experiments_Run2\he_vgg16\confusion_matrix.png"
src_cm_cl    = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\results\custom_experiments_Run2\clahe_vgg16\confusion_matrix.png"
src_lc_orig  = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\results\custom_experiments_Run2\original_vgg16\learning_curves.png"

for src, dst in [
    (src_roc_orig, 'roc_orig_vgg16.png'),
    (src_roc_he,   'roc_he_vgg16.png'),
    (src_roc_cl,   'roc_clahe_vgg16.png'),
    (src_cm_orig,  'cm_orig_vgg16.png'),
    (src_cm_he,    'cm_he_vgg16.png'),
    (src_cm_cl,    'cm_clahe_vgg16.png'),
    (src_lc_orig,  'lc_orig_vgg16.png'),
]:
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, dst))
        print(f"Copied {dst}")

# Also copy the overall comparison plots
src_perf = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\results\custom_experiments_Run2\model_performance_comparison.png"
src_heat = r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\results\custom_experiments_Run2\performance_heatmaps.png"
src_trade= r"D:\OneDrive\A4_JobREF\9Z_CodeWork\2403CW_GlaucomeClassification\Python_Code\results\custom_experiments_Run2\sensitivity_specificity_tradeoff.png"
for src, dst in [(src_perf,'perf_comparison.png'),(src_heat,'perf_heatmap_existing.png'),(src_trade,'sens_spec_existing.png')]:
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, dst))
        print(f"Copied {dst}")

print("\nAll figures generated in:", OUT)
print("Files:", os.listdir(OUT))
