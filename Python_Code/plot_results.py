import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for premium visualization
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

def plot_custom_experiments(summary_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found at {summary_path}")
        return
        
    try:
        df = pd.read_excel(summary_path)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return
        
    print(f"Loaded summary with columns: {list(df.columns)}")
    
    # Find metric columns
    acc_col = [c for c in df.columns if 'Accuracy' in c]
    auc_col = [c for c in df.columns if 'AUC' in c]
    sens_col = [c for c in df.columns if 'Sensitivity' in c]
    spec_col = [c for c in df.columns if 'Specificity' in c]
    
    if not acc_col or not auc_col:
        print("Error: Could not find Accuracy or AUC columns in the summary file.")
        return
        
    acc_col = acc_col[0]
    auc_col = auc_col[0]
    sens_col = sens_col[0] if sens_col else None
    spec_col = spec_col[0] if spec_col else None
    
    prefix = acc_col.split('_')[0]
    print(f"Detected evaluation prefix: {prefix}")
    
    # Process scores to percentages for consistency
    df['Accuracy (%)'] = df[acc_col] * 100
    df['AUC'] = df[auc_col]
    if sens_col:
        df['Sensitivity (%)'] = df[sens_col] * 100
    if spec_col:
        df['Specificity (%)'] = df[spec_col] * 100

    # Capitalize model and dataset names for clean labels
    df['Model Name'] = df['Model'].apply(lambda x: x.replace('_', ' ').title())
    df['Preprocessing'] = df['Dataset'].apply(lambda x: x.upper())
    
    # Define a premium color palette mapping datasets to distinct tech accent colors:
    # Original -> Ocean Blue, HE -> Sunset Orange, CLAHE -> Emerald Teal
    unique_datasets = df['Preprocessing'].unique()
    palette_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(unique_datasets)]
    color_map = dict(zip(unique_datasets, palette_colors))
    
    # --- Plot 1: Accuracy and AUC comparison by Model ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Accuracy Bar Chart
    sns.barplot(
        data=df,
        x='Model Name',
        y='Accuracy (%)',
        hue='Preprocessing',
        palette=color_map,
        ax=axes[0],
        edgecolor='black',
        linewidth=0.5
    )
    axes[0].set_title(f'{prefix} Accuracy by Model & Preprocessing')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_xlabel('Model Architecture')
    
    # Adjust y-limit dynamically
    min_acc = df['Accuracy (%)'].min()
    axes[0].set_ylim(max(0, min_acc - 5), 105)
    
    for p in axes[0].patches:
        height = p.get_height()
        if height > 0:
            axes[0].annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, color='black', xytext=(0, 3),
                        textcoords='offset points')
    
    # AUC Bar Chart
    sns.barplot(
        data=df,
        x='Model Name',
        y='AUC',
        hue='Preprocessing',
        palette=color_map,
        ax=axes[1],
        edgecolor='black',
        linewidth=0.5
    )
    axes[1].set_title(f'{prefix} AUC by Model & Preprocessing')
    axes[1].set_ylabel('Area Under ROC Curve (AUC)')
    axes[1].set_xlabel('Model Architecture')
    
    # Adjust y-limit dynamically
    min_auc = df['AUC'].min()
    axes[1].set_ylim(max(0.0, min_auc - 0.05), 1.02)
    
    for p in axes[1].patches:
        height = p.get_height()
        if height > 0:
            axes[1].annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, color='black', xytext=(0, 3),
                        textcoords='offset points')
                        
    plt.suptitle('Glaucoma Classification Benchmark: Model & Dataset Performance', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'model_performance_comparison.png')}")
    
    # --- Plot 2: Heatmaps of Performance ---
    acc_pivot = df.pivot(index='Model Name', columns='Preprocessing', values='Accuracy (%)')
    auc_pivot = df.pivot(index='Model Name', columns='Preprocessing', values='AUC')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # Accuracy Heatmap
    sns.heatmap(
        acc_pivot,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=.5,
        ax=axes[0],
        cbar_kws={'label': 'Accuracy (%)'}
    )
    axes[0].set_title(f'{prefix} Accuracy (%)')
    axes[0].set_ylabel('Model')
    axes[0].set_xlabel('Preprocessing Method')
    
    # AUC Heatmap
    sns.heatmap(
        auc_pivot,
        annot=True,
        fmt=".3f",
        cmap="Oranges",
        linewidths=.5,
        ax=axes[1],
        cbar_kws={'label': 'AUC'}
    )
    axes[1].set_title(f'{prefix} AUC')
    axes[1].set_ylabel('Model')
    axes[1].set_xlabel('Preprocessing Method')
    
    plt.suptitle('Performance Heatmaps (Models vs Preprocessing)', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_heatmaps.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'performance_heatmaps.png')}")
    
    # --- Plot 3: Sensitivity vs Specificity Trade-off ---
    if sens_col and spec_col:
        plt.figure(figsize=(10, 8), dpi=300)
        
        sns.scatterplot(
            data=df,
            x='Sensitivity (%)',
            y='Specificity (%)',
            hue='Preprocessing',
            style='Model Name',
            s=120,
            palette=color_map,
            edgecolor='black',
            alpha=0.85
        )
        
        # Reference lines
        plt.axvline(90, color='red', linestyle='--', alpha=0.5, label='90% Target')
        plt.axhline(90, color='red', linestyle='--', alpha=0.5)
        
        plt.title(f'Sensitivity vs Specificity Trade-off ({prefix} Set)', fontsize=14, pad=15)
        plt.xlabel('Sensitivity (Recall) (%)')
        plt.ylabel('Specificity (%)')
        
        min_sens = df['Sensitivity (%)'].min()
        min_spec = df['Specificity (%)'].min()
        plt.xlim(max(0, min_sens - 5), 105)
        plt.ylim(max(0, min_spec - 5), 105)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sensitivity_specificity_tradeoff.png'), bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'sensitivity_specificity_tradeoff.png')}")
        
    print(f"All plots successfully generated and saved to: {output_dir}")

if __name__ == "__main__":
    # Determine the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_file = os.path.join(script_dir, "results", "custom_experiments", "custom_experiments_summary.xlsx")
    out_directory = os.path.join(script_dir, "results", "custom_experiments")
    
    plot_custom_experiments(summary_file, out_directory)
