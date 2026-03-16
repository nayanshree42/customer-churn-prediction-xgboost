import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compute_shap_values(model, X_train, X_test):
    """
    Compute SHAP values using TreeExplainer.
    Returns explainer and shap_values for test set.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    return explainer, shap_values


def plot_shap_summary(shap_values, X_test, save_path='outputs/plots/shap_summary.png'):
    """
    Beeswarm summary plot: global feature importance with direction.
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Summary Plot — Feature Impact on Churn Prediction',
              fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_shap_bar(shap_values, X_test, save_path='outputs/plots/shap_bar.png'):
    """
    Bar plot of mean absolute SHAP values — clean global importance ranking.
    """
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.title('SHAP Feature Importance (Mean |SHAP Value|)',
              fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_shap_waterfall(shap_values, index=0,
                        save_path='outputs/plots/shap_waterfall.png'):
    """
    Waterfall plot for a single prediction explanation.
    """
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values[index], show=False)
    plt.title(f'SHAP Waterfall — Prediction #{index} Explained',
              fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_shap_dependence(shap_values, X_test, feature,
                         save_path=None):
    """
    Dependence plot: how one feature's value affects its SHAP value,
    colored by the most interacting feature.
    """
    if save_path is None:
        save_path = f'outputs/plots/shap_dependence_{feature}.png'

    plt.figure(figsize=(8, 5))
    shap.dependence_plot(feature, shap_values.values, X_test,
                         show=False)
    plt.title(f'SHAP Dependence Plot — {feature}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def get_top_shap_features(shap_values, X_test, n=10) -> pd.DataFrame:
    """
    Return a DataFrame of top-N features by mean absolute SHAP value.
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': mean_abs
    }).sort_values('mean_abs_shap', ascending=False).head(n)
    return importance_df
