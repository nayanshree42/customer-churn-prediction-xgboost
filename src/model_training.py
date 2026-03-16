import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def split_data(df, target_col='Churn', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y,
                            test_size=test_size,
                            stratify=y,
                            random_state=random_state)


def get_baseline_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42)
    }


def compare_models(models: dict, X_train, y_train, cv=5):
    """
    Run cross-validated ROC-AUC for each model.
    Returns dict of {model_name: mean_auc_score}.
    """
    results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train,
            cv=skf, scoring='roc_auc', n_jobs=-1
        )
        results[name] = scores
        print(f"{name}: Mean AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    return results


def tune_xgboost(X_train, y_train, n_iter=30, cv=5, random_state=42):
    """
    RandomizedSearchCV tuning for XGBoost.
    Returns best estimator.
    """
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        xgb, param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=skf,
        verbose=1,
        n_jobs=-1,
        random_state=random_state
    )
    search.fit(X_train, y_train)

    print(f"\nBest Params: {search.best_params_}")
    print(f"Best CV AUC: {search.best_score_:.4f}")

    return search.best_estimator_


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Full evaluation: AUC, confusion matrix, classification report, ROC curve.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*50}")
    print(f"{model_name} — Test ROC-AUC: {auc:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['No Churn', 'Churn']))

    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                ax=axes[0])
    axes[0].set_title(f'{model_name} — Confusion Matrix')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')

    # ROC Curve
    RocCurveDisplay.from_predictions(
        y_test, y_prob,
        name=model_name,
        ax=axes[1]
    )
    axes[1].set_title(f'{model_name} — ROC Curve (AUC={auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--')

    plt.tight_layout()
    plt.savefig(f'outputs/plots/{model_name.replace(" ", "_")}_eval.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    return auc
