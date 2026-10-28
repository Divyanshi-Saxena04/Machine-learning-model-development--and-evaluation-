import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV,
                                     cross_validate, learning_curve, validation_curve)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, confusion_matrix, roc_curve,
                             precision_recall_curve, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
SEED = 42
rng = np.random.default_rng(SEED)
R = {}

# ============================================================ 1. DATA CLEANING
raw = pd.read_csv("titanic.csv")
R["raw_shape"] = list(raw.shape)
R["missing"] = {k: int(v) for k, v in raw.isna().sum().items() if v > 0}
R["dupes"] = int(raw.duplicated().sum())
R["class_balance"] = raw.survived.value_counts().to_dict()

# drop: 'deck' (77% missing); columns that duplicate information (class, who, adult_male, embark_town, alive, alone)
df = raw.drop(columns=["deck", "class", "who", "adult_male", "embark_town", "alive", "alone"]).copy()
# feature engineering
df["family_size"] = df.sibsp + df.parch + 1
df["is_alone"] = (df.family_size == 1).astype(int)
df["log_fare"] = np.log1p(df.fare)
R["clean_shape"] = list(df.shape)

# EDA numbers
R["eda"] = dict(
    by_sex=raw.groupby("sex").survived.mean().round(3).to_dict(),
    by_class=raw.groupby("pclass").survived.mean().round(3).to_dict(),
    by_embarked=raw.groupby("embarked").survived.mean().round(3).to_dict(),
    by_alone=df.groupby("is_alone").survived.mean().round(3).to_dict(),
    age_mean=float(raw.age.mean()), age_median=float(raw.age.median()),
    fare_skew=float(raw.fare.skew()), logfare_skew=float(df.log_fare.skew()),
    corr_age_pclass=float(raw[["age", "pclass"]].corr().iloc[0, 1]))

fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
sns.barplot(data=raw, x="sex", y="survived", ax=ax[0, 0], palette=["#C0504D", "#2F5D8A"], errorbar=("ci", 95), hue="sex", legend=False)
ax[0, 0].set_title("Survival rate by sex"); ax[0, 0].set_ylabel("Survival rate"); ax[0, 0].set_xlabel("")
sns.barplot(data=raw, x="pclass", y="survived", ax=ax[0, 1], color="#2F5D8A", errorbar=("ci", 95))
ax[0, 1].set_title("Survival rate by class"); ax[0, 1].set_ylabel(""); ax[0, 1].set_xlabel("Passenger class")
sns.histplot(data=raw, x="age", hue="survived", multiple="stack", bins=25, ax=ax[0, 2], palette=["#C0504D", "#2F5D8A"])
ax[0, 2].set_title("Age distribution by outcome"); ax[0, 2].set_xlabel("Age")
sns.histplot(raw.fare, bins=40, ax=ax[1, 0], color="#7A7A7A"); ax[1, 0].set_title(f"Fare (skew {raw.fare.skew():.1f})")
sns.histplot(df.log_fare, bins=40, ax=ax[1, 1], color="#2F5D8A"); ax[1, 1].set_title(f"log(1 + fare) (skew {df.log_fare.skew():.1f})")
miss = raw.isna().mean().sort_values(ascending=False)
miss = miss[miss > 0] * 100
ax[1, 2].barh(miss.index, miss.values, color="#C0504D"); ax[1, 2].set_title("Missing values (%)"); ax[1, 2].invert_yaxis()
for i, v in enumerate(miss.values): ax[1, 2].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
ax[1, 2].set_xlim(0, 100)
plt.tight_layout(); plt.savefig("m_fig1_eda.png", dpi=200); plt.close()

# ============================================================ 2. SPLIT + PREPROCESSING
features_num = ["age", "log_fare", "sibsp", "parch", "family_size"]
features_cat = ["pclass", "sex", "embarked"]
X = df[features_num + features_cat + ["is_alone"]].copy()
X["pclass"] = X["pclass"].astype(str)
y = df["survived"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
R["split"] = dict(train=len(X_tr), test=len(X_te), train_rate=float(y_tr.mean()), test_rate=float(y_te.mean()))

def make_pre(scale=True):
    num = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    cat = [("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(drop="first", handle_unknown="ignore"))]
    return ColumnTransformer([("num", Pipeline(num), features_num), ("cat", Pipeline(cat), features_cat)],
                             remainder="passthrough")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ============================================================ 3. MODELS
models = {
    "Baseline (majority)": Pipeline([("pre", make_pre()), ("clf", DummyClassifier(strategy="most_frequent"))]),
    "Logistic regression": Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
    "Decision tree (unpruned)": Pipeline([("pre", make_pre(False)), ("clf", DecisionTreeClassifier(random_state=SEED))]),
    "Decision tree (tuned)": Pipeline([("pre", make_pre(False)), ("clf", DecisionTreeClassifier(random_state=SEED))]),
    "Random forest": Pipeline([("pre", make_pre(False)), ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
}
grids = {
    "Logistic regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "Decision tree (tuned)": {"clf__max_depth": [2, 3, 4, 5, 6, 8], "clf__min_samples_leaf": [1, 5, 10, 20]},
    "Random forest": {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 6, 8, None], "clf__min_samples_leaf": [1, 3, 5]},
}
R["best_params"] = {}
R["grid_cv_auc"] = {}
fitted = {}
for name, pipe in models.items():
    if name in grids:
        gs = GridSearchCV(pipe, grids[name], cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
        fitted[name] = gs.best_estimator_
        R["best_params"][name] = {k.replace("clf__", ""): (v if not isinstance(v, (np.integer, np.floating)) else v.item()) for k, v in gs.best_params_.items()}
        R["grid_cv_auc"][name] = float(gs.best_score_)
    else:
        fitted[name] = pipe.fit(X_tr, y_tr)

# ============================================================ 4. EVALUATION
scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
R["cv"], R["train"], R["test"], R["cm"] = {}, {}, {}, {}
proba, pred = {}, {}
for name, est in fitted.items():
    cvr = cross_validate(est, X_tr, y_tr, cv=cv, scoring=scoring, return_train_score=True)
    R["cv"][name] = {k: [float(cvr[f"test_{k}"].mean()), float(cvr[f"test_{k}"].std())] for k in scoring}
    R["cv"][name]["train_auc"] = float(cvr["train_roc_auc"].mean())
    R["cv"][name]["train_acc"] = float(cvr["train_accuracy"].mean())
    p = est.predict_proba(X_te)[:, 1]; yh = est.predict(X_te)
    proba[name], pred[name] = p, yh
    R["test"][name] = dict(accuracy=accuracy_score(y_te, yh), precision=precision_score(y_te, yh, zero_division=0),
                           recall=recall_score(y_te, yh), f1=f1_score(y_te, yh), roc_auc=roc_auc_score(y_te, p),
                           pr_auc=average_precision_score(y_te, p), brier=brier_score_loss(y_te, p))
    R["train"][name] = dict(accuracy=accuracy_score(y_tr, est.predict(X_tr)), roc_auc=roc_auc_score(y_tr, est.predict_proba(X_tr)[:, 1]))
    R["cm"][name] = confusion_matrix(y_te, yh).tolist()

# bootstrap CI for test AUC and accuracy of the chosen (logistic) model
def boot_ci(name, metric, B=2000):
    vals = []
    n = len(y_te); yt = y_te.values
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(set(yt[idx])) < 2: continue
        vals.append(roc_auc_score(yt[idx], proba[name][idx]) if metric == "auc" else accuracy_score(yt[idx], pred[name][idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
R["boot"] = {n: {"auc": boot_ci(n, "auc"), "acc": boot_ci(n, "acc")} for n in ["Logistic regression", "Decision tree (tuned)", "Random forest"]}

# McNemar-style comparison LR vs RF (paired disagreement counts)
from scipy.stats import binomtest
a = (pred["Logistic regression"] == y_te.values); b = (pred["Random forest"] == y_te.values)
n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
R["mcnemar"] = dict(lr_only=n01, rf_only=n10, p=float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0)

# ---- Fig 2: confusion matrices
show = ["Baseline (majority)", "Logistic regression", "Decision tree (unpruned)", "Decision tree (tuned)", "Random forest"]
fig, ax = plt.subplots(1, 5, figsize=(16, 3.8))
for a_, n in zip(ax, show):
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred[n]), display_labels=["Died", "Survived"]).plot(ax=a_, cmap="Blues", colorbar=False, values_format="d")
    a_.set_title(n, fontsize=10); a_.grid(False)
    a_.set_xlabel("Predicted", fontsize=9); a_.set_ylabel("Actual" if a_ is ax[0] else "", fontsize=9)
plt.tight_layout(); plt.savefig("m_fig2_confusion.png", dpi=200); plt.close()

# ---- Fig 3: ROC + PR
cols = {"Baseline (majority)": "#999999", "Logistic regression": "#2F5D8A", "Decision tree (unpruned)": "#C0504D",
        "Decision tree (tuned)": "#E4A84B", "Random forest": "#5B8C5A"}
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for n in show[1:]:
    fpr, tpr, _ = roc_curve(y_te, proba[n]); ax[0].plot(fpr, tpr, color=cols[n], lw=2, label=f"{n} (AUC {R['test'][n]['roc_auc']:.3f})")
    pr, rc, _ = precision_recall_curve(y_te, proba[n]); ax[1].plot(rc, pr, color=cols[n], lw=2, label=f"{n} (AP {R['test'][n]['pr_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
ax[0].set_title("ROC curves (test set)"); ax[0].legend(fontsize=8, loc="lower right")
ax[1].axhline(y_te.mean(), color="k", ls="--", lw=1, label="Prevalence"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("Precision–recall curves (test set)"); ax[1].legend(fontsize=8, loc="lower left"); ax[1].set_ylim(0.3, 1.02)
plt.tight_layout(); plt.savefig("m_fig3_roc_pr.png", dpi=200); plt.close()

# ---- Fig 4: metric comparison bar chart
met = ["accuracy", "precision", "recall", "f1", "roc_auc"]
fig, ax = plt.subplots(figsize=(12, 4.4))
w = 0.16
for i, n in enumerate(show):
    ax.bar(np.arange(5) + (i - 2) * w, [R["test"][n][m] for m in met], w, label=n, color=cols[n])
ax.set_xticks(range(5)); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]); ax.set_ylim(0, 1.05)
ax.set_title("Test-set performance by model", pad=34); ax.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
plt.tight_layout(); plt.savefig("m_fig4_metrics.png", dpi=200); plt.close()

# ---- Fig 5: overfitting diagnostics (validation curve on depth + learning curves)
depths = list(range(1, 16))
tree_pipe = Pipeline([("pre", make_pre(False)), ("clf", DecisionTreeClassifier(random_state=SEED, min_samples_leaf=1))])
tr_s, va_s = validation_curve(tree_pipe, X_tr, y_tr, param_name="clf__max_depth", param_range=depths, cv=cv, scoring="roc_auc")
R["valcurve"] = dict(depths=depths, train=tr_s.mean(1).tolist(), val=va_s.mean(1).tolist())
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].plot(depths, tr_s.mean(1), "o-", color="#C0504D", label="Training AUC"); ax[0].plot(depths, va_s.mean(1), "o-", color="#2F5D8A", label="CV AUC")
ax[0].fill_between(depths, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), color="#2F5D8A", alpha=.15)
best_d = depths[int(np.argmax(va_s.mean(1)))]; R["valcurve"]["best_depth"] = best_d
ax[0].axvline(best_d, color="k", ls=":", lw=1); ax[0].set_xlabel("Tree max depth"); ax[0].set_ylabel("ROC AUC")
ax[0].set_title("Decision tree: under- vs over-fitting"); ax[0].legend(fontsize=8)
for a_, n, c_ in [(ax[1], "Logistic regression", "#2F5D8A"), (ax[2], "Random forest", "#5B8C5A")]:
    sizes, trs, vas = learning_curve(fitted[n], X_tr, y_tr, cv=cv, scoring="roc_auc", train_sizes=np.linspace(0.1, 1, 8), random_state=SEED, n_jobs=-1)
    a_.plot(sizes, trs.mean(1), "o-", color="#C0504D", label="Training AUC"); a_.plot(sizes, vas.mean(1), "o-", color=c_, label="CV AUC")
    a_.fill_between(sizes, vas.mean(1) - vas.std(1), vas.mean(1) + vas.std(1), color=c_, alpha=.15)
    a_.set_xlabel("Training examples"); a_.set_title(f"Learning curve: {n}"); a_.legend(fontsize=8); a_.set_ylim(0.7, 1.02)
    R.setdefault("learn", {})[n] = dict(gap=float(trs.mean(1)[-1] - vas.mean(1)[-1]), final_val=float(vas.mean(1)[-1]))
plt.tight_layout(); plt.savefig("m_fig5_fit_diagnostics.png", dpi=200); plt.close()

# ---- Fig 6: interpretation (LR odds ratios + RF importance)
lr = fitted["Logistic regression"]
names = lr.named_steps["pre"].get_feature_names_out()
names = [n.replace("num__", "").replace("cat__", "").replace("remainder__", "") for n in names]
coefs = lr.named_steps["clf"].coef_[0]
orr = pd.Series(np.exp(coefs), index=names).sort_values()
R["odds"] = orr.round(3).to_dict()
rf = fitted["Random forest"]
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values()
R["rf_imp"] = imp.round(3).to_dict()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].barh(orr.index, orr.values - 1, left=1, color=["#C0504D" if v < 1 else "#2F5D8A" for v in orr.values])
ax[0].axvline(1, color="k", lw=1); ax[0].set_xscale("log"); ax[0].set_title("Logistic regression: odds ratios (log scale)")
ax[0].set_xlabel("Odds ratio (per 1 SD for numeric / vs. reference for categorical)")
ax[1].barh(imp.index, imp.values, color="#5B8C5A"); ax[1].set_title("Random forest: impurity-based feature importance")
plt.tight_layout(); plt.savefig("m_fig6_interpretation.png", dpi=200); plt.close()

# ---- Fig 7: threshold + calibration
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
th = np.linspace(0.05, 0.95, 91); p = proba["Logistic regression"]
pr_, rc_, f_, ac_ = [], [], [], []
for t in th:
    yh = (p >= t).astype(int)
    pr_.append(precision_score(y_te, yh, zero_division=0)); rc_.append(recall_score(y_te, yh)); f_.append(f1_score(y_te, yh)); ac_.append(accuracy_score(y_te, yh))
ax[0].plot(th, pr_, label="Precision", color="#2F5D8A"); ax[0].plot(th, rc_, label="Recall", color="#C0504D")
ax[0].plot(th, f_, label="F1", color="#5B8C5A"); ax[0].plot(th, ac_, label="Accuracy", color="#7A7A7A", ls="--")
ax[0].axvline(0.5, color="k", ls=":", lw=1); ax[0].set_xlabel("Decision threshold"); ax[0].set_title("Logistic regression: threshold trade-offs"); ax[0].legend(fontsize=8)
R["best_f1_threshold"] = float(th[int(np.argmax(f_))]); R["best_f1"] = float(max(f_))
for n in ["Logistic regression", "Random forest", "Decision tree (tuned)"]:
    fp, mp = calibration_curve(y_te, proba[n], n_bins=6, strategy="quantile"); ax[1].plot(mp, fp, "o-", color=cols[n], label=n)
ax[1].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect"); ax[1].set_xlabel("Mean predicted probability"); ax[1].set_ylabel("Observed survival rate")
ax[1].set_title("Calibration (test set)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("m_fig7_threshold_calibration.png", dpi=200); plt.close()

# ============================================================ 5. ERROR ANALYSIS (chosen model: logistic regression)
te = X_te.copy(); te["y"] = y_te.values; te["pred"] = pred["Logistic regression"]; te["prob"] = proba["Logistic regression"]
te["err"] = (te.y != te.pred).astype(int)
te["FN"] = ((te.y == 1) & (te.pred == 0)).astype(int); te["FP"] = ((te.y == 0) & (te.pred == 1)).astype(int)
R["err"] = dict(
    by_sex=te.groupby("sex").err.agg(["mean", "sum", "count"]).round(3).to_dict("index"),
    by_class=te.groupby("pclass").err.agg(["mean", "sum", "count"]).round(3).to_dict("index"),
    age_missing=te.assign(m=te.age.isna()).groupby("m").err.agg(["mean", "sum", "count"]).round(3).to_dict("index"),
    fn=int(te.FN.sum()), fp=int(te.FP.sum()),
    sex_class_recall={f"{s}-{c}": dict(n=int(len(g)), acc=float(1 - g.err.mean())) for (s, c), g in te.groupby(["sex", "pclass"])})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
g = te.groupby(["sex", "pclass"]).err.mean().unstack() * 100
sns.heatmap(g, annot=True, fmt=".0f", cmap="Reds", ax=ax[0], cbar_kws={"label": "Error rate (%)"}, vmin=0, vmax=60)
ax[0].set_title("Test error rate by sex × class (LR)"); ax[0].set_xlabel("Class"); ax[0].set_ylabel("")
sns.histplot(data=te, x="prob", hue="y", bins=20, multiple="layer", ax=ax[1], palette=["#C0504D", "#2F5D8A"], alpha=.55)
ax[1].axvline(0.5, color="k", ls=":"); ax[1].set_title("Predicted survival probability by true outcome"); ax[1].set_xlabel("Predicted probability")
plt.tight_layout(); plt.savefig("m_fig8_errors.png", dpi=200); plt.close()

# ============================================================ 6. Robustness: repeated CV & dedupe sensitivity
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
R["repeated_cv_auc"] = {n: [float(s.mean()), float(s.std())] for n in ["Logistic regression", "Decision tree (tuned)", "Random forest"]
                        for s in [cross_val_score(fitted[n], X, y, cv=rcv, scoring="roc_auc", n_jobs=-1)]}
dd = df.drop_duplicates(); Xd = dd[features_num + features_cat + ["is_alone"]].copy(); Xd["pclass"] = Xd["pclass"].astype(str)
R["dedupe"] = dict(n=len(dd), auc=float(cross_val_score(fitted["Logistic regression"], Xd, dd.survived, cv=rcv, scoring="roc_auc", n_jobs=-1).mean()))
# ablation: without sex
Xa = X.drop(columns=["sex"]); cat_bak = features_cat
features_cat = ["pclass", "embarked"]
lr_a = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000, C=R["best_params"]["Logistic regression"]["C"]))])
R["ablation_no_sex_auc"] = float(cross_val_score(lr_a, Xa, y, cv=rcv, scoring="roc_auc", n_jobs=-1).mean())
features_cat = cat_bak


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, np.integer): return int(o)
    return o
json.dump(clean(R), open("ml_results.json", "w"), indent=1)
print(json.dumps(clean({k: R[k] for k in ["split", "best_params", "grid_cv_auc", "test", "train", "cv", "boot", "mcnemar", "cm"]}), indent=1))
