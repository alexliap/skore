# %%
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from skore import ComparisonReport, CrossValidationReport
X, y = load_breast_cancer(return_X_y=True)
estimator_1 = LogisticRegression()
# estimator_2 = LogisticRegression(C=20)  # Different regularization
estimator_2 = RandomForestClassifier(random_state=0)
report_1 = CrossValidationReport(estimator_1, X, y)
report_2 = CrossValidationReport(estimator_2, X, y)
report = ComparisonReport([report_1, report_2])
report = ComparisonReport({"model1": report_1, "model2": report_2})

# %%
df = report.metrics.report_metrics().copy()
df

# %%
import pandas as pd
pd.options.display.max_rows = 999

df_full = report.metrics.report_metrics(aggregate=None)
df_full

# %%
display_1 = report_1.metrics.roc()
# display_1.plot(roc_curve_kwargs={"alpha": 0.5})

# %%
display_2 = report_2.metrics.roc()
# display_2.plot(roc_curve_kwargs={"alpha": 0.5})

# %%
import matplotlib.pyplot as plt
from skore.sklearn._plot.utils import _validate_style_kwargs

pos_label = 1
roc_curve_kwargs = [{}, {}]

lines = []
line_kwargs = {"alpha": 0.2}

ml_task = "binary classification"
data_source = "test"

fig, ax = plt.subplots()
displays = [display_1, display_2]
for display_idx, display in enumerate(displays):
    est_name = display.estimator_names[0]
    fpr_cv = display.fpr[pos_label]
    tpr_cv = display.tpr[pos_label]
    roc_auc_cv = display.roc_auc[pos_label]

    line_kwargs_validated = _validate_style_kwargs(
        line_kwargs, roc_curve_kwargs[display_idx]
    )
    line_kwargs_validated["label"] = (
        f"{est_name}" # (AUC = {roc_auc_est:0.2f})"
    )
    
    for fpr_est, tpr_est in zip(fpr_cv, tpr_cv):
        (line,) = ax.plot(fpr_est, tpr_est, **line_kwargs_validated, color=plt.get_cmap("tab10")(display_idx % 10))
        lines.append(line)

info_pos_label = (
    f"\n(Positive label: {pos_label})" if pos_label is not None else ""
)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    title=f"{ml_task.title()} on $\\bf{{{data_source}}}$ set",
)
# %%
displays[0].fpr[pos_label]
# %%

[len(x) for x in displays[0].fpr[pos_label]]
# %%
