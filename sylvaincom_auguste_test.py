# %%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from skore import ComparisonReport, CrossValidationReport
X, y = make_classification(random_state=42)
estimator_1 = LogisticRegression()
estimator_2 = LogisticRegression(C=2)  # Different regularization
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
display_1.plot(roc_curve_kwargs={"alpha": 0.5})

# %%
display_2 = report_2.metrics.roc()
display_2.plot(roc_curve_kwargs={"alpha": 0.5})

# %%
display_1.fpr

# %%
display_1.tpr

# %%
display_1.roc_auc

# %%
display_1.estimator_names

# %%
import matplotlib.pyplot as plt
from skore.sklearn._plot.utils import _validate_style_kwargs

pos_label = 1
roc_curve_kwargs = [{}, {}]

lines = []
line_kwargs = {}

ml_task = "binary classification"
data_source = "test"

fig, ax = plt.subplots()
displays = [display_1, display_2]
for display_idx, display in enumerate(displays):
    est_name = display.estimator_names[0]
    fpr_est = display.fpr[pos_label][display_idx]
    tpr_est = display.tpr[pos_label][display_idx]
    roc_auc_est = display.roc_auc[pos_label][display_idx]

    line_kwargs_validated = _validate_style_kwargs(
        line_kwargs, roc_curve_kwargs[display_idx]
    )
    line_kwargs_validated["label"] = (
        f"{est_name} (AUC = {roc_auc_est:0.2f})"
    )
    (line,) = ax.plot(fpr_est, tpr_est, **line_kwargs_validated, alpha=0.5)
    lines.append(line)

info_pos_label = (
    f"\n(Positive label: {pos_label})" if pos_label is not None else ""
)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    title=f"{ml_task.title()} on $\\bf{{{data_source}}}$ set",
)