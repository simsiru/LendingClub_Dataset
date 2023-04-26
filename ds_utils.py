import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from math import ceil
from typing import Optional, Callable, Tuple
from collections import defaultdict
from sklearn.model_selection import cross_validate, cross_val_predict,\
   GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve
from yellowbrick.classifier import DiscriminationThreshold,\
  PrecisionRecallCurve, ROCAUC, ConfusionMatrix
from yellowbrick.regressor import ResidualsPlot, PredictionError


def evaluate_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_fcn: dict,
    preprocessor=None,
    return_models=True,
    multi_class=False
  ) -> Tuple[pd.DataFrame, dict]:
  """Evaluate multiple models with selected metrics functions"""

  results_dict = defaultdict(list)

  trained_models = {}

  for name, clf in models.items():
    if preprocessor is not None:
      clf_pl = Pipeline(steps=[('preproc', preprocessor), ('clf', clf)])
    else:
      clf_pl = clf

    clf_pl.fit(X_train, y_train)

    if return_models:
      trained_models[name] = clf_pl

    if multi_class:
      pred = clf_pl.predict_proba(X_test)
    else:
      pred = clf_pl.predict(X_test)

    results_dict['model_name'].append(name)

    for name, fcn in metrics_fcn.items():
      results_dict[f'{name}'].append(fcn(y_test, pred).round(3))

  return pd.DataFrame(results_dict), trained_models


def plot_cm_pr_roc_curves_yb(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: Optional[pd.DataFrame]=None,
    y_test: Optional[pd.Series]=None,
    preprocessor=None,
    plot_size: Optional[tuple]=(16,26),
    folds: Optional[int]=5,
    class_labels: Optional[list]=None,
    plot_threshold: bool=True,
    per_class: bool=True
  ) -> None:
  """Plot confusion matrix, precision/recall, threshold and ROC curves for 
  multiple models"""

  plot_cols = 4

  if len(set(y))>2 or not plot_threshold:
    plot_threshold = False
    plot_cols = 3

  fig, ax = plt.subplots(len(models), plot_cols, figsize=plot_size)

  if X_test is None or y_test is None:
    X, X_test, y, y_test = train_test_split(
      X, y, test_size=0.1, stratify=y, random_state=0)

  for i, (name, model) in enumerate(models.items()):
    if preprocessor is not None:
      model = Pipeline(
          steps = [
              ('preprocessor', preprocessor),
              ('clf', model)
          ]
      )

    # Confusion matrix
    cm = ConfusionMatrix(model, classes=class_labels, ax=ax[i][0])
    cm.fit(X, y)
    cm.score(X_test, y_test)
    cm.finalize()

    # Precision/Recall curve
    prc = PrecisionRecallCurve(
      model, per_class=per_class, classes=class_labels, ax=ax[i][1])
    prc.fit(X, y)
    prc.score(X_test, y_test)
    prc.finalize()

    # ROC curve
    roc = ROCAUC(model, per_class=per_class, classes=class_labels, ax=ax[i][2])
    roc.fit(X, y)
    roc.score(X_test, y_test)
    roc.finalize()

    if plot_threshold:
      # Threshold plot
      thres = DiscriminationThreshold(
        model, ax=ax[i][3], n_trials=folds, exclude=['queue_rate', 'fscore'])
      thres.fit(X, y)
      thres.finalize()

  plt.tight_layout()


def plot_residuals_yb(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: Optional[pd.DataFrame]=None,
    y_test: Optional[pd.Series]=None,
    preprocessor=None,
    plot_size: Optional[tuple]=(16,26)
  ) -> None:
  """Plot confusion matrix, precision/recall, threshold and ROC curves for 
  multiple models"""

  fig, ax = plt.subplots(len(models), 2, figsize=plot_size)

  if X_test is None or y_test is None:
    X, X_test, y, y_test = train_test_split(
      X, y, test_size=0.1, random_state=0)

  for i, (name, model) in enumerate(models.items()):
    if preprocessor is not None:
      model = Pipeline(
          steps = [
              ('preprocessor', preprocessor),
              ('clf', model)
          ]
      )

    res = ResidualsPlot(model, ax=ax[i][0])
    res.fit(X, y)
    res.score(X_test, y_test)
    res.finalize()

    pred_er = PredictionError(model, ax=ax[i][1])
    pred_er.fit(X, y)
    pred_er.score(X_test, y_test)
    pred_er.finalize()

  plt.tight_layout()


def plot_cm_pr_roc_curves(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor=None,
    test_X: Optional[pd.DataFrame]=None,
    test_y: Optional[pd.Series]=None,
    plot_size: Optional[tuple]=(16,26),
    cv_predict: Optional[bool]=True,
    folds: Optional[int]=5,
    n_jobs: Optional[int]=-1,
    trained_models: bool=False,
    yb_kwargs: Optional[dict]={'exclude':['queue_rate', 'fscore']}
  ) -> None:
  """Plot confusion matrix, precision/recall, threshold and ROC curves for 
  multiple models"""

  if y.unique()>2:
    multi_class = True
    plot_cols = 3
  else:
    multi_class = False
    plot_cols = 4

  fig, ax = plt.subplots(len(models), plot_cols, figsize=plot_size)

  if test_X is None or test_y is None:
    test_X = X
    test_y = y

  for i, (name, model) in enumerate(models.items()):
    if preprocessor is not None:
      model = Pipeline(
          steps = [
              ('preprocessor', preprocessor),
              ('clf', model)
          ]
      )

    # Confusion matrix
    if cv_predict:
      pred = cross_val_predict(model, X, y, cv=folds, n_jobs=n_jobs)
    else:
      if not trained_models:
        model.fit(X, y)

      pred = model.predict(test_X)

    conf_mat = confusion_matrix(test_y, pred)

    sns.heatmap(
        data=conf_mat,
        annot=True,
        fmt='',
        ax=ax[i][0]
    )

    ax[i][0].set_title(f'{name} confusion matrix')
    ax[i][0].set_xlabel('Predicted label')
    ax[i][0].set_ylabel('True label')

    # Precision/Recall curve
    if cv_predict:
      pred = cross_val_predict(model, X, y, cv=folds, n_jobs=n_jobs, 
                               method='predict_proba')[:,1]
    else:
      pred = model.predict_proba(test_X)[:,1]

    precision, recall, thresholds = precision_recall_curve(test_y, pred)

    ax[i][1].plot(precision, recall)
    ax[i][1].set_title(f'{name} precision recall curve')
    ax[i][1].set_xlabel('Recall')
    ax[i][1].set_ylabel('Precision')

    # ROC curve
    fpr, tpr, thresholds = roc_curve(test_y, pred)

    ax[i][2].plot(fpr, tpr)
    ax[i][2].set_title(f'{name} ROC curve')
    ax[i][2].set_xlabel('False positive rate')
    ax[i][2].set_ylabel('True positive rate')
    ax[i][2].plot([0, 1], [0, 1], 'k--')

    if not multi_class:
      # Threshold plot
      visualizer = DiscriminationThreshold(
        model, ax=ax[i][3], n_trials=folds, **yb_kwargs)
      visualizer.fit(X, y)
      visualizer.finalize()

  plt.tight_layout()


def gridsearch_models(
    models_params: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metrics: list,
    best_metric: str,
    cv_folds: int=5,
    n_jobs: Optional[int]=-1,
    n_iter: Optional[int]=10,
    return_train_score: Optional[bool]=False,
    rand_search: Optional[bool]=True
  ) -> Tuple[pd.DataFrame, dict]:
  """Gridsearch multiple models in terms of a selected metric 
  and calculate other metrics"""

  best_estimators = {}

  results_dict = defaultdict(list)

  for model, param_grid in models_params:
    model_name = model['clf'].__class__.__name__

    if rand_search:
      grid_search = RandomizedSearchCV(
          model, param_grid, cv=cv_folds,
          scoring=metrics, n_jobs=n_jobs,
          refit=best_metric, n_iter=n_iter,
          return_train_score=return_train_score,
          random_state=0)
    else:
      grid_search = GridSearchCV(
          model, param_grid, cv=cv_folds,
          scoring=metrics, n_jobs=n_jobs,
          refit=best_metric, 
          return_train_score=return_train_score)

    grid_search.fit(X_train, y_train)

    result = grid_search.cv_results_
    idx = grid_search.best_index_

    results_dict['model_name'].append(model_name)

    for metric in metrics:
      results_dict[f'mean_{metric}'].append(
          result[f'mean_test_{metric}'][idx].round(3))
      
      results_dict[f'std_{metric}'].append(
          result[f'std_test_{metric}'][idx].round(3))
      
    best_estimators[model_name] = grid_search.best_estimator_

  return pd.DataFrame(results_dict), best_estimators


def cross_validate_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metrics: list,
    preprocessor=None,
    cv_folds: int=5,
    n_jobs: Optional[int]=-1
  ) -> pd.DataFrame:
  """Cross validate multiple models with selected metrics"""

  results_dict = defaultdict(list)

  for name, clf in models.items():
    if preprocessor is not None:
      clf_pl = Pipeline(steps=[('preproc', preprocessor), ('clf', clf)])
    else:
      clf_pl = clf

    result = cross_validate(
        clf_pl,
        X_train,
        y_train,
        cv=cv_folds,
        scoring=metrics,
        n_jobs=n_jobs
    )

    results_dict['model_name'].append(name)

    for metric in metrics:
      results_dict[f'mean_{metric}'].append(
          result[f'test_{metric}'].mean().round(3))
      results_dict[f'std_{metric}'].append(
          result[f'test_{metric}'].std().round(3))

  return pd.DataFrame(results_dict)


def with_hue(
    ax: plt.Axes,
    x_data: pd.Series,
    n_x_cat: int,
    n_hue_cat: int
  ) -> None:
  """Annotate bars with percentages with respect to hue categories"""

  a = [p.get_height() for p in ax.patches]

  patches = [p for p in ax.patches]

  for i in range(n_x_cat):
    total = x_data.value_counts().values[i]

    for j in range(n_hue_cat):
      p_idx = j*n_x_cat+i

      percentage = '{:.1f}%'.format(100*a[p_idx]/total)

      x = patches[p_idx].get_x() + patches[p_idx].get_width() / 2

      y = patches[p_idx].get_height()

      ax.annotate(
          percentage,
          (x, y),
          ha='center',
          va='center',
          xytext=(0, 5),
          textcoords='offset points',
          size=9
      )


def without_hue(
    ax: plt.Axes,
    x_data: pd.Series
  ) -> None:
  """Annotate bars with percentages with respect to x categories"""

  total = len(x_data)

  for p in ax.patches:
    percentage = '{:.1f}%'.format(100*p.get_height()/total)

    x = p.get_x() + p.get_width() / 2

    y = p.get_height()

    ax.annotate(
        percentage,
        (x, y),
        ha='center',
        va='center',
        xytext=(0, 5),
        textcoords='offset points',
        size=10
    )


def plot_subplots_barplots(
    df: pd.DataFrame,
    feature_names: list, 
    plot_func: Callable,
    plot_on_x: Optional[bool]=True,
    plot_hue: Optional[str]=None,
    n_rows: Optional[int]=None, 
    n_cols: Optional[int]=None, 
    size: Optional[tuple]=(20,10),
    kwargs: Optional[dict]={},
    show_percentage: Optional[bool]=True
  ) -> None:
  """Plot subplots with a specified seaborn countplot, barplot function
  which takes care of putting percentages on bars"""

  if n_rows is None or n_cols is None:
    n_rows = ceil(len(feature_names)/4)
    n_cols = 4

  fig, ax = plt.subplots(n_rows, n_cols, figsize=size)

  curr_plot = 0

  feature_names = list(feature_names)

  if plot_hue is not None and plot_hue in feature_names:
    feature_names.remove(plot_hue)

  for i in range(n_rows):
    for j in range(n_cols):
      if curr_plot==len(feature_names):
        fig.delaxes(ax[i][j])
        continue

      bar_order = df[feature_names[curr_plot]].value_counts().index

      if show_percentage:
        if plot_on_x:
          plot_func(
              data=df,
              x=feature_names[curr_plot],
              hue=plot_hue,
              ax=ax[i][j],
              order=bar_order,
              **kwargs
          )
        else:
          plot_func(
              data=df,
              y=feature_names[curr_plot],
              hue=plot_hue,
              ax=ax[i][j],
              order=bar_order,
              **kwargs
          )

        if plot_hue is not None:
          with_hue(
              ax[i][j],
              df[feature_names[curr_plot]],
              df[feature_names[curr_plot]].nunique(),
              df[plot_hue].nunique()
          )
        else:
          without_hue(
              ax[i][j],
              df[feature_names[curr_plot]]
          )

      else:
        if plot_on_x:
          plot_func(
              data=df,
              x=feature_names[curr_plot],
              hue=plot_hue,
              ax=ax[i][j],
              order=bar_order,
              **kwargs
          )
        else:
          plot_func(
              data=df,
              y=feature_names[curr_plot],
              hue=plot_hue,
              ax=ax[i][j],
              order=bar_order,
              **kwargs
          )

      curr_plot += 1

  plt.tight_layout()


def plot_subplots(
    df: pd.DataFrame,
    feature_names: list, 
    plot_func: Callable,
    plot_on_x: Optional[bool]=True,
    plot_hue: Optional[str]=None,
    plot_x: Optional[str]=None,
    plot_y: Optional[str]=None,
    n_rows: Optional[int]=None, 
    n_cols: Optional[int]=None, 
    size: Optional[tuple]=(20,10),
    kwargs: Optional[dict]={}
  ) -> None:
  """Plot subplots with a specified seaborn function"""

  if n_rows is None or n_cols is None:
    n_rows = ceil(len(feature_names)/4)
    n_cols = 4

  fig, ax = plt.subplots(n_rows, n_cols, figsize=size)

  curr_plot = 0

  feature_names = list(feature_names)

  if plot_hue is not None and plot_hue in feature_names:
    feature_names.remove(plot_hue)

  for i in range(n_rows):
    for j in range(n_cols):
      if curr_plot==len(feature_names):
        fig.delaxes(ax[i][j])
        continue

      if plot_on_x:
        plot_func(
            data=df,
            x=feature_names[curr_plot],
            hue=plot_hue,
            y=plot_y,
            ax=ax[i][j],
            **kwargs
        )
      else:
        plot_func(
            data=df,
            y=feature_names[curr_plot],
            hue=plot_hue,
            x=plot_x,
            ax=ax[i][j],
            **kwargs
        )

      curr_plot += 1

  plt.tight_layout()


def inspect_df(df: pd.DataFrame, sel_nan_above_perc: int=60) -> list:
  """Print information about duplicates and NaN value count and percentage
  in each column of a dataframe"""

  perc = df.isna().mean().sort_values(
    ascending=False).rename('percentage')*100

  count = df.isna().sum().sort_values(
    ascending=False).rename('count')

  merged = pd.merge(
      left=perc,
      right=count,
      left_index=True,
      right_index=True
  )

  merged = merged[(merged['count']!=0) & (merged['percentage']>=sel_nan_above_perc)]

  if merged.empty:
    print('No missing values\n')
  else:
    print(f'Rows with NaN values in each column:\n\n{merged}\n')

  dupl_count = df.duplicated().sum()

  if dupl_count==0:
    print('No duplicate rows')
  else:
    print(f'Number of duplicate rows: {dupl_count}')

  return list(merged.index)


"""def backwards_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float=0.05,
    model_type: str='lin_reg'
  ) -> list:
  if model_type=='log_reg':
    lm = sm.Logit(y_train, X_train)
  if model_type=='lin_reg':
    lm = sm.OLS(y_train, X_train)
  if model_type=='log_reg_multi':
    lm = sm.MNLogit(y_train, X_train)
  res = lm.fit()

  if model_type=='log_reg_multi':
    col_max_pvalues = ('', 0)

    for col in res.pvalues.columns:
      max_pvalue = res.pvalues[col].max()

      if col_max_pvalues[1] < max_pvalue:
        col_max_pvalues = (col, max_pvalue)

    max_pvalue = col_max_pvalues[1]
  else:
    max_pvalue = res.pvalues.max()

  if max_pvalue > alpha:
    if model_type=='log_reg_multi':
      idx = res.pvalues[col_max_pvalues[0]][res.pvalues[col_max_pvalues[0]]==max_pvalue].index[0]
    else:
      idx = res.pvalues[res.pvalues==max_pvalue].index[0]

    X_train.drop(
      columns=idx,
      inplace=True
    )

    backwards_selection(
        X_train,
        y_train,
        alpha,
        model_type
    )
  else:
    print(res.summary2())

  return list(X_train.columns)


def multi_plots(
      df: pd.DataFrame,
      plot_info: dict,
      n_rows: int,
      n_cols: int,
      size: Tuple[int, int]=(12,12),
      n_plots: int=None
  ) -> None:
  fig, ax = plt.subplots(
      n_rows,
      n_cols,
      figsize=size
  )

  if n_plots is None:
    n_plots = n_rows * n_cols

  curr_plot = 0

  temp_dict = plot_info.copy()
  for item in temp_dict.items():
    if not item[1]:
      del plot_info[item[0]]


  for row in range(n_rows):
    for col in range(n_cols):
      if curr_plot >= n_plots:
        fig.delaxes(ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col])
        curr_plot += 1
        continue

      if len(plot_info['plot_kinds']) == 1:
        kind = plot_info['plot_kinds'][0]
      else:
        kind = plot_info['plot_kinds'][curr_plot]

      if 'plot_x' in plot_info:
        if len(plot_info['plot_x']) == 1:
          x = plot_info['plot_x'][0]
        else:
          x = plot_info['plot_x'][curr_plot]
      else:
        x = None

      if 'plot_y' in plot_info:
        if len(plot_info['plot_y']) == 1:
          y = plot_info['plot_y'][0]
        else:
          y = plot_info['plot_y'][curr_plot]
      else:
        y = None

      if 'plot_hue' in plot_info:
        if len(plot_info['plot_hue']) == 1:
          hue = plot_info['plot_hue'][0]
        else:
          hue = plot_info['plot_hue'][curr_plot]
      else:
        hue = None

      if 'plot_kwargs' in plot_info:
        if len(plot_info['plot_kwargs']) == 1:
          kwargs = plot_info['plot_kwargs'][0]
        else:
          kwargs = plot_info['plot_kwargs'][curr_plot]

        if kind == 'hist':
          sns.histplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              **kwargs,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
        if kind == 'box':
          sns.boxplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              **kwargs,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
        if kind == 'count':
          sns.countplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              **kwargs,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
        if kind == 'bar':
          sns.barplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              **kwargs,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
        if kind == 'scatter':
          sns.scatterplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              **kwargs,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
        if kind == 'line':
          sns.lineplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              **kwargs,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
        if kind == 'violin':
          sns.violinplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              **kwargs,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
      else:
        if kind == 'hist':
          sns.histplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
        if kind == 'box':
          sns.boxplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
        if kind == 'count':
          sns.countplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
        if kind == 'bar':
          sns.barplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
        if kind == 'scatter':
          sns.scatterplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
        if kind == 'line':
          sns.lineplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )
        if kind == 'violin':
          sns.violinplot(
              data=df,
              x=x,
              y=y,
              hue=hue,
              palette='Set2',
              ax=ax[curr_plot] if n_rows == 1 or n_cols == 1 else ax[row][col]
          )

      if n_rows == 1 or n_cols == 1:
        if kind in ['bar','count'] and 'bar_labels' in plot_info:
          ax[curr_plot].bar_label(ax[curr_plot].containers[0])

        if 'plot_titles' in plot_info:
          if len(plot_info['plot_titles']) == 1:
            text = plot_info['plot_titles'][0]
          else:
            text = plot_info['plot_titles'][curr_plot]
          ax[curr_plot].set_title(text)

        if 'plot_x_labels' in plot_info:
          if len(plot_info['plot_x_labels']) == 1:
            text = plot_info['plot_x_labels'][0]
          else:
            text = plot_info['plot_x_labels'][curr_plot]
          ax[curr_plot].set_xlabel(text)

        if 'plot_y_labels' in plot_info:
          if len(plot_info['plot_y_labels']) == 1:
            text = plot_info['plot_y_labels'][0]
          else:
            text = plot_info['plot_y_labels'][curr_plot]
          ax[curr_plot].set_ylabel(text)
      else:
        if kind in ['bar','count'] and 'bar_labels' in plot_info:
          ax[row][col].bar_label(ax[row][col].containers[0])

        if 'plot_titles' in plot_info:
          if len(plot_info['plot_titles']) == 1:
            text = plot_info['plot_titles'][0]
          else:
            text = plot_info['plot_titles'][curr_plot]
          ax[row][col].set_title(text)

        if 'plot_x_labels' in plot_info:
          if len(plot_info['plot_x_labels']) == 1:
            text = plot_info['plot_x_labels'][0]
          else:
            text = plot_info['plot_x_labels'][curr_plot]
          ax[row][col].set_xlabel(text)

        if 'plot_y_labels' in plot_info:
          if len(plot_info['plot_y_labels']) == 1:
            text = plot_info['plot_y_labels'][0]
          else:
            text = plot_info['plot_y_labels'][curr_plot]
          ax[row][col].set_ylabel(text)

      curr_plot += 1

  fig.tight_layout()

  plt.show()


plot_info = {}
plot_info['plot_kinds'] = ['box']
plot_info['plot_x'] = list(df.columns)
plot_info['plot_y'] = []
plot_info['plot_hue'] = []
plot_info['plot_kwargs'] = [{'width':0.3}]

multi_plots(
    df,
    plot_info,
    n_rows=3,
    n_cols=4,
    size=(22,8),
    n_plots=11
)"""