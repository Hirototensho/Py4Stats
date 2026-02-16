#!/usr/bin/env python
# coding: utf-8



from __future__ import annotations




import pandas as pd
import numpy as np
from py4etrics.heckit import Heckit

import statsmodels.formula.api as smf
import patsy

from py4stats import building_block as build # py4stats のプログラミングを補助する関数群
from py4stats import regression_tools as reg

from functools import singledispatch


# ## 型ヒントの準備



from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    Literal,
    overload,
    Protocol,
)

# --- statsmodels types ---
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.contrast import ContrastResults
from statsmodels.discrete.discrete_model import (
    BinaryResultsWrapper,
    PoissonResultsWrapper,
    NegativeBinomialResultsWrapper,
)

# --- matplotlib types ---
try:
    from matplotlib.axes import Axes
except Exception:  # notebooks etc.
    Axes = Any  # type: ignore

# --- handy type aliases ---
ConfLevel = float
Normalize = Union[bool, Literal["all", "index", "columns"]]
TableStyle = Literal["two_line", "one_line"]
StatsKey = Literal["std_err", "statistics", "p_value", "conf_int"]

# 限界効果（margeff）を返せるモデルの最小要件（型チェッカ向け）
class _HasMargEff(Protocol):
    def get_margeff(self, *args: Any, **kwargs: Any) -> Any: ...


# ### 回帰式を使った Heckit のインターフェース



def Heckit_from_formula(selection, outcome, data, **kwargs):
  # ステップ1：第１段階の説明変数
  y_select, exog_select = patsy.dmatrices(
      selection, data, return_type = 'dataframe',
      NA_action=patsy.NAAction(NA_types=[]) # 欠測値の除外を止める
      )

  # ステップ2：第２段階の説明変数と被説明変数
  endog, exog_outcome = patsy.dmatrices(
      outcome, data, return_type = 'dataframe',
      NA_action=patsy.NAAction(NA_types=[]) # 欠測値の除外を止める
      )

  endog_name = endog.columns.to_list()[0]

  model = Heckit(endog[endog_name], exog_outcome, exog_select, **kwargs)
  return model, exog_outcome, exog_select


# ## `HeckitResults` 用の `tidy()` メソッド



# regression_tools.tdy() にメソッドを後付けする実験的試み
from py4etrics.heckit import HeckitResults

@reg.tidy.register(HeckitResults)
def tidy_heckit(
    model,
    name_selection = None,
    name_outcome = None,
    conf_level = 0.95
    ):
  tidy_outcome = reg.tidy_regression(
      model,
      name_of_term = name_outcome,
      conf_level = conf_level
      )

  tidy_outcome.index = 'O: ' + tidy_outcome.index.to_series()

  tidy_select = reg.tidy_regression(
      model.select_res,
      name_of_term = name_selection,
      conf_level = conf_level
      )

  tidy_select.index = 'S: ' + tidy_select.index.to_series()

  res = pd.concat([tidy_outcome, tidy_select])
  return res




@reg.coefplot.register(HeckitResults)
def coefplot_Heckit(
    mod: Any,
    subset: Optional[Sequence[str]] = None,
    conf_level: Sequence[float] = (0.95, 0.99),
    palette: Sequence[str] = ("#1b69af", "#629CE7"),
    show_Intercept: bool = False,
    show_vline: bool = True,
    ax: Optional[Axes] = None,
    name_selection = None,
    name_outcome = None,
    **kwargs: Any,
) -> None:
    """Plot regression coefficients with multiple confidence intervals.

    This function extracts coefficient tables via `tidy` and draws a dot-and-line
    plot showing estimates and confidence intervals.

    Args:
        mod:
            A fitted model object supported by `tidy`.
        subset (Sequence[str] | None):
            Terms to include. If None, includes all terms.
        conf_level (Sequence[float]):
            Two confidence levels, e.g. (0.95, 0.99). The first is drawn thicker.
        palette (Sequence[str]):
            Two colors used for intervals and points.
        show_Intercept (bool):
            If False, drop the intercept term ('Intercept') from the plot.
        show_vline (bool):
            If True, draw a vertical reference line at 0.
        ax (matplotlib.axes.Axes | None):
            Axes to draw on. If None, a new figure/axes is created.
        **kwargs:
            Additional args forwarded to the low-level plotting function.

    Returns:
        None
    """
    build.assert_float(conf_level, lower = 0, upper = 1, inclusive = 'neither')
    build.assert_character(palette, arg_name = 'palette', len_arg = 2)

    # 回帰係数の表を抽出
    tidy_ci_high = reg.tidy(
        mod, conf_level = conf_level[0], 
        name_selection = name_selection,
        name_outcome = name_outcome
        )
    tidy_ci_row = reg.tidy(
        mod, conf_level = conf_level[1],
        name_selection = name_selection,
        name_outcome = name_outcome
        )

    # subset が指定されていれば、回帰係数の部分集合を抽出する
    if subset is not None:
        tidy_ci_high = tidy_ci_high.loc[subset, :]
        tidy_ci_row = tidy_ci_row.loc[subset, :]

    # グラフの作成
    reg.coef_dot(
        tidy_ci_high, tidy_ci_row, palette = palette,
        show_Intercept = show_Intercept, show_vline = show_vline,
        ax = ax, **kwargs
        )


# ### 限界効果を推定する関数



from scipy.stats import norm

def finv_mills(x): return norm.pdf(x) / norm.cdf(x)

# 適切なダミー変数かどうかを判定する関数
def is_dummy(x): return x.isin([0, 1]).all(axis = 'index') & (x.nunique() == 2)

def f_d_lambda(var_name, Z, gamma, beta_lambda):
  z1 = Z.mean().copy()
  z0 = z1.copy()
  z1[var_name] = 1
  z0[var_name] = 0
  res = beta_lambda * (finv_mills(z1 @ gamma) - finv_mills(z0 @ gamma))
  return res

def f_d_log_cdf(var_name, Z, gamma, beta_lambda):
  z1 = Z.mean().copy()
  z0 = z1.copy()
  z1[var_name] = 1
  z0[var_name] = 0
  res = (np.log(norm.cdf(z1 @ gamma)) - np.log(norm.cdf(z0 @ gamma)))
  return res




from py4etrics.heckit import HeckitResults

def heckitmfx_compute(
    model,
    exog_select,
    exog_outcome,
    exponentiate = False,
    params = None):

  assert isinstance(model, HeckitResults), \
    'model には HeckitResults クラスのオブジェクトを代入してください。'

  # 回帰係数の抽出 --------------
  if params is not None:
    # 回帰係数が指定された場合の処理（デルタ法の実装用）
    n_gamma = int(model.select_res.df_model + 1)
    beta = pd.Series(params[n_gamma:], exog_outcome.columns)
    gamma = pd.Series(params[:n_gamma], exog_select.columns)
  else:
    beta = pd.Series(model.params, exog_outcome.columns)
    gamma = pd.Series(model.select_res.params, exog_select.columns)

  beta.name = 'beta'
  gamma.name = 'gamma'

  # 逆ミルズ比の回帰係数 --------------
  beta_lambda = model.params_inverse_mills

  est = pd.merge(beta[1:], gamma[1:], how='outer', left_index=True, right_index=True)
  # 賃金関数で使われていない変数については beta に nan が発生するため、0で置換します。
  est[est.isna()] = 0

  # 連続変数用の処理--------------
  alpha = exog_select @ gamma

  lambda_value = finv_mills(alpha)

  delta = lambda_value * (lambda_value + alpha)
  selection = gamma * lambda_value.mean()
  ei_2 = gamma * beta_lambda * delta.mean()

  #  ダミー変数用の処理 --------------
  dummy_vars = is_dummy(exog_select)

  if(dummy_vars.sum() >= 1):

    d_lambda_val = [
        f_d_lambda(var_name, exog_select, gamma, beta_lambda)
        for var_name in dummy_vars[dummy_vars].index.to_list()
        ]

    d_log_cdf_val = [
        f_d_log_cdf(var_name, exog_select, gamma, beta_lambda)
        for var_name in dummy_vars[dummy_vars].index.to_list()
        ]

    ei_2[dummy_vars] = d_lambda_val
    selection[dummy_vars] = d_log_cdf_val
  # 限界効果の計算 ---------------------
  est['conditional'] = est['beta'] - ei_2
  est['selection'] = selection
  est['unconditional'] = est['conditional'] + est['selection']
  est = est.loc[:, ['unconditional', 'conditional', 'selection', 'beta', 'gamma']]

  if(exponentiate):
    est.loc[:, ['unconditional', 'conditional', 'selection', 'beta']] = \
    log_to_pct(est.loc[:, ['unconditional', 'conditional', 'selection', 'beta']])

  est.index.name  = 'term'
  return est




def log_to_pct(x): return 100 * (np.exp(x) - 1)


# ### デルタ法により限界効果の標準誤差を推定する関数



def jacobian(f, x, h=0.00001, *args):
    J = []
    x = np.array(x).astype(float)
    for i in range(len(x)):
        x1 = x.copy()
        x0 = x.copy()
        x1[i] = x[i] + h
        x0[i] = x[i] - h
        J.append((f(x1, *args) - f(x0, *args)) / (2 * h))
    return np.column_stack(J)
    return J




from scipy.stats import norm

def heckitmfx(
    model,
    exog_select,
    exog_outcome,
    type_estimate = 'unconditional',
    exponentiate = False,
    alpha = 0.05
    ):

  type_estimate = build.arg_match(
      type_estimate, arg_name = 'type_estimate',
       values = ['unconditional', 'conditional', 'selection']
      )

  # 限界効果の推定
  estimate = heckitmfx_compute(
      model, exog_select, exog_outcome,
      exponentiate = exponentiate
      ).loc[:, type_estimate]

  # 共分散行列の作成
  vcv1 = model.select_res.cov_params()
  vcv2 = model.cov_params()

  O = np.zeros(shape = (vcv1.shape[0], vcv2.shape[0]))

  vcv = np.block([[vcv1, O], [O.T, vcv2]])

  # ヤコブ行列の計算
  J_mat = jacobian(
      f = lambda x : heckitmfx_compute(
          model, exog_select, exog_outcome,
          params = x, exponentiate = exponentiate
          ).loc[:, type_estimate],
      x = np.append(model.select_res.params, model.params)
      )
  # デルタ法による標準誤差の推定
  std_err = np.sqrt(np.diag(J_mat @ vcv @ J_mat.T))

  # Z統計量の推定値を計算
  statistic = estimate / std_err
  z_alpha = norm.isf(alpha/2)

  # 結果の出力
  res = pd.DataFrame({
    'type':type_estimate,
    'estimate':estimate,
    'std_err':std_err,
    'statistic': statistic,
    'p_value': 2 * norm.sf(statistic.abs()), # 両側p-値
    'conf_lower': estimate - z_alpha * std_err,
    'conf_higher': estimate + z_alpha * std_err
    })

  return res

