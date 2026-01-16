#!/usr/bin/env python
# coding: utf-8



from __future__ import annotations


# # `reg_tools`：回帰分析の結果を要約する関数群

# `regression_tools` モジュールに実装された主要な関数の依存関係
# 
# ``` python
# compare_ols()                     # 複数のOLSモデルの係数を比較する表を作成
# ├─ tidy()                        # モデルから係数情報を抽出（RegressionResultsWrapper用）
# ├─ gazer()                       # 整形して見やすい形式に加工（係数＋統計量）
# ├─ lineup_models()              # モデル名を列にして横方向に整形
# └─ make_glance_tab()            # 各モデルの当てはまり指標（rsquared等）を整形
# 
# compare_mfx()                     # 複数のロジット系モデルの限界効果を比較
# ├─ tidy_mfx()                   # モデルから限界効果（dydxなど）を抽出
# │  └─ build.arg_match()         # 引数検証（method, at など）
# ├─ gazer()                      # 結果の整形
# ├─ lineup_models()             # モデル名を列にして整形
# └─ make_glance_tab()           # モデル当てはまり指標を整形（上と共通）
# 
# gazer()                           # tidyデータを人間が読みやすい文字列形式に整形
# ├─ build.arg_match()             # 統計量・表形式の引数検証
# ├─ build.p_stars()              # 有意性を表すアスタリスク付与
# └─ build.style_number()         # 数値整形
# 
# make_glance_tab()                 # glance情報（rsq, AICなど）を縦表に整形
# ├─ glance()                     # モデルごとの指標を抽出（singledispatch）
# └─ build.arg_match()            # stats_glance の妥当性チェック
# 
# lineup_models()                   # モデル比較表の列統合
# （依存なし）
# 
# tidy()                            # 結果を tidy形式のDataFrame に変換（dispatch対応）
# tidy_mfx()                        # 限界効果を tidy形式に変換
# 
# coefplot()                        # 回帰係数のエラーバーグラフ作成
# └─ coef_dot()                   # 実際のプロットを作成
# 
# mfxplot()                         # 限界効果のプロット
# └─ tidy_mfx(), coef_dot()
# 
# Blinder_Oaxaca()                  # Oaxaca-Blinder 分解
# └─ assert_reg_result()
# 
# plot_Blinder_Oaxaca()             # Oaxaca 分解の結果をプロット
# ├─ Blinder_Oaxaca()
# └─ build.arg_match()
# ```



# 依存するライブラリーの読込
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import t
from scipy.stats import f
from functools import singledispatch
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

import sys

from py4stats import building_block as build # py4stats のプログラミングを補助する関数群

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
# ConfLevel = float
Normalize = Union[bool, Literal["all", "index", "columns"]]
# TableStyle = Literal["two_line", "one_line"]
StatsKey = Literal["std_err", "statistics", "p_value", "conf_int"]

# # 限界効果（margeff）を返せるモデルの最小要件（型チェッカ向け）
# class _HasMargEff(Protocol):
#     def get_margeff(self, *args: Any, **kwargs: Any) -> Any: ...


# ## 回帰分析の結果をデータフレームに変換する関数



from functools import singledispatch

# definition of tidy --------------------------------------------------
@singledispatch
def tidy(
  x: Any, 
  name_of_term: Optional[Sequence[str]] = None, 
  conf_level: float = 0.95, 
  **kwargs: Any) -> pd.DataFrame:
  """Convert model/test results into a tidy (long) DataFrame.

  This function provides a unified interface to extract coefficient tables
  and test summaries into a standardized DataFrame. Concrete methods are
  registered via `functools.singledispatch` for supported object types.

  Supported types (in this module):
      - statsmodels.regression.linear_model.RegressionResultsWrapper
      - statsmodels.stats.contrast.ContrastResults

  Args:
      x:
          A statsmodels result object. The actual accepted type depends on
          registered implementations.
      name_of_term (Sequence[str] | None):
          Optional names for the coefficient terms. Forwarded to statsmodels
          helper utilities when supported.
      conf_level (float):
          Confidence level for intervals. Must be in (0, 1).
      **kwargs:
          Additional keyword arguments reserved for future extensions.

  Returns:
      pandas.DataFrame:
          Tidy coefficient/test table. The set of columns depends on the
          underlying result type, but commonly includes:
          `estimate`, `std_err`, `statistics`, `p_value`, `conf_lower`, `conf_higher`.

  Raises:
      NotImplementedError:
          If no `tidy` implementation is registered for the type of `x`.
  """
  raise NotImplementedError(f'tidy mtethod for object {type(x)} is not implemented.')




from statsmodels.iolib.summary import summary_params_frame
from statsmodels.regression.linear_model import RegressionResultsWrapper

@tidy.register(RegressionResultsWrapper)
@tidy.register(statsmodels.imputation.mice.MICEResults)
def tidy_regression(
    x: RegressionResultsWrapper,
    name_of_term: Optional[Sequence[str]] = None,
    conf_level: float = 0.95,
    add_one_sided: bool = False,
    to_jp: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Tidy a statsmodels OLS regression result into a DataFrame.

    This function extracts coefficient estimates, standard errors, t-statistics,
    p-values, and confidence intervals from a fitted regression result.

    Args:
        x (statsmodels.regression.linear_model.RegressionResultsWrapper):
            Fitted regression result (e.g., from `sm.OLS(...).fit()` or
            `smf.ols(...).fit()`).
        name_of_term (Sequence[str] | None):
            Optional term names for the coefficient table. Passed as `xname`
            to statsmodels summary utilities.
        conf_level (float):
            Confidence level for two-sided confidence intervals. Must be in (0, 1).
        add_one_sided (bool):
            If True, add an additional column `one_sided_p_value` computed from
            the t distribution with `df_resid`.
        to_jp (bool):
            If True, rename columns/index to Japanese labels using `tidy_to_jp`.
        **kwargs:
            Reserved for compatibility.

    Returns:
        pandas.DataFrame:
            Coefficient table indexed by term names (`term`) with columns:
            - estimate: coefficient estimate
            - std_err: standard error
            - statistics: t-statistic
            - p_value: two-sided p-value
            - conf_lower: lower bound of CI
            - conf_higher: upper bound of CI
            And optionally:
            - one_sided_p_value: one-sided p-value

    Raises:
        AssertionError:
            If `conf_level` is not in (0, 1).
    """
    # 引数のアサーション ----------------------------------------------------------------------------------
    build.assert_float(conf_level, lower = 0, upper = 1, inclusive = 'neither', arg_name = 'conf_level')
    build.assert_logical(add_one_sided, arg_name = 'add_one_sided')
    # --------------------------------------------------------------------------------------------------
    tidied = summary_params_frame(x, alpha = 1 - conf_level, xname = name_of_term)

    tidied.index.name = 'term'

    rename_cols = {
        'coef':'estimate',
        'std err':'std_err',
        't':'statistics', 'P>|t|': 'p_value',
        'Conf. Int. Low': 'conf_lower',
        'Conf. Int. Upp.': 'conf_higher'
    }

    tidied = tidied.rename(columns = rename_cols)

    if add_one_sided:
        tidied = add_one_sided_p_value(x, tidied)

    # 列名を日本語に変換
    if to_jp:
        tidied = tidy_to_jp(tidied, conf_level = 0.95)

    return tidied




from statsmodels.stats.contrast import ContrastResults

@tidy.register(ContrastResults)
def tidy_test(
    x: ContrastResults,
    conf_level: float = 0.95,
    **kwargs: Any,
) -> pd.DataFrame:
  """Tidy a statsmodels ContrastResults into a DataFrame.

  This function converts hypothesis test results (t-test / F-test / Wald-type)
  into a standardized DataFrame.

  Behavior depends on `x.distribution`:
      - If distribution is 'F': returns a one-row table including df.
      - Otherwise: uses `x.summary_frame(...)` and renames common columns.

  Args:
      x (statsmodels.stats.contrast.ContrastResults):
          Contrast test result returned by statsmodels methods such as
          `t_test`, `f_test`, or similar APIs.
      conf_level (float):
          Confidence level for intervals. Must be in (0, 1).
      **kwargs:
          Reserved for compatibility.

  Returns:
      pandas.DataFrame:
          Tidy test summary indexed by `term` (usually 'contrast').

          For F distribution:
              Columns include `statistics`, `p_value`, `df_denom`, `df_num`.

          For other distributions:
              Columns typically include `estimate`, `std_err`, `statistics`,
              `p_value`, `conf_lower`, `conf_higher`.

  Raises:
      AssertionError:
          If `conf_level` is not in (0, 1).
  """
  build.assert_float(conf_level, lower = 0, upper = 1, inclusive = 'neither', arg_name = 'conf_level')

  if(x.distribution == 'F'):
    tidied = pd.DataFrame({
    'statistics':x.statistic,
    'p_value':x.pvalue,
    'df_denom':int(x.df_denom),
    'df_num':int(x.df_num)
  }, index = ['contrast'])

  else:
    tidied = x.summary_frame(alpha = 1 - conf_level)

    rename_cols = {
        'coef':'estimate',
        'std err':'std_err',
        't':'statistics', 'P>|t|': 'p_value',
        'Conf. Int. Low': 'conf_lower',
        'Conf. Int. Upp.': 'conf_higher'
    }

    tidied = tidied.rename(columns = rename_cols)

  tidied.index.name = 'term'
  return tidied


# ### 片側t-検定



from scipy.stats import t
from scipy.stats import norm

# definition of tidy --------------------------------------------------
@singledispatch
def tidy_one_sided(x: Any, conf_level: float = 0.95, **kwargs: Any) -> pd.DataFrame:
  """Compute one-sided test summaries in a tidy DataFrame.

  This is a specialized variant of `tidy` that recomputes p-values and
  confidence intervals for one-sided tests.

  Supported types (in this module):
      - statsmodels.stats.contrast.ContrastResults
      - statsmodels.regression.linear_model.RegressionResultsWrapper

  Args:
      x:
          A supported statsmodels result object.
      conf_level (float):
          One-sided confidence level. Must be in (0, 1).
      **kwargs:
          Reserved for compatibility.

  Returns:
      pandas.DataFrame:
          Tidy table similar to `tidy(x)` but with one-sided `p_value` and
          recalculated confidence bounds.

  Raises:
      NotImplementedError:
          If the type of `x` is not supported.
  """
  raise NotImplementedError(f'tidy mtethod for object {type(x)} is not implemented.')




@tidy_one_sided.register(ContrastResults)
def tidy_one_sided_t_test(x: ContrastResults, conf_level: float = 0.95) -> pd.DataFrame:
  """Compute one-sided p-values and confidence bounds for a ContrastResults test.

  The function calls `tidy(x)` first and then overwrites:
      - p_value
      - conf_lower
      - conf_higher

  based on the distribution used in the test:
      - 't': uses Student t distribution
      - 'norm': uses standard normal distribution

  Args:
      x (statsmodels.stats.contrast.ContrastResults):
          Contrast test result.
      conf_level (float):
          One-sided confidence level. Must be in (0, 1).

  Returns:
      pandas.DataFrame:
          Updated tidy table with one-sided inference results.

  Raises:
      NotImplementedError:
          If `x.distribution` is not supported (not 't' or 'norm').
  """
  build.assert_float(conf_level, lower = 0, upper = 1, inclusive = 'neither', arg_name = 'conf_level')
  tidied = tidy(x)

  # 仮説検定にt分布が用いられている場合
  if(x.distribution == 't'):
    tidied['p_value'] = t.sf(abs(tidied['statistics']), x.dist_args[0])
    t_alpha = t.isf(1 - conf_level, df = x.dist_args[0])
    tidied['conf_lower'] = tidied['estimate'] - t_alpha * tidied['std_err']
    tidied['conf_higher'] = tidied['estimate'] + t_alpha * tidied['std_err']

  # 仮説検定に正規分布が用いられている場合
  elif(x.distribution == 'norm'):
    tidied['p_value'] = norm.sf(abs(tidied['statistics']))
    z_alpha = norm.isf(1 - conf_level)
    tidied['conf_lower'] = tidied['estimate'] - z_alpha * tidied['std_err']
    tidied['conf_higher'] = tidied['estimate'] + z_alpha * tidied['std_err']
  else:
    raise NotImplementedError(f'tidy mtethod for distribution {x.distribution} is not implemented.')

  return tidied




@tidy_one_sided.register(RegressionResultsWrapper)
def tidy_one_sided_regression(
    x: RegressionResultsWrapper,
    conf_level: float = 0.95,
    null_hypotheses: Union[int, float] = 0,
) -> pd.DataFrame:
    """Compute one-sided inference for regression coefficients.

    This function starts from `tidy(x)` and then recomputes:
        - H_null: null hypothesis value for each coefficient
        - statistics: (estimate - H_null) / std_err
        - p_value: one-sided p-values (t or normal depending on model)
        - conf_lower / conf_higher: one-sided confidence bounds

    Args:
        x (statsmodels.regression.linear_model.RegressionResultsWrapper):
            Fitted regression result.
        conf_level (float):
            One-sided confidence level. Must be in (0, 1).
        null_hypotheses (float | int):
            Null hypothesis value used for all coefficients.

    Returns:
        pandas.DataFrame:
            Tidy coefficient table with one-sided p-values and confidence bounds.

    Raises:
        AssertionError:
            If `conf_level` is not in (0, 1) or `null_hypotheses` is not numeric.
    """
    # 引数のアサーション ----------------------------------------------------------------------------------
    build.assert_float(conf_level, lower = 0, upper = 1, inclusive = 'neither', arg_name = 'conf_level')
    build.assert_numeric(null_hypotheses, arg_name = 'null_hypotheses')
    # --------------------------------------------------------------------------------------------------


    tidied = tidy(x)

    tidied['H_null'] = null_hypotheses

    tidied['statistics'] = (tidied['estimate'] - tidied['H_null']) / tidied['std_err']
    # 仮説検定にt分布が用いられている場合
    if(x.use_t):
        tidied['p_value'] = t.sf(abs(tidied['statistics']), x.df_resid)
        t_alpha = t.isf(1 - conf_level, df = x.df_resid)
        tidied['conf_lower'] = tidied['estimate'] - t_alpha * tidied['std_err']
        tidied['conf_higher'] = tidied['estimate'] + t_alpha * tidied['std_err']
    # 仮説検定に正規分布が用いられている場合
    else:
        tidied['p_value'] = norm.sf(abs(tidied['statistics']))
        z_alpha = norm.isf(1 - conf_level)
        tidied['conf_lower'] = tidied['estimate'] - z_alpha * tidied['std_err']
        tidied['conf_higher'] = tidied['estimate'] + z_alpha * tidied['std_err']

    return tidied




from scipy.stats import t
def tidy_to_jp(tidied: pd.DataFrame, conf_level: float = 0.95) -> pd.DataFrame:
  """Rename tidy regression columns to Japanese labels.

  Args:
      tidied (pandas.DataFrame):
          A tidy coefficient table produced by `tidy(...)`.
      conf_level (float):
          Confidence level used to label confidence interval columns.

  Returns:
      pandas.DataFrame:
          A renamed DataFrame with Japanese column/index labels.
  """
  tidied = tidied\
      .rename(columns = {
          'term':'説明変数',
          'estimate':'回帰係数', 'std_err':'標準誤差',
          'statistics':'t-値', 'p_value':'p-値',
          'conf_lower': str(int(conf_level*100)) + '%信頼区間下側',
          'conf_higher': str(int(conf_level*100)) + '%信頼区間上側',
          'one_sided_p_value':'片側p-値'
          })

  tidied.index.name = '説明変数'

  return tidied

def add_one_sided_p_value(x: RegressionResultsWrapper, tidied: pd.DataFrame) -> pd.DataFrame:
      """Add one-sided p-values column to a tidy regression table.

      Args:
          x (statsmodels.regression.linear_model.RegressionResultsWrapper):
              Fitted regression result. Used for degrees of freedom (`df_resid`).
          tidied (pandas.DataFrame):
              Tidy coefficient table that contains `statistics`.

      Returns:
          pandas.DataFrame:
              Copy of `tidied` with an additional column `one_sided_p_value`.
      """
      tidied = tidied.copy()
      tidied['one_sided_p_value'] = t.sf(abs(tidied['statistics']), x.df_resid)
      return tidied


# `glance()`



from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.discrete.discrete_model import BinaryResultsWrapper, PoissonResultsWrapper, NegativeBinomialResultsWrapper
from functools import singledispatch

@singledispatch
def glance(x: Any) -> pd.DataFrame:
    """Return a one-row model fit summary.

    `glance` is inspired by the R package `broom::glance`, returning fit-level
    statistics (not coefficient-level) in a standardized one-row DataFrame.

    Supported types (in this module):
        - statsmodels discrete model results: Binary/Poisson/NegativeBinomial
        - statsmodels OLS regression: RegressionResultsWrapper

    Args:
        x:
            A supported statsmodels result object.

    Returns:
        pandas.DataFrame:
            One-row DataFrame of model-level fit statistics.

    Raises:
        NotImplementedError:
            If the type of `x` is not supported.
    """
    raise NotImplementedError(f'glance mtethod for object {type(x)} is not implemented.')




# 一般化線型モデル用のメソッド
@glance.register(BinaryResultsWrapper)
@glance.register(PoissonResultsWrapper)
@glance.register(NegativeBinomialResultsWrapper)
def glance_glm(
    x: Union[BinaryResultsWrapper, PoissonResultsWrapper, NegativeBinomialResultsWrapper]
) -> pd.DataFrame:
  """Compute fit statistics for a statsmodels GLM-type model result.

    This method is registered for:
        - BinaryResultsWrapper
        - PoissonResultsWrapper
        - NegativeBinomialResultsWrapper

    Args:
        x:
            Fitted statsmodels GLM-family result object.

    Returns:
        pandas.DataFrame:
            One-row DataFrame with columns such as:
            - prsquared, LL-Null, df_null, logLik, AIC, BIC, deviance,
                nobs, df, df_resid
  """
  res = pd.DataFrame({
      'prsquared':x.prsquared,
      'LL-Null':x.llnull ,
      'df_null':x.nobs - 1,
      'logLik':x.llf,
      'AIC':x.aic,
      'BIC':x.bic,
      'deviance':-2*x.llf,
      'nobs':x.nobs,
      'df': int(x.df_model),
      'df_resid':int(x.df_resid)
  }, index = [0])
  return res




# 線形回帰用のメソッド
@glance.register(RegressionResultsWrapper)
def glance_ols(x: RegressionResultsWrapper) -> pd.DataFrame:
    """Compute fit statistics for a statsmodels OLS regression result.

    Args:
        x (statsmodels.regression.linear_model.RegressionResultsWrapper):
            Fitted OLS regression result.

    Returns:
        pandas.DataFrame:
            One-row DataFrame with columns such as:
            - rsquared, rsquared_adj, nobs, df, sigma, F_values, p_values, AIC, BIC
    """
    res = pd.DataFrame({
        'rsquared':x.rsquared,
        'rsquared_adj':x.rsquared_adj,
        'nobs':int(x.nobs),
        'df':int(x.df_model),
        'sigma':np.sqrt(x.mse_resid),
        'F_values':x.fvalue,
        'p_values':x.f_pvalue,
        'AIC':x.aic,
        'BIC':x.bic
    }, index = [0])
    return res




def log_to_pct(est: Union[float, pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
    """Convert log-scale estimates to percent change.

    Args:
        est (float or array-like):
            Log-scale estimate(s).

    Returns:
        Same type as input:
            Percent change computed as `100 * (exp(est) - 1)`.
    """
    return 100 * (np.exp(est) - 1)


# ## `reg.compare_ols()`
# 
# ### 概要
# 
# 　`reg.compare_ols()` は計量経済学の実証論文でよく用いられる、回帰分析の結果を縦方向に並べて比較する表をする関数です。
# 　使用方法は次の通りで、`sm.ols()` や `smf.ols()` で作成した分析結果のオブジェクトのリストを代入します。  
# 
# ```python
# penguins = load_penguins() # サンプルデータの読み込み
# 
# fit1 = smf.ols('body_mass_g ~ bill_length_mm + species', data = penguins).fit()
# fit2 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species', data = penguins).fit()
# fit3 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species + sex', data = penguins).fit()
# 
# compare_tab1 = reg.compare_ols([fit1, fit2, fit3])
# compare_tab1
# ```



from statsmodels.regression.linear_model import RegressionResultsWrapper
import varname

def assert_reg_reuslt(x: Any) -> None:
  """Assert that inputs are statsmodels RegressionResultsWrapper objects.

  Args:
      x:
          A list-like object expected to contain `RegressionResultsWrapper`.

  Raises:
      AssertionError:
          If any element is not a `RegressionResultsWrapper`.
  """
  x = pd.Series(x)
  condition =  x.apply(lambda x: isinstance(x, (RegressionResultsWrapper))).all()
  assert condition, f"Argument '{varname.argname('x')}' must be of type '{RegressionResultsWrapper}'."




import pandas.api.types

def compare_ols(
    list_models: Sequence[RegressionResultsWrapper],
    model_name: Optional[Sequence[str]] = None,
    subset: Optional[Sequence[str]] = None,
    stats: StatsKey = "std_err",
    add_stars: bool = True,
    stats_glance: Optional[Sequence[str]] = ("rsquared_adj", "nobs", "df"),
    digits: int = 4,
    table_style: Literal["two_line", "one_line"] = "two_line",
    line_break: str = "\n",
    **kwargs: Any,
) -> pd.DataFrame:
    """Create a side-by-side comparison table for multiple regression models.

        This function is similar in spirit to regression tables commonly used in
        empirical economics papers: coefficients are aligned vertically and models
        are placed in columns.

        Steps:
            1. Convert each model to a tidy coefficient table via `tidy`.
            2. Format coefficients/statistics with `gazer`.
            3. Align model outputs with `lineup_models`.
            4. Optionally append fit statistics via `make_glance_tab`.

        Args:
            list_models (Sequence[RegressionResultsWrapper]):
                List of fitted regression results (e.g., from `.fit()`).
            model_name (Sequence[str] | None):
                Display names for models. If None, auto-generated as
                `["model 1", "model 2", ...]`.
            subset (Sequence[str] | None):
                If provided, restrict output to these terms (rows).
            stats (str):
                Statistic to show in parentheses. One of:
                - 'std_err', 'statistics', 'p_value', 'conf_int'
            add_stars (bool):
                If True, append significance stars based on p-values.
            stats_glance (Sequence[str] | None):
                Fit statistics to append at the bottom. If None, do not append.
            digits (int):
                Number of decimal places for numeric formatting.
            table_style (str):
                Display style of estimate/statistics:
                - 'two_line': estimate and stats on separate lines
                - 'one_line': estimate and stats on the same line
            line_break (str):
                Line break string used when `table_style='two_line'`.
            **kwargs:
                Additional keyword arguments forwarded to helper functions.

        Returns:
            pandas.DataFrame:
                Comparison table indexed by term names.
    """
    # 引数のアサーション ----------------------------------------------------------------------------------
    if model_name is not None:
        build.assert_character(model_name, arg_name = 'model_name')

    build.assert_count(digits, arg_name = 'digits')
    build.assert_logical(add_stars, arg_name = 'add_stars')
    build.assert_character(line_break, arg_name = 'line_break')
    # --------------------------------------------------------------------------------------------------
    assert pandas.api.types.is_list_like(list_models), "argument 'list_models' is must be a list of models."
    assert_reg_reuslt(list_models)

    tidy_list = [tidy(mod) for mod in list_models]

    # モデル名が指定されていない場合、連番を作成する
    if model_name is None:
        model_name = [f'model {i + 1}' for i in range(len(tidy_list))]

    # tidy_list の各要素に gazer() 関数を適用
    gazer_list = [gazer(
        df, digits = digits, stats = stats, add_stars = add_stars,
        table_style = table_style, line_break = line_break,
        **kwargs
        ) for df in tidy_list]

    # lineup_models() を適用してモデルを比較する表を作成
    res = lineup_models(
            gazer_list, model_name = model_name,
            subset = subset, **kwargs
        )
    res.index.name = 'term'
    # 表の下部にモデルの当てはまりに関する統計値を追加
    if stats_glance is not None: # もし stats_glance が None なら統計値を追加しない
        res2 = make_glance_tab(
            list_models,
            model_name = model_name,
            stats_glance = stats_glance,
            digits = digits
            )
        res = pd.concat([res, res2])

    return res




# 複数のモデルを比較する表を作成する関数 対象を sm.ols() に限定しないバージョン
def lineup_models(
    gazer_list: Sequence[pd.DataFrame],
    model_name: Optional[Sequence[str]] = None,
    subset: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Align formatted coefficient tables from multiple models.

    Args:
        gazer_list (Sequence[pandas.DataFrame]):
            List of formatted tables returned by `gazer`, each with a single
            column `value`.
        model_name (Sequence[str] | None):
            Names for each model column. If None, auto-generated.
        subset (Sequence[str] | None):
            Optional list of terms to keep.
        **kwargs:
            Reserved for compatibility.

    Returns:
        pandas.DataFrame:
            Wide table with one column per model and one row per term.
    """
    # モデル名が指定されていない場合、連番を作成する
    if model_name is None:
        model_name = [f'model {i + 1}' for i in range(len(gazer_list))]

    # model_name が列名になるように、辞書の key に設定してから pd.concat() で結合
    res = pd.concat(dict(zip(model_name, gazer_list)), axis = 'columns')\
        .droplevel(1, axis = 'columns') # 列名が2重に設定されるので、これを削除して1つにします。

    # subset が指定された場合は該当する変数を抽出します。
    if subset is not None:
        build.assert_character(subset, arg_name = 'subset')
        res = res.loc[subset, :]

    # モデルで使用されていない変数について NaN が発生するので、空白で置き換えます。
    res = res.fillna('')

    return res




# 回帰係数と検定統計量を縦に並べる関数
# 2024年1月30日変更 引数 stats と table_style について
# 妥当な値が指定されているかを検証する機能を追加しました。
# 2024年3月18日変更 数値の体裁を整える処理を build.style_number() を使ったものに変更しました。
def gazer(
    res_tidy: pd.DataFrame,
    estimate: str = "estimate",
    stats: StatsKey = "std_err",
    digits: int = 4,
    add_stars: bool = True,
    # p_min: float = 0.01,
    table_style: str = "two_line",  # match_arg が部分一致なので Literal にしない方が安全
    line_break: str = "\n",
    **kwargs: Any,
) -> pd.DataFrame:
    """Format coefficient estimates and statistics into a display-ready table.

    The returned table has a single column `value`, which typically contains
    strings like:
        - two_line: "1.234\\n(0.123)***"
        - one_line: "1.234(0.123)***"
    or confidence interval style:
        - "1.234\\n[0.100, 2.000]***"

    Args:
        res_tidy (pandas.DataFrame):
            Tidy coefficient table produced by `tidy` or similar functions.
        estimate (str):
            Column name for estimates (default: 'estimate').
        stats (str):
            Statistic to display. One of:
            - 'std_err', 'statistics', 'p_value', 'conf_int'
        digits (int):
            Number of decimal places for formatting.
        add_stars (bool):
            If True, append significance stars based on `p_value`.
        table_style (str):
            Formatting style. Partial matching may be allowed by `build.match_arg`.
        line_break (str):
            Line break to insert when `table_style='two_line'`.
        **kwargs:
            Reserved for compatibility.

    Returns:
        pandas.DataFrame:
            A one-column DataFrame with formatted strings, column name `value`.
    """
    # 引数に妥当な値が指定されているかを検証
    stats = build.arg_match(
        stats, ['std_err', 'statistics', 'p_value', 'conf_int'],
        arg_name = 'stats'
        )
    # こちらは部分一致可としています。
    table_style = build.match_arg(
        table_style, ['two_line', 'one_line'],
        arg_name = 'table_style'
        )
    build.assert_logical(add_stars, arg_name = 'add_stars')
    build.assert_count(digits, arg_name = 'digits')
    # --------------------
    res = res_tidy.copy()
    # 有意性を表すアスタリスクを作成します
    res['stars'] = ' ' + build.p_stars(res['p_value'])

    # # `estimate` と `stats` を見やすいフォーマットに変換します。
    # res[[estimate, stats]] = res[[estimate, stats]]\
    #     .apply(build.style_number, digits = digits)

    # table_style に応じて改行とアスタリスクを追加する

    if(table_style == 'two_line'):
        sep = line_break
        if add_stars:
            sep = res['stars'] + sep
        sufix = ''

    elif(table_style == 'one_line'):
        sep = ''
        if add_stars:
            sufix = res['stars']
        else:
            sufix = ''

    if(stats == 'conf_int'):
      res[[estimate, 'conf_lower', 'conf_higher']] \
        = res[[estimate, 'conf_lower', 'conf_higher']]\
          .apply(build.style_number, digits = digits)

      res['value'] =  res[estimate] + sep\
       + '[' + res['conf_lower'] + ', ' + res['conf_higher'] + ']'\
       + sufix
    else:
      # `estimate` と `stats` を見やすいフォーマットに変換します。
      res[[estimate, stats]] = res[[estimate, stats]]\
        .apply(build.style_number, digits = digits)
      res['value'] = res[estimate] + sep + '(' + res[stats] + ')' + sufix

    # モデルで使用されていない変数について NaN が発生するので、空白で置き換えます。
    res = res.fillna('')

    return res[['value']]




def make_glance_tab(
    list_models: Sequence[Any],
    model_name: Optional[Sequence[str]] = None,
    stats_glance: Sequence[str] = ("rsquared_adj", "nobs", "df"),
    digits: int = 4,
    **kwargs: Any,
) -> pd.DataFrame:
    """Create a fit-statistics table appended to `compare_ols` output.

    Args:
        list_models (Sequence[Any]):
            List of fitted model results supported by `glance`.
        model_name (Sequence[str] | None):
            Names for each model column. If None, auto-generated.
        stats_glance (Sequence[str]):
            Which fit statistics to include. Must be present in at least one
            model's `glance` output.
        digits (int):
            Number of decimals for rounding/formatting.
        **kwargs:
            Reserved for compatibility.

    Returns:
        pandas.DataFrame:
            Table of fit statistics with rows as statistics and columns as models.
    """
    # 引数に妥当な値が指定されているかを検証
    build.assert_count(digits, arg_name = 'digits')
    # --------------------
    # モデル名が指定されていない場合、連番を作成する
    if model_name is None:
        model_name = [f'model {i + 1}' for i in range(len(list_models))]

    glance_list = [glance(mod) for mod in list_models]

    # glance_list 内のデータフレームの列名の和集合を取得
    # つまり、代入されたどのモデルの、当てはまりの指標にもない名前を指定することはできないという処理
    union_set = glance_list[0].columns
    for i in range(1, len(glance_list)):
        union_set = union_set.union(glance_list[i].columns)

    # 引数に妥当な値が指定されているかを検証
    stats_glance = build.arg_match(
                stats_glance,
                values = union_set.to_list(),
                arg_name = 'stats_glance',
                multiple = True
                )

    res = pd.concat(glance_list)\
        .loc[:, stats_glance]\
        .round(digits)\
        .apply(build.pad_zero, digits = digits).T

    res.columns = model_name
    res[res == 'nan'] = ''
    res.index.name = 'term'
    return res


# ### `gazer()` 関数の多項ロジットモデルバージョン

# ## 回帰係数の視覚化関数
# 



# 利用するライブラリー
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.iolib.summary import summary_params_frame

from functools import singledispatch

# 回帰分析の結果から回帰係数のグラフを作成する関数 --------

@singledispatch
def coefplot(
    mod: Any,
    subset: Optional[Sequence[str]] = None,
    conf_level: Sequence[float] = (0.95, 0.99),
    palette: Sequence[str] = ("#1b69af", "#629CE7"),
    show_Intercept: bool = False,
    show_vline: bool = True,
    ax: Optional[Axes] = None,
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
  raise NotImplementedError(f'tidy mtethod for object {type(mod)} is not implemented.')




from statsmodels.regression.linear_model import RegressionResultsWrapper
@coefplot.register(RegressionResultsWrapper)
def coefplot_regression(
    mod: Any,
    subset: Optional[Sequence[str]] = None,
    conf_level: Sequence[float] = (0.95, 0.99),
    palette: Sequence[str] = ("#1b69af", "#629CE7"),
    show_Intercept: bool = False,
    show_vline: bool = True,
    ax: Optional[Axes] = None,
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
    build.assert_float(conf_level, lower = 0, upper = 1, inclusive = 'neither', arg_name = 'conf_level')
    build.assert_character(palette, arg_name = 'palette')

    # 回帰係数の表を抽出
    tidy_ci_high = tidy(mod, conf_level = conf_level[0])
    tidy_ci_row = tidy(mod, conf_level = conf_level[1])

    # subset が指定されていれば、回帰係数の部分集合を抽出する
    if subset is not None:
        tidy_ci_high = tidy_ci_high.loc[subset, :]
        tidy_ci_row = tidy_ci_row.loc[subset, :]

    # グラフの作成
    coef_dot(
        tidy_ci_high, tidy_ci_row, palette = palette,
        show_Intercept = show_Intercept, show_vline = show_vline,
        ax = ax, **kwargs
        )




def coef_dot(
    tidy_ci_high: pd.DataFrame,
    tidy_ci_low: pd.DataFrame,
    ax: Optional[Axes] = None,
    show_Intercept: bool = False,
    show_vline: bool = True,
    palette: Sequence[str] = ("#1b69af", "#629CE7"),
    estimate: str = "estimate",
    conf_lower: str = "conf_lower",
    conf_higher: str = "conf_higher",
) -> None:
    """Draw coefficient dots and confidence intervals from tidy tables.

    Args:
        tidy_ci_high (pandas.DataFrame):
            Tidy table for the higher confidence level (typically narrower line).
        tidy_ci_low (pandas.DataFrame):
            Tidy table for the lower confidence level (typically wider line).
        ax (matplotlib.axes.Axes | None):
            Axes to draw on. If None, a new figure/axes is created.
        show_Intercept (bool):
            If False, remove 'Intercept' before plotting.
        show_vline (bool):
            If True, draw a vertical reference line at 0.
        palette (Sequence[str]):
            Colors for high/low interval lines and points.
        estimate (str):
            Column name containing point estimates.
        conf_lower (str):
            Column name containing lower bounds.
        conf_higher (str):
            Column name containing upper bounds.

    Returns:
        None
    """
    tidy_ci_high = tidy_ci_high.copy()
    tidy_ci_low = tidy_ci_low.copy()

    # 切片項を除外する
    if not show_Intercept:
        tidy_ci_high = tidy_ci_high.loc[~ tidy_ci_high.index.isin(['Intercept']), :]
        tidy_ci_low = tidy_ci_low.loc[~ tidy_ci_low.index.isin(['Intercept']), :]


    if ax is None:
        fig, ax = plt.subplots()

    # 図の描画 -----------------------------
    # 垂直線の描画
    if show_vline:
        ax.axvline(0, ls = "--", color = '#969696')

    # エラーバーの作図
    ax.hlines(
        y = tidy_ci_low.index, xmin = tidy_ci_low[conf_lower], xmax = tidy_ci_low[conf_higher],
        linewidth = 1.5,
        color = palette[1]
    )
    ax.hlines(
        y = tidy_ci_high.index, xmin = tidy_ci_high[conf_lower], xmax = tidy_ci_high[conf_higher],
        linewidth = 3,
        color = palette[0]
    )

    # 回帰係数の推定値を表す点の作図
    ax.scatter(
      x = tidy_ci_high[estimate],
      y = tidy_ci_high.index,
      c = palette[0],
      s = 60
    )
    ax.set_ylabel('');


# ## `reg.compare_mfx()`
# 



def tidy_mfx(
    x: _HasMargEff,
    at: MfxAt = "overall",
    method: Literal['coef', 'dydx', 'eyex', 'dyex', 'eydx'] = "dydx",
    dummy: bool = False,
    conf_level: float = 0.95,
    **kwargs: Any,
) -> pd.DataFrame:
  """Tidy marginal effects output from a fitted model.

  This function calls `x.get_margeff(...)` to estimate marginal effects and
  converts the result into a tidy DataFrame with standardized column names.

  Args:
      x:
          A fitted model result object that implements `get_margeff`.
      at (str):
          Evaluation point for marginal effects. One of:
          - 'overall', 'mean', 'median', 'zero'
      method (str):
          Type of marginal effect. One of:
          - 'coef', 'dydx', 'eyex', 'dyex', 'eydx'
      dummy (bool):
          Whether to treat binary variables as discrete changes.
      conf_level (float):
          Confidence level for confidence intervals. Must be in (0, 1).
      **kwargs:
          Additional arguments forwarded to `get_margeff`.

  Returns:
      pandas.DataFrame:
          Tidy marginal effects table with columns:
          - estimate, std_err, statistics, p_value, conf_lower, conf_higher

  Raises:
      AssertionError:
          If `conf_level` is not in (0, 1).
  """
  # 引数に妥当な値が指定されているかを検証
  build.assert_float(conf_level, lower = 0, upper = 1, inclusive = 'neither', arg_name = 'conf_level')
  at = build.arg_match(at, ['overall', 'mean', 'median', 'zero'], arg_name = 'at')

  method = build.arg_match(
      method,
      values = ['coef', 'dydx', 'eyex', 'dyex', 'eydx'],
      arg_name = 'method'
      )
  # 限界効果の推定
  est_margeff = x.get_margeff(dummy = dummy, at = at, method = method, **kwargs)
  tab = est_margeff.summary_frame()

  method_dict = {
            'coef':'coef',
            'dydx':'dy/dx',
            'eyex':'d(lny)/d(lnx)',
            'dyex':'dy/d(lnx)',
            'eydx':'d(lny)/dx',
        }

  tab = tab.rename(columns = {
            method_dict[method]:'estimate',
            'Std. Err.':'std_err',
            'z':'statistics',
            'Pr(>|z|)':'p_value',
            'Conf. Int. Low':'conf_lower',
            'Cont. Int. Hi.':'conf_higher'
            })

  # conf_level に 0.95 以外の値が指定されていた場合は、信頼区間を個別に推定して値を書き換えます。
  if(conf_level != 0.95):
    CI = est_margeff.conf_int(alpha = 1 - conf_level)
    tab['conf_lower'] = CI[:, 0]
    tab['conf_higher'] = CI[:, 1]

  return tab




# 複数のロジットモデルを比較する表を作成する関数
def compare_mfx(
    list_models: Sequence[RegressionResultsWrapper],
    model_name: Optional[Sequence[str]] = None,
    subset: Optional[Sequence[str]] = None,
    stats: StatsKey = "std_err",
    add_stars: bool = True,
    stats_glance: Optional[Sequence[str]] = ("prsquared", "nobs", "df"),
    at: MfxAt = "overall",
    method: Literal['coef', 'dydx', 'eyex', 'dyex', 'eydx'] = "dydx",
    dummy: bool = False,
    digits: int = 4,
    table_style: Literal["two_line", "one_line"] = "two_line",
    line_break: str = "\n",
    **kwargs: Any,
) -> pd.DataFrame:
        """Create a comparison table for marginal effects across models.

        This function is analogous to `compare_ols`, but compares either:
            - coefficients (when method='coef'), or
            - marginal effects (otherwise) computed via `tidy_mfx`.

        Args:
            list_models:
                List of fitted models.
            model_name (Sequence[str] | None):
                Names for model columns. If None, auto-generated.
            subset (Sequence[str] | None):
                Terms to include. If None, includes all.
            stats (str):
                Statistic to show in parentheses. One of:
                - 'std_err', 'statistics', 'p_value', 'conf_int'
            add_stars (bool):
                Whether to append significance stars.
            stats_glance (Sequence[str] | None):
                Fit statistics to append at the bottom. If None, do not append.
            at (str):
                Evaluation point for marginal effects.
            method (str):
                Marginal effect type (see `tidy_mfx`).
            dummy (bool):
                Whether to treat binary variables as discrete changes.
            digits (int):
                Number of decimals for formatting.
            table_style (str):
                Output style ('two_line' or 'one_line').
            line_break (str):
                Line break used in two-line style.
            **kwargs:
                Additional keyword args forwarded to helper functions.

        Returns:
            pandas.DataFrame:
                Comparison table of marginal effects/coefficients.
        """
        assert pandas.api.types.is_list_like(list_models), "argument 'list_models' is must be a list of models."
        assert_reg_reuslt(list_models)
        # 限界効果の推定-------------
        if method == 'coef':
            tidy_list = [tidy(mod) for mod in list_models]
        else:
            tidy_list = [
                tidy_mfx(mod, at = at, method = method, dummy = dummy)
                for mod in list_models
                ]

        # モデル名が指定されていない場合、連番を作成する
        if model_name is None:
            model_name = [f'model {i + 1}' for i in range(len(tidy_list))]

            # tidy_list の各要素に gazer() 関数を適用
        gazer_list = [gazer(
            df, estimate = 'estimate',
            digits = digits, stats = stats, add_stars = add_stars,
            table_style = table_style, line_break = line_break,
            **kwargs
            ) for df in tidy_list]

        # lineup_models() を適用してモデルを比較する表を作成
        res1 = lineup_models(
            gazer_list,
            model_name = model_name,
            subset = subset,
            **kwargs
            )

        res1.index.name = 'term'
        # 表の下部にモデルの当てはまりに関する統計値を追加
        if stats_glance is not None: # もし stats_glance が None なら統計値を追加しない
            res2 = make_glance_tab(
                list_models,
                model_name = model_name,
                stats_glance = stats_glance,
                digits = digits
                )
            result = pd.concat([res1, res2])

        return result




# 回帰分析の結果から回帰係数のグラフを作成する関数 --------
def mfxplot(
    mod: _HasMargEff,
    subset: Optional[Sequence[str]] = None,
    conf_level: Sequence[float] = (0.95, 0.99),
    at: MfxAt = "overall",
    method: Literal['coef', 'dydx', 'eyex', 'dyex', 'eydx'] = "dydx",
    dummy: bool = False,
    palette: Sequence[str] = ("#1b69af", "#629CE7"),
    show_Intercept: bool = False,
    show_vline: bool = True,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> None:
    """Plot marginal effects (or coefficients) with confidence intervals.

    Args:
        mod:
            A fitted model object that supports marginal effects (get_margeff).
        subset (Sequence[str] | None):
            Terms to include.
        conf_level (Sequence[float]):
            Two confidence levels, e.g. (0.95, 0.99).
        at (str):
            Evaluation point for marginal effects.
        method (str):
            Marginal effect type.
        dummy (bool):
            Whether to treat binary variables as discrete changes.
        palette (Sequence[str]):
            Two colors used for intervals and points.
        show_Intercept (bool):
            Whether to show intercept term.
        show_vline (bool):
            Whether to draw a vertical reference line at 0.
        ax (matplotlib.axes.Axes | None):
            Axes to draw on.
        **kwargs:
            Extra args forwarded to plotting utilities.

    Returns:
        None
    """

    # 回帰係数の表を抽出
    tidy_ci_high = tidy_mfx(
        mod, at = at, method = method, dummy = dummy, conf_level = conf_level[0]
        )
    tidy_ci_row =  tidy_mfx(
        mod, at = at, method = method, dummy = dummy, conf_level = conf_level[1]
        )

    # subset が指定されていれば、回帰係数の部分集合を抽出する
    if subset is not None:
        tidy_ci_high = tidy_ci_high.loc[subset, :]
        tidy_ci_row = tidy_ci_row.loc[subset, :]

    # グラフの作成
    coef_dot(
        tidy_ci_high, tidy_ci_row, estimate = 'estimate', palette = palette,
        show_Intercept = show_Intercept, show_vline = show_vline,
        ax = ax, **kwargs
        )


# ### 多項ロジスティック回帰用



def gazer_MNlogit(
    MNlogit_margeff: pd.DataFrame,
    endog_categories: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Format multinomial logit marginal effects into a comparison-style table.

    This helper reshapes marginal effects output so that `gazer(...)` can be
    applied per outcome category and concatenated horizontally.

    Args:
        MNlogit_margeff (pandas.DataFrame):
            Marginal effects summary frame for MNLogit.
        endog_categories (Sequence[str] | None):
            Outcome categories to include. If None, inferred from `endog` column.
        **kwargs:
            Additional keyword args forwarded to `gazer`.

    Returns:
        pandas.DataFrame:
            A wide table with one column per outcome category and formatted
            marginal effect values.
    """
    if ~pd.Series(MNlogit_margeff.columns).isin(['endog']).any():
        MNlogit_margeff = MNlogit_margeff.reset_index(level = 'endog')

    if endog_categories is None:
        endog_categories = MNlogit_margeff['endog'].unique()

    # gazer 関数で扱えるように列名を修正します。
    MNlogit_margeff = MNlogit_margeff.rename(columns = {
            'Std. Err.':'std_err',
            'z':'statistics',
            'Pr(>|z|)':'p_value',
            'Conf. Int. Low':'conf_lower',
            'Cont. Int. Hi.':'conf_higher'
            }
    )

    list_gazer = list(map(
        lambda categ : gazer(
        MNlogit_margeff.query('endog == @categ'),
        estimate = 'dy/dx',
        **kwargs
        ),
        endog_categories
        ))

    endog_categories2 = [i.split('[')[1].split(']')[0] for i in endog_categories]

    # # flm_total.keys() で回帰式を作成したときに設定したモデル名を抽出し、列名にします。
    res = pd.concat(dict(zip(list(endog_categories2), list_gazer)), axis = 'columns')\
        .droplevel(1, axis = 'columns') # 列名が2重に設定されるので、これを削除して1つにします。

    return res


# # Blinder Oaxaca 分解
# 
#  式については朝井(2014, p.9)を参照しました。
# 
# - 朝井 友紀子 (2014) 「労働市場における男女差の30年― 就業のサンプルセレクションと男女間賃金格差」『日本労働研究雑誌』, No.648, pp.6–16



def Blinder_Oaxaca(
    model1: RegressionResultsWrapper,
    model2: RegressionResultsWrapper,
) -> pd.DataFrame:
    """Compute Blinder-Oaxaca decomposition between two linear models.

    This implementation follows a standard two-group decomposition:
        - observed_diff: difference explained by covariates
        - unobserved_diff: difference attributed to coefficients (including intercept)

    Args:
        model1 (statsmodels.regression.linear_model.RegressionResultsWrapper):
            Fitted regression model for group 1.
        model2 (statsmodels.regression.linear_model.RegressionResultsWrapper):
            Fitted regression model for group 2.

    Returns:
        pandas.DataFrame:
            Decomposition results indexed by terms with columns:
            - observed_diff
            - unobserved_diff
    """
    assert_reg_reuslt(model1)
    assert_reg_reuslt(model2)

    X_1 = pd.DataFrame(model1.model.exog, columns = model1.model.exog_names)
    X_2 = pd.DataFrame(model2.model.exog, columns = model2.model.exog_names)

    X_bar_1 = X_1.mean()
    X_bar_2 = X_2.mean()
    X_diff = X_bar_2 - X_bar_1

    result = pd.DataFrame({
        'observed_diff':X_diff * model2.params,
        'unobserved_diff':X_bar_1 * (model2.params - model1.params)
    })

    result.index.name = 'terms'
    return result




def plot_Blinder_Oaxaca(
    model1: RegressionResultsWrapper,
    model2: RegressionResultsWrapper,
    diff_type: Union[str, Sequence[str]] = ("observed_diff", "unobserved_diff"),
    ax: Optional[Union[Axes, Sequence[Axes]]] = None,
) -> None:
  """Plot Blinder-Oaxaca decomposition results.

  Args:
      model1 (statsmodels.regression.linear_model.RegressionResultsWrapper):
          Fitted regression model for group 1.
      model2 (statsmodels.regression.linear_model.RegressionResultsWrapper):
          Fitted regression model for group 2.
      diff_type (str | Sequence[str]):
          Which decomposition component(s) to plot. Allowed values:
          - 'observed_diff'
          - 'unobserved_diff'
          If multiple are provided, create multiple subplots.
      ax (matplotlib.axes.Axes | Sequence[Axes] | None):
          Axes to draw on. If None, a new figure/axes is created.

  Returns:
      None
  """
  diff_type = build.arg_match(
      diff_type, ['observed_diff', 'unobserved_diff'],
      multiple = True
      )
  fig = None
  result = Blinder_Oaxaca(model1, model2)
  if isinstance(diff_type, list) == False:
    diff_type = [diff_type]

  if ax is None:
    fig, ax = plt.subplots(1, len(diff_type), figsize = (1.1 * len(diff_type) * 4, 4), sharey = True)

  if len(diff_type) == 1:
    ax = [ax]

  for i, t in enumerate(diff_type):
    ax[i].stem(result[t], orientation = 'horizontal', basefmt = 'C7--')
    ax[i].set_yticks(range(len(result.index)), result.index)
    # ax[i].invert_yaxis()
    ax[i].set_title(t);

  if fig is not None:
    fig.tight_layout()

