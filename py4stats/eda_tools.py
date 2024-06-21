#!/usr/bin/env python
# coding: utf-8

# # `eda_tools`：データセットを要約する関数群

# In[ ]:


from py4stats import bilding_block as bild # py4stats のプログラミングを補助する関数群
import functools
from functools import singledispatch
import pandas_flavor as pf

import pandas as pd
import numpy as np
import scipy as sp


# In[ ]:


import pandas_flavor as pf

def missing_percent(x, axis = 'index', pct = True):
  return (100**pct) * x.isna().mean(axis = axis)

@pf.register_dataframe_method
@singledispatch
def diagnose(self):
  """
  ## `diagnose()`
  ### ## 返り値 Value
  - `index`（行名）：もとのデータフレームの列名に対応しています。
  - `dtype`：該当する列のpandasにおけるデータの型。「〇〇の個数」や「〇〇の金額」といった列の dtype が `object` になっていたら、文字列として読み込まれているので要注意です。

  - `missing_count`：該当する列のなかで `NaN` などの欠測値になっている数
  - `missing_percent`：該当する列のなかで欠測値が占めている割合で 欠`missing_percent = 100 * missing_count/ nrow` として計算されます。もし `missing_percent = 100` なら、その列は完全に空白です。
  - `unique_count`：その列で重複を除外したユニークな値の数。例えばある列の中身が「a, a, b, b, b」であればユニークな値は `a` と `b` の2つなのでユニーク値の数は2です。もし ユニーク値の数 = 1 であれば、その行にはたった1種類の値しか含まれていないことが分かりますし、例えば都道府県を表す列のユニーク値の数が47より多ければ、都道府県以外のものが混ざっていると考えられます。
  - `unique_rate`： サンプルに占めるユニークな値の割合。 `unique_rate = 100 * unique_count / nrow`と計算されます。 `unique_rate = 100` であれば、全ての行に異なる値が入っています。一般的に実数値の列はユニーク率が高くなりますが、年齢の「20代」や価格の「400円代」のように、階級に分けられている場合にはユニーク率が低くなります。
  """
  self = self.copy()
  # 各種集計値の計算 ------------
  result = pd.DataFrame({
    'dtype':self.dtypes,
    'missing_count':self.isna().sum(),
    'missing_percent':missing_percent(self),
    'unique_count':self.nunique(),
    'unique_rate': 100 * self.nunique() / len(self),
  })

  return result


# ### 異なるデータフレームの列を比較する関数

# In[ ]:


def compare_df_cols(df_list, return_match = 'all', df_name = None, dropna = False):
  """複数の pandas.DataFrame に含まれる同じ名前を持つ列同士のデータ型 `dtype` を比較します。"""
  # 引数のアサーション ----------------------
  assert isinstance(df_list, list) & \
        all([isinstance(v, pd.DataFrame) for v in df_list]),\
        "argument 'df_list' is must be a list of pandas.DataFrame."

  return_match = bild.arg_match(
      return_match,
       ['all', 'match', 'mismatch'],
      arg_name = 'return_match'
      )
  # --------------------------------------
  # df_name が指定されていなければ、自動で作成します。
  if df_name is None:
      df_name = [f'df{i + 1}' for i in range(len(df_list))]

  df_list = [v.copy() for v in df_list] # コピーを作成
  dtype_list = [v.dtypes for v in df_list]
  res = pd.concat(dtype_list, axis = 1)
  res.columns = df_name
  res.index.name = 'term'
  res['match_dtype'] = res.nunique(axis = 1, dropna = dropna) == 1

  if(return_match == 'match'):
    res = res[res['match_dtype']]
  elif(return_match == 'mismatch'):
    res = res[~res['match_dtype']]

  return res


# 平均値などの統計値の近接性で比較するバージョン

# In[ ]:


import itertools

def compare_df_stats(
    df_list, return_match = 'all', df_name = None,
    stats = 'mean', rtol = 1e-05, atol = 1e-08,
    **kwargs
    ):
  """複数の pandas.DataFrame に含まれる同じ名前を持つ列同士のデータ型 `dtype` を比較します。"""
  # 引数のアサーション ----------------------
  assert isinstance(df_list, list) & \
        all([isinstance(v, pd.DataFrame) for v in df_list]),\
        "argument 'df_list' is must be a list of pandas.DataFrame."

  return_match = bild.arg_match(
      return_match,
       ['all', 'match', 'mismatch'],
      arg_name = 'return_match'
      )
  # --------------------------------------
  # df_name が指定されていなければ、自動で作成します。
  if df_name is None:
      df_name = [f'df{i + 1}' for i in range(len(df_list))]

  df_list = [v.copy() for v in df_list] # コピーを作成
  stats_list = [
      v.select_dtypes(include = ['int', 'float', 'bool'])\
      .dropna(axis = 1, how = 'all').agg(stats, **kwargs)
      for v in df_list
      ]
  res = pd.concat(stats_list, axis = 1)
  res.columns = df_name
  res.index.name = 'term'

  # データフレームのペア毎に、統計値が近いかどうかを比較します。
  pairwise_comparesion = \
   [pd.Series(
    np.isclose(
        res.iloc[:, i], res.iloc[:, j],
        rtol = rtol, atol = atol
    ), index = res.index)
   for i, j in itertools.combinations(range(len(res.columns)), 2)
    ]

  res['match_stats'] = pd.concat(pairwise_comparesion, axis = 1).all(axis = 1)

  if(return_match == 'match_stats'):
    res = res[res['match']]
  elif(return_match == 'mismatch'):
    res = res[~res['match_stats']]

  return res


# In[ ]:


# レコード毎の近接性（数値の場合）または一致性（数値以外）で評価する関数
def compare_df_record(df1, df2, rtol = 1e-05, atol = 1e-08):
  all_columns = df1.columns
  number_col = df1.select_dtypes(include = 'number').columns
  nonnum_col = df1.select_dtypes(exclude = 'number').columns

  res_number_col = [np.isclose(df1[v], df2[v]) for v in number_col]

  res_nonnum_col = [(df1[v] == df2[v]) for v in nonnum_col]

  result = pd.concat([
      pd.DataFrame(res_number_col, index = number_col).T,
      pd.DataFrame(res_nonnum_col, index = nonnum_col).T
  ], axis = 'columns').loc[:, all_columns]

  return result


# ## グループ別平均（中央値）の比較

# In[ ]:


@singledispatch
def compare_group_means(group1, group2, group_names = ['group1', 'group2']):
  group1 = remove_constant(group1)
  group2 = remove_constant(group2)

  res = pd.DataFrame({
    group_names[0]:group1.mean(numeric_only = True),
    group_names[1]:group2.mean(numeric_only = True)
    })

  s2A = group1.var(numeric_only = True)
  s2B = group2.var(numeric_only = True)
  nA = group1.shape[0]
  nB = group2.shape[0]

  s2_pooled = ((nA - 1) * s2A + (nB - 1) * s2B) / (nA + nB - 2)
  res['norm_diff'] = (res[group_names[0]] - res[group_names[1]]) / np.sqrt(s2_pooled)

  res['abs_diff'] = (res[group_names[0]] - res[group_names[1]]).abs()
  res['rel_diff'] = 2 * (res[group_names[0]] - res[group_names[1]]) \
                    /(res[group_names[0]] + res[group_names[1]])
  return res


# In[ ]:


@singledispatch
def compare_group_median(group1, group2, group_names = ['group1', 'group2']):
  group1 = remove_constant(group1)
  group2 = remove_constant(group2)

  res = pd.DataFrame({
    group_names[0]:group1.median(numeric_only = True),
    group_names[1]:group2.median(numeric_only = True)
    })

  res['abs_diff'] = (res[group_names[0]] - res[group_names[1]]).abs()
  res['rel_diff'] = 2 * (res[group_names[0]] - res[group_names[1]]) \
                    /(res[group_names[0]] + res[group_names[1]])
  return res


# In[ ]:


def plot_mean_diff(group1, group2, stats_diff = 'norm_diff', ax = None):
  stats_diff = bild.arg_match(
      stats_diff, ['norm_diff', 'abs_diff', 'rel_diff']
      )
  group_means = compare_group_means(group1, group2)

  if ax is None:
    fig, ax = plt.subplots()

  ax.stem(group_means[stats_diff], orientation = 'horizontal', basefmt = 'C7--');

  ax.set_yticks(range(len(group_means.index)), group_means.index)

  ax.invert_yaxis();


# In[ ]:


def plot_median_diff(group1, group2, stats_diff = 'rel_diff', ax = None):
  stats_diff = bild.arg_match(
      stats_diff, ['abs_diff', 'rel_diff']
      )

  group_median = compare_group_median(group1, group2)

  if ax is None:
    fig, ax = plt.subplots()

  ax.stem(group_median[stats_diff], orientation = 'horizontal', basefmt = 'C7--')
  ax.set_yticks(range(len(group_median.index)), group_median.index)
  ax.invert_yaxis();


# ## 完全な空白列 and / or 行の除去

# In[ ]:


@pf.register_dataframe_method
def remove_empty(self, cols = True, rows = True, cutoff = 1, quiet = True):
  df_shape = self.shape

  # 空白列の除去 ------------------------------
  if cols :
    empty_col = missing_percent(self, axis = 'index', pct = False) >= cutoff
    self = self.loc[:, ~empty_col]

    if not(quiet) :
      ncol_removed = empty_col.sum()
      col_removed = empty_col[empty_col].index.to_series().astype('str').to_list()
      print(
            f"Removing {ncol_removed} empty column(s) out of {df_shape[1]} columns" +
            f"(Removed: {','.join(col_removed)}). "
            )
  # 空白行の除去 ------------------------------
  if rows :
    empty_rows = missing_percent(self, axis = 'columns', pct = False) >= cutoff
    self = self.loc[~empty_rows, :]

    if not(quiet) :
        nrow_removed = empty_rows.sum()
        row_removed = empty_rows[empty_rows].index.to_series().astype('str').to_list()
        print(
              f"Removing {nrow_removed} empty row(s) out of {df_shape[0]} rows" +
              f"(Removed: {','.join(row_removed)}). "
          )

  return self


# ## 定数列の除去

# In[ ]:


@pf.register_dataframe_method
@singledispatch
def remove_constant(self, quiet = True, dropna = False):
  df_shape = self.shape
  # データフレーム(self) の行が定数かどうかを判定
  constant_col = self.nunique(dropna = dropna) == 1
  self = self.loc[:, ~constant_col]

  if not(quiet) :
    ncol_removed = constant_col.sum()
    col_removed = constant_col[constant_col].index.to_series().astype('str').to_list()

    print(
        f"Removing {ncol_removed} constant column(s) out of {df_shape[1]} columns" +
        f"(Removed: {','.join(col_removed)}). "
     )

  return self


# In[ ]:


# 列名に特定の文字列を含む列を除外する関数
@pf.register_dataframe_method
def filtering_out(self, contains = None, starts_with = None, ends_with = None, axis = 1):
  axis = str(axis)
  axis = bild.arg_match(axis, ['1', 'columns', '0', 'index'], arg_name = 'axis')
  self = self.copy()

  if((axis == '1') | (axis == 'columns')):
      if contains is not None:
        assert isinstance(contains, str), "'contains' must be a string."
        self = self.loc[:, ~self.columns.str.contains(contains)]

      if starts_with is not None:
        assert isinstance(starts_with, str), "'starts_with' must be a string."
        self = self.loc[:, ~self.columns.str.startswith(starts_with)]

      if ends_with is not None:
        assert isinstance(ends_with, str), "'ends_with' must be a string."
        self = self.loc[:, ~self.columns.str.endswith(ends_with)]
  else:
      if contains is not None:
        assert isinstance(contains, str), "'contains' must be a string."
        self = self.loc[~self.index.to_series().str.contains(contains), :]

      if starts_with is not None:
        assert isinstance(starts_with, str), "'starts_with' must be a string."
        self = self.loc[~self.index.to_series().str.startswith(starts_with), :]

      if ends_with is not None:
        assert isinstance(ends_with, str), "'ends_with' must be a string."
        self = self.loc[~self.index.to_series().str.endswith(ends_with), :]

  return self


# ## クロス集計表ほか

# In[ ]:


@pf.register_dataframe_method
@singledispatch
def crosstab2(
    data, index, columns, values=None, rownames=None, colnames=None,
    aggfunc=None, margins=False, margins_name='All', dropna=True, normalize=False
    ):

    res = pd.crosstab(
        index = data[index], columns = data[columns], values = values,
        rownames = rownames, colnames = colnames,
        aggfunc = aggfunc, margins = margins, margins_name = margins_name,
        dropna = dropna, normalize = normalize
        )
    return res


# In[ ]:


@pf.register_dataframe_method
@singledispatch
def freq_table(self, subset, sort = True, ascending = False, dropna = False):
  count = self.value_counts(
      subset = subset, sort = sort, ascending = ascending,
      normalize=False, dropna = dropna
      )

  rel_count = self.value_counts(
      subset = subset, sort = sort, ascending = ascending,
      normalize=True, dropna = dropna
      )

  res = pd.DataFrame({
          'freq':count,
          'perc':rel_count,
          'cumfreq':count.cumsum(),
          'cumperc':rel_count.cumsum()
      })
  return res


# In[ ]:


@pf.register_dataframe_method
@singledispatch
def tabyl(
    self,
    index,
    columns,
    margins = True,
    margins_name = 'All',
    normalize = 'index',
    dropna = False,
    rownames = None,
    colnames = None,
    digits = 1,
    ):
    if(not isinstance(normalize, bool)):
      normalize = bild.arg_match(
          normalize, ['index', 'columns', 'all'],
          arg_name = 'normalize'
          )

    if self[index].dtype == "bool":
        self[index] = self[index].astype(str)
    if self[columns].dtype == "bool":
        self[columns] = self[columns].astype(str)

    # 度数クロス集計表（最終的な表では左側の数字）
    c_tab1 = pd.crosstab(
        index = self[index], columns = self[columns], values = None,
        rownames = rownames, colnames = colnames,
        aggfunc = None, margins = margins, margins_name = margins_name,
        dropna = dropna, normalize = False
        )

    c_tab1 = c_tab1.apply(bild.style_number, digits = 0)

    if(normalize != False):

      # 回答率クロス集計表（最終的な表では括弧内の数字）
      c_tab2 = pd.crosstab(
          index = self[index], columns = self[columns], values = None,
          rownames = rownames, colnames = colnames,
          aggfunc = None, margins = margins, margins_name = margins_name,
          dropna = dropna, normalize = normalize
          )

      # 2つめのクロス集計表の回答率をdigitsで指定した桁数のパーセントに換算し、文字列化します。
      c_tab2 = c_tab2.apply(bild.style_percent, digits = digits)

      col = c_tab2.columns
      idx = c_tab2.index
      # 1つめのクロス集計表も文字列化して、↑で計算したパーセントに丸括弧と%記号を追加したものを文字列として結合します。
      c_tab1.loc[idx, col] = c_tab1.astype('str').loc[idx, col] + ' (' + c_tab2 + ')'

    return c_tab1


# ## `diagnose_category()`：カテゴリー変数専用の要約関数

# In[ ]:


@pf.register_dataframe_method
@pf.register_series_method
@singledispatch
def is_dummy(self, cording = [0, 1]): return set(self) == set(cording)

@is_dummy.register(pd.DataFrame)
def _(self, cording = [0, 1]): return self.apply(is_dummy, cording = cording)


# In[ ]:


# カテゴリカル変数についての集計関数 --------------
# 情報エントロピーと、その値を0から1に標準化したもの --------------
def entropy(X, base = 2, axis = 0):
    vc = pd.Series(X).value_counts(normalize = True, sort = False)
    res = sp.stats.entropy(pk = vc,  base = base, axis = axis)
    return res

def std_entropy(X, axis = 0):
    K = pd.Series(X).nunique()
    res = entropy(X, base = K) if K > 1 else 0.0
    return res

def freq_mode(x, normalize = False):
    res = x.value_counts(normalize = normalize, dropna = False).iloc[0]
    return res

# カテゴリカル変数についての概要を示す関数
def diagnose_category(data):
    # 01のダミー変数はロジカル変数に変換
    data = data.copy()
    data.loc[:, is_dummy(data)] = (data.loc[:, is_dummy(data)] == 1)
    # 文字列 or カテゴリー変数のみ抽出
    data = data.select_dtypes(include = [object, 'category', bool])

    n = len(data)
    # describe で集計表の大まかな形を作成
    res = data.describe().T
    res['freq'] = res['freq'].astype('int')
    # 追加の集計値を計算して代入
    res = res.assign(
        unique_percent = 100 * data.nunique(dropna = False) / n,
        missing_percent = missing_percent(data),
        pct_mode = (100 * (res['freq'] / n)),
        std_entropy = data.agg(std_entropy)
    )
    # 見やすいように並べ替え
    res = res.loc[:, [
        'count', 'missing_percent', 'unique', 'unique_percent',
        'top', 'freq', 'pct_mode', 'std_entropy'
        ]]

    return res


# ## その他の補助関数

# In[ ]:


def weighted_mean(x, w):
  wmean = (x * w).sum() / w.sum()
  return wmean

def scale(x, ddof = 1):
    z = (x - x.mean()) / x.std(ddof = ddof)
    return z

def min_max(x):
  mn = (x - x.min()) / (x.max() - x.min())
  return mn


# # パレート図を作図する関数

# In[ ]:


import matplotlib.pyplot as plt

# パレート図に使用するランキングを作成する関数
def make_rank_table(data, group, values, aggfunc = 'sum'):
    # ピボットテーブルを使って、カテゴリー group（例：メーカー）ごとの values （例：販売額）の合計を計算
    p_table = pd.pivot_table(
        data = data,
        index = group,
        values = values,
        aggfunc = aggfunc,
        fill_value = 0
        )
    # values の値に基づいてソート
    rank_table = p_table.sort_values(values, ascending=False)

    # シェア率と累積相対度数を計算
    rank_table['share'] = (rank_table[values] / rank_table[values].sum())
    rank_table['cumshare'] = rank_table['share'].cumsum()
    return rank_table


# In[ ]:


# パレート図を作成する関数
def Pareto_plot(
    data,
    group,
    values = None,
    top_n = None,
    aggfunc = 'sum',
    ax = None,
    fontsize = 12,
    xlab_rotation = 0,
    palette = ['#478FCE', '#252525']
    ):
  # 引数のアサーション
  if(top_n is not None): bild.assert_count(top_n, lower = 1)
  bild.assert_numeric(xlab_rotation)
  bild.assert_character(palette)

  # 指定された変数でのランクを表すデータフレームを作成
  if values is None:
      shere_rank = freq_table(data, group, dropna = True)
      cumlative = 'cumfreq'
  else:
      shere_rank = make_rank_table(data, group, values, aggfunc = aggfunc)
      cumlative = 'cumshare'

  # グラフの描画
  if ax is None:
      fig, ax = plt.subplots()

  # yで指定された変数の棒グラフ

  # top_n が指定されていた場合、上位 top_n 件を抽出します。
  if top_n is not None:
    shere_rank = shere_rank.head(top_n)

  if values is None:
      ax.bar(shere_rank.index, shere_rank['freq'], color = palette[0])
      ax.set_ylabel('freq', fontsize = fontsize * 1.1)
  else:
      # yで指定された変数の棒グラフ
      ax.bar(shere_rank.index, shere_rank[values], color = palette[0])
      ax.set_ylabel(values, fontsize = fontsize * 1.1)


  ax.set_xlabel(group, fontsize = fontsize * 1.1)

  # 累積相対度数の線グラフ
  ax2 = ax.twinx()
  ax2.plot(
      shere_rank.index, shere_rank[cumlative],
      linestyle = 'dashed', color = palette[1], marker = 'o'
      )

  ax2.set_xlabel(group, fontsize = fontsize * 1.1)
  ax2.set_ylabel(cumlative, fontsize = fontsize * 1.1)

  # x軸メモリの回転
  ax.xaxis.set_tick_params(rotation = xlab_rotation, labelsize = fontsize)
  ax2.xaxis.set_tick_params(rotation = xlab_rotation, labelsize = fontsize);
  ax.yaxis.set_tick_params(labelsize = fontsize * 0.9)
  ax2.yaxis.set_tick_params(labelsize = fontsize * 0.9);


# ### 代表値 + 区間推定関数
# 
# ```python
# import pandas as pd
# from palmerpenguins import load_penguins
# penguins = load_penguins() # サンプルデータの読み込み
# 
# from py4stats import eda_tools as eda
# 
# print(penguins['bill_depth_mm'].mean_qi().round(2))
# #>                 mean  lower  upper
# #> variable                          
# #> bill_depth_mm  17.15   13.9   20.0
# 
# print(penguins['bill_depth_mm'].median_qi().round(2))
# #>                median  lower  upper
# #> variable                           
# #> bill_depth_mm    17.3   13.9   20.0
# 
# print(penguins['bill_depth_mm'].mean_ci().round(2))
# #>                 mean  lower  upper
# #> variable                          
# #> bill_depth_mm  17.15  16.94  17.36
# ```

# In[ ]:


@pf.register_dataframe_method
@pf.register_series_method
def mean_qi(self, width = 0.975, point_fun = 'mean'):

  bild.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither')
  if(isinstance(self, pd.DataFrame)):
    var_name = self.columns
  else:
    var_name = [self.name]

  res = pd.DataFrame({
      'mean':self.apply('mean'),
      'lower':self.quantile(1 - width),
      'upper':self.quantile(width),
  }, index = var_name
  )

  res.index.name = 'variable'
  return res


# In[ ]:


@pf.register_dataframe_method
@pf.register_series_method
def median_qi(self, width = 0.975, point_fun = 'median'):

  bild.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither')
  if(isinstance(self, pd.DataFrame)):
    var_name = self.columns
  else:
    var_name = [self.name]

  res = pd.DataFrame({
      'median':self.apply('median'),
      'lower':self.quantile(1 - width),
      'upper':self.quantile(width),
  }, index = var_name
  )

  res.index.name = 'variable'
  return res


# In[ ]:


from scipy.stats import t
@pf.register_dataframe_method
@pf.register_series_method
def mean_ci(self, width = 0.95):

  bild.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither')
  if(isinstance(self, pd.DataFrame)):
    var_name = self.columns
  else:
    var_name = [self.name]

  n = len(self)
  t_alpha = t.isf((1 - width) / 2, df = n - 1)
  x_mean = self.mean()
  x_std = self.std()

  res = pd.DataFrame({
      'mean':x_mean,
      'lower':x_mean - t_alpha * x_std / np.sqrt(n),
      'upper':x_mean + t_alpha * x_std / np.sqrt(n),
      }, index = var_name
    )
  res.index.name = 'variable'
  return res


# ## 正規表現を文字列関連の論理関数

# In[ ]:


import regex
def detect_Kanzi(s):
  p = regex.compile(r'.*\p{Script=Han}+.*')
  res = p.fullmatch(s)
  return res is not None


# In[ ]:


@pf.register_series_method
def is_number(self, na_default = True):
  """文字列が数字であるかどうかを判定する関数"""
  rex_phone = '[0-9]{0,4}(?: |-)[0-9]{0,4}(?: |-)[0-9]{0,4}'
  rex_han = '[Script=Han]+'

  self_str = self.copy().astype(str)

  res = self_str.str.contains('[0-9]+', regex = True)\
    & ~ self_str.str.contains(rex_phone, regex = True)\
    & ~ self_str.str.contains('[\u3041-\u309F]+', regex = True)\
    & ~ self_str.str.contains('[\u30A1-\u30FF]+', regex = True)\
    & ~ self_str.str.contains('[A-z]+', regex = True)\
    & ~ self_str.map(detect_Kanzi)\
    & ~ is_ymd_like(self_str)

  exponent = self_str.str.contains('[0-9]+[E,e]+(?:\+|-)[0-9]+', regex = True)
  res[exponent] = True

  res[self.isna()] = na_default

  return res


# In[ ]:


@pf.register_series_method
def is_ymd(self, na_default = True):
  """与えられた文字列が ymd 形式の日付かどうかを判定する関数"""
  rex_ymd = '[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}'

  self_str = self.copy().astype(str)

  res = self_str.str.contains(rex_ymd, regex = True)

  res[self.isna()] = na_default

  return res

@pf.register_series_method
def is_ymd_like(self, na_default = True):
  """与えられた文字列が ymd 形式っぽい日付かどうかを判定する関数"""
  rex_ymd_like = '[Script=Han]{0,2}[0-9]{1,4}(?:年|-)[0-9]{1,2}(?:月|-)[0-9]{1,2}(?:日)?'

  self_str = self.copy().astype(str)

  res = self_str.str.contains(rex_ymd_like, regex = True)

  res[self.isna()] = na_default

  return res


# ## set missing values in pd.Series

# In[ ]:


def set_n_miss(x, n = 10, method = 'random', random_state = None, na_value = pd.NA):
  method = bild.arg_match(method, ['random', 'first', 'last'])
  bild.assert_count(n, upper = len(x))

  x = x.copy()
  n_miss = x.isna().sum()
  non_miss = x.dropna().index.to_series()

  if(method == 'random'):
    index_to_na = non_miss.sample(n = n - n_miss, random_state = random_state)
  elif(method == 'first'):
    index_to_na = non_miss.head(n - n_miss)
  elif(method == 'last'):
    index_to_na = non_miss.tail(n - n_miss)

  x[index_to_na] = na_value

  return x


# In[ ]:


def set_prop_miss(x, prop = 0.1, method = 'random', random_state = None, na_value = pd.NA):
  method = bild.arg_match(method, ['random', 'first', 'last'])
  bild.assert_numeric(prop, lower = 0, upper = 1)

  x = x.copy()
  prop_miss = x.isna().mean()
  non_miss = x.dropna().index.to_series()

  if(method == 'random'):
    index_to_na = non_miss.sample(frac = prop - prop_miss, random_state = random_state)
  elif(method == 'first'):
    n = round(len(non_miss) * (prop - prop_miss))
    index_to_na = non_miss.head(n)
  elif(method == 'last'):
    n = round(len(non_miss) * (prop - prop_miss))
    index_to_na = non_miss.tail(n)

  x[index_to_na] = na_value

  return x


# - `eda.set_n_miss()`： `pd.Series` の欠損数が `n` 個になるように欠測値を追加します。
# - `eda.set_prop_miss()`： `pd.Series` の欠損率が約 `prop` になるように欠測値を追加します。
# 
# 引数
# 
# - method：**str**</br>
#     - `'random'`（初期設定）：`x` のランダムな位置を、欠損値に変換します。
#     - `'first'`：`x` 冒頭を欠損値に変換します。
#     - `'last'`：`x` 末尾を欠損値に変換します。
# 
# ```python
# from py4stats import eda_tools as eda
# from palmerpenguins import load_penguins
# penguins = load_penguins() # サンプルデータの読み込み
# s = penguins['bill_depth_mm'].copy()
# 
# print(s.isna().sum()) # 当初の欠測値の数
# #> 2
# 
# print(eda.set_n_miss(s, n = 20).isna().sum())
# #> 20
# 
# print(eda.set_n_miss(s, method = 'first'))
# #> 0       NaN
# #> 1       NaN
# #> 2       NaN
# #> 3       NaN
# #> 4       NaN
# #>        ...
# #> 339    19.8
# #> 340    18.1
# #> 341    18.2
# #> 342    19.0
# #> 343    18.7
# #> Name: bill_depth_mm, Length: 344, dtype: float64
# 
# print(eda.set_n_miss(s, method = 'last'))
# #> 0      18.7
# #> 1      17.4
# #> 2      18.0
# #> 3       NaN
# #> 4      19.3
# #>        ...
# #> 339     NaN
# #> 340     NaN
# #> 341     NaN
# #> 342     NaN
# #> 343     NaN
# #> Name: bill_depth_mm, Length: 344, dtype: float64
# 
# print(eda.set_prop_miss(s, prop = 0.2).isna().mean())
# #> 0.19767441860465115
# ```

# In[ ]:


@pf.register_dataframe_method
def check_that(data, rule_dict, **kwargs):
  if(isinstance(rule_dict, pd.Series)): rule_dict = rule_dict.to_dict()

  [bild.assert_character(x, arg_name = 'rule_dict') for x in rule_dict.values()]

  result_list = []
  for i, name in enumerate(rule_dict):
    condition = data.eval(rule_dict[name], **kwargs)
    condition = pd.Series(condition)
    assert bild.is_logical(condition),\
    f"Result of rule(s) must be of type 'bool'. But result of '{name}' is '{condition.dtype}'."

    if len(condition) == len(data):
      in_exper = [s in rule_dict[name] for s in data.columns]
      any_na = data.loc[:, in_exper].isna().any(axis = 'columns')
      condition[any_na] = pd.NA
      condition = condition.astype('boolean')

    res_df = pd.DataFrame({
        'item':len(condition),
        'passes':condition.sum(skipna = True),
        'fails':(~condition).sum(skipna = True),
        'coutna':condition.isna().sum(),
        'expression':rule_dict[name]
        }, index = [name])

    result_list.append(res_df)

  result_df = pd.concat(result_list)
  result_df.index.name = 'name'

  return result_df


# In[ ]:


@pf.register_dataframe_method
def check_viorate(data, rule_dict, **kwargs):
  if(isinstance(rule_dict, pd.Series)): rule_dict = rule_dict.to_dict()
  [bild.assert_character(x, arg_name = 'rule_dict') for x in rule_dict.values()]

  df_viorate = pd.DataFrame()
  for i, name in enumerate(rule_dict):
    condition = data.eval(rule_dict[name], **kwargs)
    assert bild.is_logical(condition),\
    f"Result of rule(s) must be of type 'bool'. But result of '{name}' is '{condition.dtype}'."

    df_viorate[name] = ~condition

  df_viorate['any'] = df_viorate.any(axis = 'columns')
  df_viorate['all'] = df_viorate.all(axis = 'columns')

  return df_viorate


# ### helper function for pandas `DataFrame.eval()`

# In[ ]:


def implies_exper(P, Q):
  return f"{Q} | ~({P})"

@pf.register_dataframe_method
@singledispatch
def is_complet(self): return self.notna().all(axis = 'columns')

@is_complet.register(pd.Series)
def is_complet(*arg): return pd.concat(arg, axis = 'columns').notna().all(axis = 'columns')


# In[ ]:


def Sum(*arg): return pd.concat(arg, axis = 'columns').sum(axis = 'columns')
def Mean(*arg): return pd.concat(arg, axis = 'columns').mean(axis = 'columns')
def Max(*arg): return pd.concat(arg, axis = 'columns').max(axis = 'columns')
def Min(*arg): return pd.concat(arg, axis = 'columns').min(axis = 'columns')
def Median(*arg): return pd.concat(arg, axis = 'columns').median(axis = 'columns')

