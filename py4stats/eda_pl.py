#!/usr/bin/env python
# coding: utf-8

# # `eda_pl`：`eda_tools` の polars データフレーム向けメソッド

# In[ ]:


from py4stats import bilding_block as bild # py4stats のプログラミングを補助する関数群
from py4stats import eda_tools as eda        # 基本統計量やデータの要約など
import functools
from functools import singledispatch
import pandas_flavor as pf

import pandas as pd
import numpy as np
import scipy as sp

import polars as pl
import tidypolars as tp


# In[ ]:


# diagnose の polars 版
@eda.diagnose.register(pl.DataFrame)
def diagnose_pl(self):
  res = pl.DataFrame({
      'columns':self.columns,
      'dtype':self.dtypes,
      'missing_count':self.null_count().row(0),
      'unique_count':[self[col].n_unique() for col in self.columns]
  }).with_columns(
      (100 * pl.col('missing_count') / len(self)).alias('missing_percent'),
      (100 * pl.col('unique_count') / len(self)).alias('unique_rate')
  )\
    .select('columns', 'dtype', 'missing_count', 'missing_percent', 'unique_count', 'unique_rate')
  return res

@eda.diagnose.register(tp.tibble.Tibble)
def diagnose_tp(self):
  return diagnose_pl(self.to_polars())


# In[ ]:


@eda.remove_constant.register(pl.DataFrame)
@eda.remove_constant.register(tp.tibble.Tibble)
def remove_constant_pl(self, quiet = True, dropna = False):
  res = eda.remove_constant(self.to_pandas(), quiet = quiet, dropna = dropna)
  return pl.from_pandas(res)


# ## グループ別平均（中央値）の比較

# In[ ]:


@eda.compare_group_means.register(pl.DataFrame)
@eda.compare_group_means.register(tp.tibble.Tibble)
def compare_group_means_pl(group1, group2, group_names = ['group1', 'group2']):
  group1 = group1.to_pandas()
  group2 = group2.to_pandas()
  res = eda.compare_group_means(group1, group2, group_names = group_names)
  return res


# In[ ]:


@eda.compare_group_median.register(pl.DataFrame)
@eda.compare_group_median.register(tp.tibble.Tibble)
def compare_group_median_pl(group1, group2, group_names = ['group1', 'group2']):
  group1 = group1.to_pandas()
  group2 = group2.to_pandas()
  res = eda.compare_group_median(group1, group2, group_names = group_names)
  return res


# ## クロス集計表ほか

# In[ ]:


@eda.freq_table.register(tp.tibble.Tibble)
@eda.freq_table.register(pl.DataFrame)
def freq_table_pl(self, subset, **kwargs):
  res = eda.freq_table(self.to_pandas(), subset = subset, **kwargs)
  return pl.from_pandas(res.reset_index())


# In[ ]:


@eda.crosstab2.register(tp.tibble.Tibble)
@eda.crosstab2.register(pl.DataFrame)
def crosstab2_pl(data, index, columns, **kwargs):

  res = eda.crosstab2(data.to_pandas(), index = index, columns = columns, **kwargs)

  return pl.from_pandas(res.reset_index())


# In[ ]:


@eda.tabyl.register(tp.tibble.Tibble)
@eda.tabyl.register(pl.DataFrame)
def tabyl_pl(data, index, columns, **kwargs):

  res = eda.tabyl(data.to_pandas(), index = index, columns = columns, **kwargs)

  return res

