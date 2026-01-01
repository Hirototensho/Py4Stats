# `Py4Stats` 

　`Py4Stats` は、主に実証研究で用いられる、探索的データ分析および回帰結果レポート用のユーティリティライブラリで、回帰分析を中心とする分析でよく使われるR言語の機能を、Python で実装しています。本ライブラリの主な機能は [Get started](https://github.com/Hirototensho/Py4Stats/blob/main/INTRODUCTION.md) を、実装されている関数の一覧は [Function reference](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md) を参照してください。

## Installation

[`uv`](https://github.com/astral-sh/uv) をお使いの場合、次のコードで `py4stats` をインストールできます。

``` python
! uv add git+https://github.com/Hirototensho/py4stats.git
```

一方で、`pip` をお使いの場合には、次のコードで `py4stats` をインストールできます。

``` python
! pip install git+https://github.com/Hirototensho/py4stats.git
```

## 使用例

　[`py4stats.compare_ols()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_ols.md) ：計量経済学の実証論文でよく用いられる、回帰分析の結果を列方向に並べて比較する表を作成します。

``` python
import py4stats as py4st
import statsmodels.formula.api as smf
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込み

# 回帰分析の実行
fit1 = smf.ols('body_mass_g ~ bill_length_mm + species', data = penguins).fit()
fit2 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species', data = penguins).fit()
fit3 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species + sex', data = penguins).fit()

compare_tab1 = py4st.compare_ols(list_models = [fit1, fit2, fit3]) # 表の作成
compare_tab1
```

| term                 | model 1        | model 2          | model 3         |
|:---------------------|:---------------|:-----------------|:----------------|
| Intercept            | 153.7397       | -1,742.7202  *** | 843.9812  **    |
|                      | (268.9012)     | (313.7697)       | (403.5956)      |
| species[T.Chinstrap] | -885.8121  *** | -539.6864  ***   | -245.1516  ***  |
|                      | (88.2502)      | (86.9425)        | (84.5952)       |
| species[T.Gentoo]    | 578.6292  ***  | 1,492.8283  ***  | 1,443.3525  *** |
|                      | (75.3623)      | (118.4442)       | (107.7844)      |
| bill_length_mm       | 91.4358  ***   | 55.6461  ***     | 26.5366  ***    |
|                      | (6.8871)       | (7.2326)         | (7.2436)        |
| bill_depth_mm        |                | 179.0434  ***    | 87.9328  ***    |
|                      |                | (19.0997)        | (20.2192)       |
| sex[T.male]          |                |                  | 437.2007  ***   |
|                      |                |                  | (49.1098)       |
| rsquared_adj         | 0.7810         | 0.8258           | 0.8613          |
| nobs                 | 342            | 342              | 333             |
| df                   | 3              | 4                | 5               |


詳細は、[`py4stats.compare_ols()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_ols.md) を参照してください。　

***
[Jump to **Function Get started**.](https://github.com/Hirototensho/Py4Stats/blob/main/INTRODUCTION.md)  
[Jump to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)

