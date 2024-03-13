# `heckit_helper.heckitmfx_compute()`

## 概要

　Type2トービットモデルの限界効果を推定します。推定方法についてはダハナ, 勝又(2023, p.136)および Hoffmann, Kassouf(2005)を参照し、関数の実装についてはR言語の [`heckitmfx::heckitmfx_log()`](https://github.com/santosglauber/heckitmfx/tree/main) 関数を参考にしています。

``` python
heckitmfx_compute(
    model, 
    exog_select, 
    exog_outcome, 
    exponentiate = False
)
```

## 引数 Argument

- `model`（必須）</br>
　 [`Py4Etrics`](https://github.com/Py4Etrics/py4etrics) モジュールの `py4etrics.heckit.Heckit()` で作成された `HeckitResults` クラスのオブジェクト
- `exog_select`**pd.DataFrame**（必須）</br>
　Type2トービットモデルのうち第1段階の selection equation(選択関数, 就業決定関数)の説明変数からなる pd.DataFrame
- `exog_outcome`**pd.DataFrame**（必須）</br>
　Type2トービットモデルのうち第2段階の regression equation(賃金関数)の説明変数からなる pd.DataFrame

これらの引数は [`heckit_helper.Heckit_from_formula()`](https://github.com/Hirototensho/Py4Stats/edit/main/man/Heckit_from_formula.md) の出力を使用することを想定しています（使用例を参照）。

- `exponentiate`**bool**</br>
　推定結果に指数関数を用いた変換を行うかどうかを表す論理値。もし False （初期設定）であれば限界効果と回帰係数の推定値をそのまま出力し、もし True であれば出力されるデータフレームのうち `unconditional`、`conditional`、`selection`、`beta` の列について指数関数 $100[\exp(x - 1)]$ を用いた変換を行います。例えば被説明変数は対数賃金であれば、変換後の限界効果はパーセンテージで表された賃金の変化率として解釈できます。

## 返り値 Value

　次の列を含む pands.DataFrame が出力されます。

- `term`（index）</br>
　説明変数の名称
- `unconditional`</br>
　Hoffmann, Kassouf(2005, p.6)の(14)式および(15)式に基づく条件付なしの平均限界効果（unconditional marginal effect）
- `conditional`</br>
　Hoffmann, Kassouf(2005, pp.4-5)の(8)式および(9)式に基づく条件付平均限界効果（conditional marginal effect）
- `selection`</br>
　Hoffmann, Kassouf(2005, p.6)の(14)式および(15)式の第3項に当たる間接効果
- `beta`</br>
　第2段階の regression equation の回帰係数
- `gamma`</br>
　第1段階の selection equation の回帰係数


## 使用例 Examples

　`heckit_helper` モジュールはヘックマンの2段階推定（Heckit）を実行を [`Py4Etrics`](https://github.com/Py4Etrics/py4etrics) モジュールの `py4etrics.heckit.Heckit()` に依存しているため、事前のインストールをお願いします。

```python
pip install git+https://github.com/Py4Etrics/py4etrics.git
```

ここでは `wooldridge` モジュールの `mroz` データを使い、春山(2023, Chap.24)のモデルを再現します。

```python
import pandas as pd
import wooldridge
from py4stats import heckit_helper

mroz = wooldridge.data('mroz') # サンプルデータの読み込み

mod_heckit, exog_outcome, exog_select = \
 heckit_helper.Heckit_from_formula(
    selection = 'lwage ~ educ + exper + expersq + nwifeinc + age + kidslt6 + kidsge6',
    outcome = 'lwage ~ educ + exper + expersq',
    data = mroz
)

res_heckit = mod_heckit.fit(cov_type_2 = 'HC1')
```

```python
print(heckit_helper.heckitmfx_compute(
    res_heckit,
    exog_select = exog_select,
    exog_outcome = exog_outcome
    ).round(4))
#>           unconditional  conditional  selection    beta   gamma
#> term                                                           
#> age             -0.0385       0.0010    -0.0395  0.0000 -0.0529
#> educ             0.2045       0.1067     0.0978  0.1091  0.1309
#> exper            0.1338       0.0417     0.0922  0.0439  0.1233
#> expersq         -0.0022      -0.0008    -0.0014 -0.0009 -0.0019
#> kidsge6          0.0263      -0.0006     0.0269  0.0000  0.0360
#> kidslt6         -0.6332       0.0157    -0.6489  0.0000 -0.8683
#> nwifeinc        -0.0088       0.0002    -0.0090  0.0000 -0.0120
```

被説明変数の `lwage` は対数賃金であるため、`exponentiate = True` として指数関数 $100[\exp(x - 1)]$ を使った変換を行うことで、限界効果を賃金の変化率として解釈できるようになります。

```python
print(heckit_helper.heckitmfx_compute(
    res_heckit,
    exog_select = exog_select,
    exog_outcome = exog_outcome,
    exponentiate = True
    ).round(4))
#>           unconditional  conditional  selection     beta   gamma
#> term                                                            
#> age             -3.7809       0.0954    -3.8725   0.0000 -0.0529
#> educ            22.6943      11.2606    10.2765  11.5235  0.1309
#> exper           14.3206       4.2543     9.6555   4.4865  0.1233
#> expersq         -0.2233      -0.0825    -0.1409  -0.0859 -0.0019
#> kidsge6          2.6604      -0.0649     2.7271   0.0000  0.0360
#> kidslt6        -46.9117       1.5782   -47.7365   0.0000 -0.8683
#> nwifeinc        -0.8730       0.0217    -0.8945   0.0000 -0.0120
```

## 注意

　`heckitmfx_compute()` の実装は実験的なものであり、 Stata における `margins` コマンドなどの既存の手法とは計算結果が一致しない可能性があります。

## 参考文献
- ダハナ・ウィラワン ドニ, 勝又壮太郎(2023) 『Rによるマーケティング・データ分析: 基礎から応用まで (ライブラリ データ分析への招待 4)』新世社.
- 春山鉄源 (2023) 『Pythonで学ぶ入門計量経済学』. https://py4etrics.github.io/index.html
- Hoffmann, Rodolfo, and Ana Lucia Kassouf. (2005). Deriving conditional and unconditional marginal effects in log earnings equations estimated by heckman’s procedure. *Applied Economics*, 37(11), 1303–1311.
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
