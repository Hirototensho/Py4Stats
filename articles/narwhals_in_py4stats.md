# Technical Notes: py4stats.eda_tools における narwhals ベースの実装

## 概要

`py4stats.eda_tools` モジュールは、複数の DataFrame バックエンドに対して共通の API を提供することを目的として、`narwhals` ライブラリを用いて実装されています。

本ドキュメントでは、本モジュールの内部実装に関する前提条件や、バックエンドの違いに起因する挙動上の注意点について説明します。

通常の利用にあたって本ドキュメントを読む必要はありませんが、実装の詳細や挙動の違いが気になる場合には参考にしてください。

## 対応している DataFrame バックエンドについて

　`py4stats.eda_tools` モジュールの関数は、第一引数として `narwhals.from_native()` によって `nw.DataFrame` 型へ変換可能な DataFrame オブジェクトを受け取ります。

具体的には、以下のようなバックエンドを想定しています。

- `pandas.DataFrame`（主に動作検証を行っているバックエンド）
- `polars.DataFrame`（簡易的な動作確認のみ）
- `pyarrow.Table`（簡易的な動作確認のみ）

本ライブラリの動作確認は、基本的に `pandas.DataFrame` を用いて実施しています。そのため、`polars` や `pyarrow` を使用した場合には、バックエンド固有の仕様差や未検証の挙動により、一部の関数でエラーが発生する可能性があります。

そのような挙動が確認された場合は、Issue 等での報告を歓迎します。

## narwhals を用いた実装方針について

　内部実装では、関数の冒頭で

``` python
nw.from_native(data)
```

を用いて入力データを nw.DataFrame に変換し、以降の処理を narwhals の抽象 API 上で行っています。

この設計により、DataFrame バックエンドごとの差異を最小限に抑えつつ、将来的な拡張性を確保することを目的としています。

一方で、narwhals は各バックエンドの完全な互換性を保証するものではないため、特定の操作や型変換についてはバックエンドごとに挙動が異なる場合があります。

## 　`pandas_flavor` を用いた DataFrame メソッド登録について

`py4stats.eda_tools` の関数のうち、単一の `DataFrame` オブジェクトを引数として受け取る関数については、`pandas_flavor.register_dataframe_method` を用いて DataFrame メソッドとして登録されています。その結果、以下のような使い方が可能です。

``` python
df.diagnose()
```
ただし、`pandas_flavor` は pandas の拡張を前提とした仕組みであるため、このメソッド形式の呼び出しは、`pandas.DataFrame` を対象としています。
　polars.DataFrame や pyarrow ベースのオブジェクトを使用する場合には、関数として直接呼び出す形での利用を推奨します。

``` python
import py4stats as py4st

py4st.diagnose(df)
```

## 今後について

　`py4stats.eda_tools` モジュールは、今後も narwhals ベースの実装を主軸として
改良・拡張を行っていく予定です。
　一方で、従来の pandas ベース実装についても、互換性や参照用の実装として当面は保持される予定です。バックエンドごとの挙動差や制限事項については、必要に応じて本ドキュメントを更新していきます。