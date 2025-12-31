# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name = 'py4stats',  # パッケージ名（pip listで表示される）
    version = "0.0.1",  # バージョン
    description = 'simple tools for regression analisys',  # 説明
    author='Hiroto Tensho',  # 作者名
    packages = find_packages(),  # 使うモジュール一覧を指定する
    license='MIT',  # ライセンス
)