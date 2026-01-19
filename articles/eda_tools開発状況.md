## eda_toolsの開発状況
2026年1月17日

**eda_toolsの開発状況**
| functions            | Input            | Pandas   | Polars   | Pyarrow   | 補足                                       |
|:---------------------|:-----------------|:---------|:---------|:----------|:-------------------------------------------|
| Mean                 | pd.Series        | ✅       | ❌       | ❌        | pd.DataFrame.eval() での使用を想定した関数     |
| Median               | pd.Series        | ✅       | ❌       | ❌        | pd.DataFrame.eval() での使用を想定した関数     |
| Max                  | pd.Series        | ✅       | ❌       | ❌        | pd.DataFrame.eval() での使用を想定した関数     |
| Min                  | pd.Series        | ✅       | ❌       | ❌        | pd.DataFrame.eval() での使用を想定した関数     |
| Pareto_plot          | DataFrame        | ✅       | ✅       | ✅        |                                            |
| Sum                  | pd.Series        | ✅       | ❌       | ❌        | pd.DataFrame.eval() での使用を想定した関数     |
| check_that           | DataFrame        | ✅       | ⭕️       | ⭕️        | 実装に pd.DataFrame.eval() を使用            |
| check_viorate        | DataFrame        | ✅       | ⭕️       | ⭕️        | 実装に pd.DataFrame.eval() を使用            |
| compare_df_cols      | DataFrame        | ✅       | ✅       | ✅        |                                            |
| compare_df_record    | DataFrame        | ✅       | ✅       | ✅        |                                            |
| compare_df_stats     | DataFrame        | ✅       | ✅       | ✅        |                                            |
| compare_group_means  | DataFrame        | ✅       | ⭕️       | ⭕️        | 出力が pd.DataFram に限定                    |
| compare_group_median | DataFrame        | ✅       | ⭕️       | ⭕️        | 出力が pd.DataFram に限定                    |
| crosstab             | DataFrame        | ✅       | ✅       | ⭕️        | Pyarrow は Polars 依存の実装                 |
| diagnose             | DataFrame        | ✅       | ✅       | ✅        |                                            |
| diagnose_category    | DataFrame        | ✅       | ✅       | ✅        |                                            |
| filtering_out        | DataFrame        | ✅       | ✅       | ✅        |                                            |
| freq_table           | DataFrame        | ✅       | ✅       | ✅        |                                            |
| implies_exper        | pd.Series        | ✅       | ❌       | ❌        | pd.DataFrame.eval() での使用を想定した関数     |
| is_dummy             | DataFrame/Series | ✅       | 🔼       | 🔼        |                                            |
| is_number            | Series           | ✅       | ✅       | ✅        |                                            |
| is_ymd_like          | Series           | ✅       | ✅       | ✅        |                                            |
| is_ymd               | Series           | ✅       | ✅       | ✅        |                                            |
| mean_ci              | DataFrame/Series | ✅       | ✅       | ✅        |                                            |
| mean_qi              | DataFrame/Series | ✅       | ✅       | ✅        |                                            |
| median_qi            | DataFrame/Series | ✅       | ✅       | ✅        |                                            |
| min_max              | Series           | ✅       | ✅       | ✅        |                                            |
| plot_mean_diff       | DataFrame        | ✅       | ✅       | ✅        |                                            |
| plot_median_diff     | DataFrame        | ✅       | ✅       | ✅        |                                            |
| plot_miss_var        | DataFrame        | ✅       | ✅       | ✅        |                                            |
| scale                | Series           | ✅       | ✅       | ✅        |                                            |
| remove_constant      | DataFrame        | ✅       | ✅       | ✅        |                                            |
| remove_empty         | DataFrame        | ✅       | ✅       | ✅        |                                            |
| tabyl                | DataFrame        | ✅       | ⭕️       | ⭕️        | 一部の処理が Pandas 依存                      |
| weighted_mean        | Series           | ✅       | ✅       | ✅        |                                            |

## 凡例

- ✅ 実装済/テスト済
- ⭕️ 実装済/テスト済（特定のバックエンドに依存）
- 🔼 実装済/テスト未
- ❌ 未実装