| functions            | Input            | Pandas   | Polars   | Pyarrow   | 補足                                       |
|:---------------------|:-----------------|:---------|:---------|:----------|:-------------------------------------------|
| Max                  | pd.Series        | ✅       | ❌       | ❌        | pd.DataFrame.eval() での使用を想定した関数 |
| Mean                 | pd.Series        | ✅       | ❌       | ❌        | pd.DataFrame.eval() での使用を想定した関数 |
| Median               | pd.Series        | ✅       | ❌       | ❌        | pd.DataFrame.eval() での使用を想定した関数 |
| Min                  | pd.Series        | ✅       | ❌       | ❌        | pd.DataFrame.eval() での使用を想定した関数 |
| Pareto_plot          | DataFrame        | ✅       | ✅       | ✅        | nan                                        |
| Sum                  | pd.Series        | ✅       | ❌       | ❌        | pd.DataFrame.eval() での使用を想定した関数 |
| check_that           | DataFrame        | ✅       | ⭕️       | ⭕️        | Pandas 依存の実装                          |
| check_viorate        | DataFrame        | ✅       | ⭕️       | ⭕️        | Pandas 依存の実装                          |
| compare_df_cols      | DataFrame        | ✅       | ✅       | ✅        | nan                                        |
| compare_df_record    | DataFrame        | ✅       | ✅       | ✅        | nan                                        |
| compare_df_stats     | DataFrame        | ✅       | ✅       | ✅        | nan                                        |
| compare_group_means  | DataFrame        | ✅       | ✅       | ✅        | nan                                        |
| compare_group_median | DataFrame        | ✅       | ✅       | ✅        | nan                                        |
| crosstab             | DataFrame        | ✅       | ✅       | ⭕️        | Pyarrow は Polars 依存の実装               |
| diagnose             | DataFrame        | ✅       | ✅       | ✅        | nan                                        |
| diagnose_category    | DataFrame        | ✅       | ✅       | ✅        | nan                                        |
| filtering_out        | DataFrame        | ✅       | ✅       | ✅        | nan                                        |
| freq_table           | DataFrame        | ✅       | ✅       | ✅        | nan                                        |
| implies_exper        | pd.Series        | ✅       | ❌       | ❌        | pd.DataFrame.eval() での使用を想定した関数 |
| is_dummy             | DataFrame/Series | ✅       | 🔼       | 🔼        | nan                                        |
| is_number            | Series           | ✅       | 🔼       | 🔼        | nan                                        |
| is_ymd_like          | Series           | ✅       | 🔼       | 🔼        | nan                                        |
| is_ymd               | Series           | ✅       | 🔼       | 🔼        | nan                                        |
| mean_ci              | DataFrame/Series | ✅       | ✅       | ✅        | nan                                        |
| mean_qi              | DataFrame/Series | ✅       | ✅       | ✅        | nan                                        |
| median_qi            | DataFrame/Series | ✅       | ✅       | ✅        | nan                                        |
| min_max              | Series           | 🔼       | 🔼       | 🔼        | nan                                        |
| plot_mean_diff       | DataFrame        | 🔼       | 🔼       | 🔼        | nan                                        |
| plot_median_diff     | DataFrame        | 🔼       | 🔼       | 🔼        | nan                                        |
| plot_miss_var        | DataFrame        | 🔼       | 🔼       | 🔼        | nan                                        |
| scale                | Series           | 🔼       | 🔼       | 🔼        | nan                                        |
| remove_constant      | DataFrame        | ✅       | ✅       | ✅        | nan                                        |
| remove_empty         | DataFrame        | ✅       | ✅       | ✅        | nan                                        |
| tabyl                | DataFrame        | ✅       | ✅       | ⭕️        | Pyarrow は Polars 依存の実装               |
| weighted_mean        | Series           | 🔼       | 🔼       | 🔼        | nan                                        |