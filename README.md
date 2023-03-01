# Kernel Ridge (KR) Model for Bond Return Estimation

**Shrinking the Term Structure** \
Damir Filipovi, Markus Pelger, Ye Ye \
Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4182649

Abstract: *We develop a conditional factor model for the term structure of treasury bonds, which unifies
non-parametric curve estimation with cross-sectional asset pricing. Our factors correspond to
the optimal non-parametric basis functions spanning the discount curve. They are investable
portfolios estimated with cross-sectional ridge regressions and derived from economic first principles. Empirically, we show that four factors explain the discount bond excess return curve
and term structure premium. Cash flows are covariances, as cash flows of coupon bonds fully
explain the factor exposure. The term structure premium depends on the market complexity
measured by the time-varying importance of higher order factors.*

# Example to get started
[KR_return_example.ipynb](https://github.com/rosieiiiii/KR_return_example/blob/main/KR_return_example.ipynb) provides an example of using the KR model for discount bond return estimation using formatted U.S. Treasury data on two example dates. This notebook can be run directly on Google Colab without having to download data to a local machine.

The figure below shows the fitted excess return of underlying securities and fitted excess return curve on 2013-12-31 relative to the next business day, 2014-01-02. 
![2013_rx](https://github.com/rosieiiiii/KR_return_example/blob/main/data_supplement/2013_excess_return.png)

# Preparing datasets
1. **Risk free rates** and **discount curve**: Download `Yields at daily frequency for daily maturities` from [Discount Bond Database](https://www.discount-bond-data.org/) under the Yields section and make sure it is saved as `yield_panel_daily_frequency_daily_maturity.csv`. \
Run the following code or run [compile_daily_riskfree.ipynb](https://github.com/rosieiiiii/KR_return_example/blob/main/data_supplement/compile_daily_riskfree.ipynb) to obtain daily discount curve and risk free rates.
```python
python3 ./source/get_g_and_riskfree.py 
```
2. **Bond cashflows and prices**: Download and select raw Treasury bond data from WRDS using [get_and_select_raw_data.ipynb](https://github.com/rosieiiiii/KR_return_example/blob/main/get_and_select_raw_data.ipynb), then construct bond cashflow matrices and price vectors in [construct_input_price_and_cashflow.ipynb](https://github.com/rosieiiiii/KR_return_example/blob/main/construct_input_price_and_cashflow.ipynb). Note that a WRDS account is needed to access CRSP data through WRDS. Generated cashflow matrices are saved in a compressed form (`.npz`) separately for each date, while price vectors are exported in a dataframe with date index.\
Filters applied:
   * Only includes fully taxable, non-callable, and non-flower bond issues.
   * Certificates of deposit are excluded (ITYPE=3).
   * Remove issues whose time series of prices terminate because of ''all exchanged'' (IWHY=3).
   * Follow Gurkaynak, Sack, and Wright (2007) and Liu and Wu (2021) and exclude the two most recently issued securities with maturities of 2, 3, 4, 5, 7 and 10 years for securities issued in 1980 or later.
   * Retain only securities whose prices on the subsequent business day are available (this is because our analysis requires daily returns of securities).
   
3. **Maturity filter**, **duration** and **yield to maturity**: Run [calculate_duration_and_ytm.ipynb](https://github.com/rosieiiiii/KR_return_example/blob/main/mask/calculate_duration_and_ytm.ipynb). Here, we obtain bond duration, yield to maturity and filter that removes securities maturing within 90 days for the fitting process ex-ante.

# Estimate returns
Use KR model to estimate discount bond returns on multiple dates ([run_multiple_dates.ipynb](https://github.com/rosieiiiii/KR_return_example/blob/main/run_multiple_dates.ipynb)). \
Or run the following code for a large number of dates. Results for each date are saved separately and can be compiled in [read_ret_curve.ipynb](https://github.com/rosieiiiii/KR_return_example/blob/main/KR_ret_models/read_ret_curve.ipynb).
```
sh ./source/run_ret_daily.sh
```

# KR Factors


# Complexity measures
In the paper, we introduce two novel measures for the state of the bond market.
* **IT-VOL**: The Idiosyncratic Treasury Volatility (IT-VOL) measures the idiosyncratic volatility normalized by the overall volatility. It captures how hard it is to explain the observed bond returns even with a flexible model.
![IT-VOL](https://github.com/rosieiiiii/KR_return_example/blob/main/data_supplement/IT_VOL.png)
* **T-COM**: The Treasury Market Complexity (T-COM) measures the complexity of the bond market. It captures how much variation is explained by higher order term structure factors. 
![T-COM](https://github.com/rosieiiiii/KR_return_example/blob/main/data_supplement/T-COM.png)
Formally, we define IT-VOL as the percentage of unexplained variation by the KR-4 factor model, and T-COM is the difference between the explained variation with KR-1 and KR-4 factor. The unexplained cross-sectional variation of a factor model is normalized by the overall cross-sectional variation on that day. The figures above show the 3-month moving average of the two market condition measures.



# Suggested citation
Filipovic, Damir and Pelger, Markus and Ye, Ye, Shrinking the Term Structure (August 4, 2022). Swiss Finance Institute Research Paper No. 61, 2022, Available at SSRN: https://ssrn.com/abstract=4182649 or http://dx.doi.org/10.2139/ssrn.4182649
