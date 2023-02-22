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

# Preparing datasets
1. **Risk free rates**: Download `Yields at daily frequency for daily maturities` from [Discount Bond Database](https://www.discount-bond-data.org/) under the Yields section and make sure it is saved as `yield_panel_daily_frequency_daily_maturity.csv`. \
Run the following code or run [data_supplement/generate_kernel_matrices.ipynb](https://github.com/rosieiiiii/KR_return_example/blob/main/data_supplement/compile_daily_riskfree.ipynb) to obtain daily risk free rates implied by KR.
```python
python3 ./source/get_riskfree.py 
```
2. **Bond cashflows and prices**: Download and process raw Treasury bond data from WRDS using (notebook)[...], . Note that a WRDS account is needed to access CRSP data through WRDS. \
Filters applied:
   * Only includes fully taxable, non-callable, and non-flower bond issues.
   * Certificates of deposit are excluded (ITYPE=3)
   * Remove issues whose time series of prices terminate because of ''all exchanged'' (IWHY=3)

# Estimate returns


# Factors


# Complexity measures


# Suggested citation
Filipovic, Damir and Pelger, Markus and Ye, Ye, Shrinking the Term Structure (August 4, 2022). Swiss Finance Institute Research Paper No. 61, 2022, Available at SSRN: https://ssrn.com/abstract=4182649 or http://dx.doi.org/10.2139/ssrn.4182649
