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


# Preparing datasets
Download raw Treasury bond data from WRDS using (notebook)[...], (KR yields)[https://www.discount-bond-data.org/].
Run the following code to preprocess data, and get supplement data needed to estimate returns.
```python
python get_riskfree.py
```

# Estimate returns

# Example to get started


# Suggested citation
Filipovic, Damir and Pelger, Markus and Ye, Ye, Shrinking the Term Structure (August 4, 2022). Swiss Finance Institute Research Paper No. 61, 2022, Available at SSRN: https://ssrn.com/abstract=4182649 or http://dx.doi.org/10.2139/ssrn.4182649
