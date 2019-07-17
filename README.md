# PathLoss Characteristic

This project aims for improving quality(Error Rate - RMSE) of **existing linear regression model**[^1] by applying variety of optimizing(machine learning) modeling techniques. <br>

## The wireless radio propagation channel 
The object function of the modified Hata Model
\begin{eqnarray}
L_{p}(d)[dB] = L_p(d_0) + 10nlog_{10}(d/d_{0}) + X_{\sigma}
\end{eqnarray}
which is designed for  the 1500-2000 MHz frequency range.

## The modified Hata model [^1]
The new path loss model that is revised from the modified Hata model [^1].
\begin{eqnarray}
L_p(d) &=& A + B + (C + \bigtriangleup_1)log_{10}(d) + D + \bigtriangleup_2 \\
A &=& 46.3 + 33.9log_{10}(f) - 13.28log_{10}(h_t) \\
B &=& -3.2 log_{10}(11.75h_r)^2 + 4.97 \\
C &=& 44.9 - 6.55log_{10}(h_t) \\
D &=& 0 \quad\text{(for suburban areas)}
\end{eqnarray}

## Linear Regression (Ridge, Lasso, etc)

## Calibrating terms in modified Hata model using multivariate linear regression

## ANN Linear Regression

## Gaussian Process Regression


[^1]: Han-Shin Jo, and Jong-Gwan Yook. “Path Loss Characteristics for IMT-Advanced Systems in Residential and Street Environments.” IEEE Antennas and Wireless Propagation Letters 9 (2010): 867–871. Web.
