# Sovereign-Debt-without-Exclusivity
Using Quantecon's [solver](https://python-advanced.quantecon.org/arellano.html) for the Arellano Economy, I've included an upgraded Python solver of a Sovereign debt model with one period and long period bonds, where lenders do not have exclusivity to the governments, thus they must forecast long bond prices.

### Model Setup

Government maximizes the constrained lifetime utility choosing optimal maturity of debt $b'_s$ and $b'_l$, and whether to default tomorrow, given prices.
```math
V^R\left(b_S, b_L, y\right)=\underset{c, b_S^{\prime}, b_L^{\prime}}{Max}\left\{u(c)+\beta E\left[V\left(b_S^{\prime}, b_L^{\prime}, y^{\prime}\right) \mid y\right]\right\}
```
```math
\text { s.t. } \quad c=y-b_S-\delta b_L+q_S\left(b_S^{\prime}, b_L^{\prime}, y\right) b_S^{\prime}+q_L\left(b_S^{\prime}, b_L^{\prime}, y\right)\left(b_L^{\prime}-(1-\delta) b_L\right)
```

```math
u^{\prime}(c) q_S\left(b_S^{\prime}, b_L^{\prime}, y\right)+\beta E\left[V_1\left(b_S^{\prime}, b_L^{\prime}, y^{\prime}\right) \mid y\right]=0
```
```math
u^{\prime}(c) q_L\left(b_S^{\prime}, b_L^{\prime}, y\right)+\beta E\left[V_2\left(b_S^{\prime}, b_L^{\prime}, y^{\prime}\right) \mid y\right]=0
```
```math
q_S\left(b_S^{\prime}, b_L^{\prime}, y\right)=\frac{1}{1+r} E\left[1-\hat{d}\left(b_S^{\prime}, b_L^{\prime}, y^{\prime}\right) \mid y\right]
```
```math
q_L\left(b_S^{\prime}, b_L^{\prime}, y\right)=\frac{1}{1+r} E\left[\left(1-\hat{d}\left(b_S^{\prime}, b_L^{\prime}, y^{\prime}\right)\right)\left(\delta+(1-\delta) q_L\left(\hat{b}_S\left(b_S^{\prime}, b_L^{\prime}, y^{\prime}\right), \hat{b}_L\left(b_S^{\prime}, b_L^{\prime}, y^{\prime}\right), y^{\prime}\right)\right) \mid y\right]
```



#### Optimality Condition

The optimal maturity of debt is given by the following optimality condition that arises from constrained lifetime utility maximization, which implies that expected marginal rate of substitution equals the ratio of prices:

```math
    \underbrace{\frac{\mathbb{E}\left[V_{b'_s}(b'_s, b'_l,y')\bigg| y \right]}{\mathbb{E}\left[V_{b'_l}(b'_s, b'_l,y')\bigg| y\right]} }_{\textbf{$\mathbb{E}\left[MRS_{b'_{s},b'_{l}}\bigg| y \right]$}} 
    = 
    \underbrace{\frac{q_s(b'_s, b'_l,y)}{q_l(b'_s, b'_l,y)}}_{\text{Price Ratio}}
```

## Prerequisites
Before you begin, ensure you have met the following requirements:
* You have installed Python 3 or higher.
* [Numpy](https://numpy.org/) for Python. 
 ```
      pip install numpy
 ```
 or just get the [Anaconda](https://docs.continuum.io/anaconda/) package where Numpy (among other nice packages) is already installed.
 * [Matplotlib](https://matplotlib.org/stable/) for Python. 
 ```
      pip install matplotlib
 ```
 or just get the [Anaconda](https://docs.continuum.io/anaconda/) package where Matplotlib (among other nice packages) is already installed.
 * [Quantecon](https://quanteconpy.readthedocs.io/en/latest/) for discretizing the AR(1) process of income, which is needed for building its grid.
   ```
      pip install quantecon
   ```
* [Numba](https://numba.pydata.org/) for optimizing the VFI process.
  ```
      pip install numba
  ```
 ## Installing the Solver

To install the Sovler, just download SL.py, and open with your edit IDLE to configure and execute it.

## Using SL.py

The VFI algorithm works efficiently. One could use it with different values of parameters. Output examples are below:
<img src="https://github.com/user-attachments/assets/d10a6182-815b-4854-8cfc-1e185d65857c" width=35% height=35%>
<img src="https://github.com/user-attachments/assets/d0e43743-35b7-4759-86e0-6d2560ef1117" width=35% height=35%>
<img src="https://github.com/user-attachments/assets/34a4bdc9-1858-4aae-a3c9-a880ab2f8fa8" width=35% height=35%>
<img src="https://github.com/user-attachments/assets/63aee0c1-b0e0-4da9-8445-33dba986c86e" width=35% height=35%>
<img src="https://github.com/user-attachments/assets/42339d51-c37d-4821-9463-50884243db5a" width=35% height=35%>


## Next Features
Next features to be added are: 
* add the simulation process. 

## Contact
If you want to contact me you can reach me at juanidinaro@gmail.com.
