# Sovereign-Debt-without-Exclusivity
Using Quantecon's [solver](https://python-advanced.quantecon.org/arellano.html) for the Arellano Economy, I've included an upgraded Python solver of a Sovereign debt model with one period and long period bonds.

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
