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
<img src="https://github.com/user-attachments/assets/42b6c560-af63-42a0-8c71-ec4a155aea3e" width=35% height=35%>
<img src="https://github.com/user-attachments/assets/ecbfb5ee-ff70-40af-87e9-af924aac4533" width=35% height=35%>
<img src="https://github.com/user-attachments/assets/9b312dc0-a807-43ea-850f-51c94c766115" width=35% height=35%>
<img src="https://github.com/user-attachments/assets/10cbcca4-cd4b-4ce1-89c0-df738a76ec6a" width=35% height=35%>
<img src="https://github.com/user-attachments/assets/383b06e5-9a80-43ac-a2ac-16db882fa157" width=35% height=35%>


## Next Features
Next features to be added are: 
* add the simulation process. 

## Contact
If you want to contact me you can reach me at juanidinaro@gmail.com.
