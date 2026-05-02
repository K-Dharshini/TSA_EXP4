# Ex.No: 4 FIT ARMA MODEL FOR TIME SERIES

### DATE: 

# AIM:
To implement ARMA model in Python.

# DATASET:
Global Child Illiteracy Dataset

# SOFTWARE REQUIRED:
Google Colab

# ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.
4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 1000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.
6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.
   
# PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv('/content/child_illiteracy_regional_trends.csv')

data = data.groupby('year')['avg_years_of_schooling'].mean().reset_index()

# Define sample size
N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

X = data['avg_years_of_schooling']

# Original Data Plot
plt.plot(X)
plt.title('Original Data')
plt.show()

# ACF & PACF
plt.subplot(2, 1, 1)
plot_acf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# ARMA(1,1)
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()

# ARMA(2,2)
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']

theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()
```

# OUTPUT:
## SIMULATED ARMA(1,1) PROCESS
<img width="708" height="378" alt="image" src="https://github.com/user-attachments/assets/57a2d750-bb04-45e1-b839-f3964a23b911" />

### Partial Autocorrelation
<img width="705" height="375" alt="image" src="https://github.com/user-attachments/assets/b7c66575-5a51-41da-93db-34830df79cbd" />

### Autocorrelation
<img width="708" height="371" alt="image" src="https://github.com/user-attachments/assets/f796013a-5504-46ad-88b6-3c8bd3fd347f" />

## SIMULATED ARMA(2,2) PROCESS
<img width="704" height="375" alt="image" src="https://github.com/user-attachments/assets/b594ac40-21b1-44ec-9a28-ca2f772540b9" />

### Partial Autocorrelation
<img width="708" height="369" alt="image" src="https://github.com/user-attachments/assets/00262403-32e3-4571-bb07-297c668847d6" />

### Autocorrelation
<img width="704" height="372" alt="image" src="https://github.com/user-attachments/assets/9ee0b95b-8efd-4155-a979-8a5beaf54f37" />


# RESULT:
Thus, the Python program is created to fit ARMA model successfully.
