%matplotlib inline
!pip install pyfftw
import numpy as np
import matplotlib.pyplot as plt
from pyfftw.interfaces.numpy_fft import fft, ifft, irfft2, rfft2
import pyfftw
pyfftw.interfaces.cache.enable()
# Number of samplepoints
N = 500
# sample spacing
T = 1.0 / 200.0
x = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

# yf = scipy.fftpack.fft(y)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf))
plt.show()
inv=ifft(yf)
res = np.round(y-inv,10)

if np.count_nonzero(res) > 0:
    print("Something wrong")
else:
    print("FFT and IFFT works perfect")
