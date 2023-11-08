#==============================================================
# Separacion de anomalias con filtro pasa bajos.
#==============================================================
#Librerias.
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import griddata 
from scipy.interpolate import Rbf
#==============================================================
#Funciones para graficar:
import Graficas as gr
import ventanas as w
#==============================================================
#IMPORTACION DE MEDICIONES
d = np.loadtxt("input_G.txt")
#--------------------------------------------------------------
#MEDICIONES
x, y, gz  = d[:,18],d[:,19],d[:,17]         #Vectores de datos.
#==============================================================
#GRILLA REGULAR
nx,ny   = 200, 200                            #Numero de elementos
grid_x  = np.linspace(min(x), max(x), nx)   #Valores de X (GRILLA)
grid_y  = np.linspace(min(y), max(y), ny)   #Valores de Y (GRILLA)
X,Y     = np.meshgrid(grid_x, grid_y)       #GRILLA.
#==============================================================
#Seccion:
sec = griddata(points=(x,y), values=gz, xi=(X,Y), method='linear', fill_value=np.nan)
cut = sec/sec
#==============================================================
#INTERPOLACIÓN
#'multiquadric'|'inverse'|'gaussian'|'linear'|'cubic'|'quintic'|'thin_plate'

rbf3 = Rbf(x, y, gz, function="linear", smooth=0.1)

xnew, ynew = np.meshgrid(grid_x, grid_y)
xnew = xnew.flatten()
ynew = ynew.flatten()

znew = rbf3(xnew, ynew)
GZ = griddata(points=(xnew,ynew), values=znew, xi=(X,Y), method='linear', fill_value=np.nan)

#==============================================================
#Calculo el espectro 2D.
size = 200
ft = np.fft.fft2(GZ,s=(size,size))  #Transformo y hago zeropadding.
#--------------------------------------------------------------
#Dominio de la ft:

#kx = np.linspace(-np.pi/2, np.pi/2, size)      #[PROBLEMA ACA!!!!]
#ky = np.linspace(-np.pi/2, np.pi/2, size)

#kx = (2*np.pi)*np.fft.fftfreq(nx)
#ky = (2*np.pi)*np.fft.fftfreq(ny)          

dx=grid_x[1]-grid_x[2]
dy=grid_y[1]-grid_y[2]

kx_adimensional = (np.pi)*np.fft.fftfreq(nx) # [-]
kx              = kx_adimensional / dx          # [1/longitud]
ky_adimensional = (np.pi)*np.fft.fftfreq(ny) # [-]
ky              = ky_adimensional / dy          # [1/longitud]
   

kxx,kyy = np.meshgrid(kx,ky)

kr = np.sqrt((kxx)**2+(kyy)**2)

#--------------------------------------------------------------
#Ventanas:

W , ventana = w.CA(kxx , kyy , 600)               # Continuacion Ascendente.
#W , ventana = w.BW(kxx , kyy , 3 , np.pi/500)     # Butterworth.
#W , ventana = w.DN(kxx , kyy , 1)                 # Derivada orden n.

#--------------------------------------------------------------
#Aplicación del filtro.
ft_w = ft * W
#--------------------------------------------------------------
#Grafico de espectros y filtro.
plt.figure(3)
plt.subplot(131)
gr.espectro(np.real(np.fft.ifftshift(ft)),'Espectro')
plt.subplot(132)
gr.espectro(np.fft.ifftshift(W), ventana)
plt.subplot(133)
gr.espectro(np.real(np.fft.ifftshift(ft_w)), 'Señal Filtrada')

#==============================================================
#Calculo de la ifft de ft_w

ift = np.real(np.fft.ifft2(ft_w))   #Antitransformo el espectro filtrado
ift = ift[0:nx, 0:ny]               #Recorto los ceros del zeropadding.
ift = ift.real                      #Me quedo solo con la parte real.
cut=1
plt.figure(4)
plt.subplot(131)
gr.mapa(X,x,Y,y,GZ*cut,'Anomalia de Bouguer')
plt.subplot(132)
gr.mapa(X,x,Y,y,ift*cut,'Anomalia Regional')
plt.subplot(133)
gr.mapa(X,x,Y,y,(GZ-ift)*cut,'Residuo')
#==============================================================