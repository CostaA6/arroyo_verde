#==============================================================
# Separacion de anomalias con filtro pasa bajos.
#==============================================================
#Librerias.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import griddata 
from scipy.interpolate import Rbf
#==============================================================
#Funciones para graficar:
import Graficas as gr
#==============================================================
#IMPORTACION DE MEDICIONES
d = np.loadtxt("input_G.txt")
#--------------------------------------------------------------
#MEDICIONES
x, y, gz  = d[:,18],d[:,19],d[:,17]         #Vectores de datos.
#==============================================================
#GRILLA REGULAR
nx,ny   = 50, 50                            #Numero de elementos
grid_x  = np.linspace(min(x), max(x), nx)   #Valores de X (GRILLA)
grid_y  = np.linspace(min(y), max(y), ny)   #Valores de Y (GRILLA)
X,Y     = np.meshgrid(grid_x, grid_y)       #GRILLA.
#--------------------------------------------------------------
#Visualizado de la grilla con las mediciones.
#plt.figure(1)
#gr.grilla(X,x,Y,y,gz,'Grilla')
#==============================================================
#INTERPOLACIÓN
GZ = griddata(points=(x,y), values=gz, xi=(X,Y), method='linear', fill_value=np.nan)
#--------------------------------------------------------------
rbf3 = Rbf(x, y, gz, function="linear", smooth=0.1)

xnew, ynew = np.meshgrid(grid_x, grid_y)
xnew = xnew.flatten()
ynew = ynew.flatten()

znew = rbf3(xnew, ynew)
GZbis = griddata(points=(xnew,ynew), values=znew, xi=(X,Y), method='linear', fill_value=np.nan)

cut = (GZ/GZ)

GZ = GZbis
#--------------------------------------------------------------
#Visualizado del mapa.
#plt.figure(2)
#gr.mapa(X,x,Y,y,GZ*cut,'Anomalia de Bouguer')
#==============================================================
#[NO SE USO LA VENTANA!!!]
#ventana = GZ/GZ
#ventana = pd.DataFrame(ventana)
#ventana.fillna(0, inplace=True)
#ventana = ventana.to_numpy()
##--------------------------------------------------------------
#img_new = np.zeros([nx, ny])
#
#mask = np.ones([3, 3], dtype = int) 
#mask = mask / 9
#
#for i in range(1, nx-1): 
#    for j in range(1, ny-1): 
#        temp = ventana[i-1, j-1]*mask[0, 0]+ventana[i-1, j]*mask[0, 1]+ventana[i-1, j + 1]*mask[0, 2]+ventana[i, j-1]*mask[1, 0]+ ventana[i, j]*mask[1, 1]+ventana[i, j + 1]*mask[1, 2]+ventana[i + 1, j-1]*mask[2, 0]+ventana[i + 1, j]*mask[2, 1]+ventana[i + 1, j + 1]*mask[2, 2] 
#         
#        img_new[i, j]= temp 
#
#ventana = img_new
#--------------------------------------------------------------
#Remplazo los NaN por ceros.
#df = pd.DataFrame(GZ)       #Convierto GZ en un df.
#df.fillna(0, inplace=True)  #Cambio los NaN con ceros.
#gz0 = df.to_numpy()         #Paso de un data frame a un array.
#
##gz0 = gz0*ventana           #Aplico la ventana a gz0

#==============================================================
#Calculo el espectro 2D.
size = 50  #2048
ft = np.fft.fft2(GZ,s=(size,size))  #Transformo y hago zeropadding.
#--------------------------------------------------------------
#Dominio de la ft:
kx = np.linspace(-np.pi, np.pi, size)
ky = np.linspace(-np.pi, np.pi, size)

kx,ky = np.meshgrid(kx, ky)
#--------------------------------------------------------------
#Ventana: Continuacion Ascendente.
#h  = 5
#kr = np.sqrt((kx**2)+(ky**2))
#W  = np.exp(-(kr)*h)
#W  = np.fft.fftshift(W)
#--------------------------------------------------------------
#Ventana: Butterworth.
n = 3
kc = np.pi/40 #25
kr = np.sqrt((kx**2)+(ky**2))
b = 1.0/np.sqrt(1.+(kr/kc)**n)
W = np.fft.fftshift(b)
#--------------------------------------------------------------
#Ventana: Derivada de orden n.
#n = 2
#kr = np.sqrt((kx**2)+(ky**2))
#W = kr**n
#--------------------------------------------------------------
#Aplicación del filtro.
ft_w = ft * W
#--------------------------------------------------------------
#Grafico de espectros y filtro.
plt.figure(3)
plt.subplot(131)
gr.espectro(np.real(np.fft.ifftshift(ft)),'Espectro')
plt.subplot(132)
gr.espectro(np.fft.ifftshift(W), 'Filtro (C.A.)')
plt.subplot(133)
gr.espectro(np.real(np.fft.ifftshift(ft_w)), 'Señal Filtrada')

#==============================================================
#Calculo de la ifft de ft_w

ift = np.real(np.fft.ifft2(ft_w))   #Antitransformo el espectro filtrado
ift = ift[0:nx, 0:ny]               #Recorto los ceros del zeropadding.
ift = ift.real                      #Me quedo solo con la parte real.

plt.figure(4)
plt.subplot(131)
gr.mapa(X,x,Y,y,GZ*cut,'Anomalia de Bouguer')
plt.subplot(132)
gr.mapa(X,x,Y,y,ift*cut,'Anomalia Regional')
plt.subplot(133)
gr.mapa(X,x,Y,y,(GZ-ift)*cut,'Anomalia Residual')

#==============================================================
#Graficos de testeo
#plt.figure(5)
#plt.subplot(132)
#gr.mapa(X,x,Y,y,cut, "Area de interes")
#plt.subplot(131)
#gr.mapa(X,x,Y,y,GZ, "Señal de Entrada")
#plt.subplot(133)
#gr.mapa(X,x,Y,y,GZ*cut, "Señal Acotada")
#==============================================================