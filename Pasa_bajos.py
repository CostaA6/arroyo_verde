#==============================================================
# Separacion de anomalias con filtro pasa bajos.
#==============================================================
#Librerias.
import matplotlib.pyplot as plt
import numpy as np
#-------------------------------------
import Graficas as gr
import ventanas as w
import Auxiliar as aux
import procesamiento as pg

#==============================================================
# LECTURA DE DATOS Y ARREGLOS DE NUMPY
d = np.loadtxt("input_G.txt")

x, y, gz  = d[:,18],d[:,19],d[:,17]         #Vectores de datos.
#==============================================================
# GRILLADO DEL DOMINIO ESPACIAL: X e Y.

nx, ny, X, Y     = pg.grilla(x,y,200,200)       #GRILLA.
#==============================================================
# Seccion sin extrapolar:
cut = aux.seccion(x,X,y,Y,gz)
#==============================================================
# INTERPOLACIÓN: GRILLA DE ANOMALIA DE BOUGUER:
#'multiquadric'|'inverse'|'gaussian'|'linear'|'cubic'|'quintic'|'thin_plate'

GZ  = pg.interpolacion(x,y,gz,nx,ny,'linear')
#==============================================================
# PREPROCESADO DE LA GRILLA DE ANOMALIA DE BOUGUER:
    
preprocesado = pg.preprocesado(GZ)
#==============================================================
# GRILLADO DEL DOMINIO DE NUMERO DE ONDA: KX y KY:

kxx, kyy = pg.grilla_TDF2(X,Y,preprocesado)
#==============================================================
# VENTANAS UTILIZADAS EN EL FILTRADO:

W , ventana = w.CA(kxx , kyy , 510)               # Continuacion Ascendente.
#W , ventana = w.BW(kxx , kyy , 10 , np.pi/3000)     # Butterworth.
#W , ventana = w.DN(kxx , kyy , 1)                 # Derivada orden n.
#W , ventana = w.GRAD(kxx , kyy)                 # Gradiente.

#==============================================================
# IMPLEMENTACION DEL FILTRO:
ift, ft, ft_w = pg.FPB_2D(kxx, kyy, W, preprocesado,nx,ny)

#==============================================================
#Grafico de espectros y filtro.

plt.figure(1)
plt.subplot(131)
gr.espectro(np.real(np.fft.ifftshift(ft)),'Espectro')
plt.subplot(132)
gr.espectro(np.fft.ifftshift(W), ventana)
plt.subplot(133)
gr.espectro(np.real(np.fft.ifftshift(ft_w)), 'Señal Filtrada')
plt.show()

#==============================================================
#Mapas de Anomalia:
plt.figure(2)
plt.subplot(131)
gr.mapa(X,x,Y,y,GZ*cut,'Anomalia de Bouguer')
plt.subplot(132)
gr.mapa(X,x,Y,y,ift*cut,'Anomalia Regional')
plt.subplot(133)
gr.mapa(X,x,Y,y,(GZ-ift)*cut,'Residuo')
plt.show()
#==============================================================