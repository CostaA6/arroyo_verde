#==============================================================
# Mapeado de los datos.
#==============================================================
#Librerias.
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata 

#Funciones para graficar:
import Graficas as gr
import Auxiliar as aux
#==============================================================
#Importamos los datos.
d = np.loadtxt("Grav_Spectrum_completo.xyz")
#--------------------------------------------------------------
#MEDICIONES
x, y = d[:,0],d[:,1]

#cota  = d[:,6]          #cota.
abc   = d[:,2]         #Anomalia de Bouguer.
#lec   = d[:,8]          #Lectura del gravimetro.
#g_obs = d[:,13]         #Gravedad Observada.
#c_al  = d[:,15]         #Correccion de aire libre.
#c_b   = d[:,16]         #Correccion de Bouguer.
#--------------------------------------------------------------
nx,ny   = 200, 200
grid_x  = np.linspace(min(x), max(x), nx)
grid_y  = np.linspace(min(y), max(y), ny)
X,Y     = np.meshgrid(grid_x, grid_y)
#--------------------------------------------------------------
cut = aux.seccion(x,X,y,Y,abc)
#Interpolacion de los datos:
#COTA  = aux.interpolacion(x,y,cota ,nx,ny,'linear') * cut
ABC   = aux.interpolacion(x,y,abc ,nx,ny,'linear') #* cut
#LEC   = aux.interpolacion(x,y,lec ,nx,ny,'linear') * cut
#G_OBS = aux.interpolacion(x,y,g_obs ,nx,ny,'linear') * cut
#C_AL  = aux.interpolacion(x,y,c_al ,nx,ny,'linear') * cut
#C_B   = aux.interpolacion(x,y,c_b ,nx,ny,'linear') * cut

#==============================================================
#Mapas:
plt.figure(1)
# plt.subplot(231)
# gr.mapa(X,x,Y,y,COTA,'Cota')
# plt.subplot(232)
# gr.mapa(X,x,Y,y,LEC,'Lectura')
# plt.subplot(233)
# gr.mapa(X,x,Y,y,G_OBS,'g Observada')
# plt.subplot(234)
# gr.mapa(X,x,Y,y,C_AL,'Correccion de Aire libre')
# plt.subplot(235)
# gr.mapa(X,x,Y,y,C_B,'Correccion de Bouguer')
# plt.subplot(236)
gr.mapa(X,x,Y,y,ABC,'Anomalia de Bouguer')
#==============================================================