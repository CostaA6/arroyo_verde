#==============================================================
#Cosmeticos
#==============================================================
#Librerias.
import matplotlib.pyplot as plt
import numpy as np
#-------------------------------------
import Graficas as gr
import Auxiliar as aux
import procesamiento as pg

import matplotlib.ticker as ticker
#==============================================================
#IMPORTACION DE MEDICIONES
d = np.loadtxt("input_G.txt")
#--------------------------------------------------------------
x, y, gz  = d[:,18],d[:,19],d[:,17]         #Vectores de datos.
#==============================================================
#GRILLA REGULAR
nx, ny, X, Y     = pg.grilla(x,y,200,200)       #GRILLA.
#==============================================================
#Seccion sin extrapolar:
cut = aux.seccion(x,X,y,Y,gz)


GZ  = pg.interpolacion(x,y,gz,nx,ny,'linear')

#==============================================================
# FALOPA:
# grid_size = 200
# cut=1
# x = np.linspace(-3, 3, grid_size)
# y = np.linspace(-3, 3, grid_size)
# X, Y = np.meshgrid(x, y)
# mu_x = 0
# mu_y = 0
# sigma = 1
# GZ = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - mu_x)**2 + (Y - mu_y)**2) / sigma**2)
#==============================================================

GZ_X = pg.DERIVADA_X(X,Y,GZ)[0]         # DERIVADA DE AB SEGUN X

GZ_Y = pg.DERIVADA_Y(X,Y,GZ)[0]         # DERIVADA DE AB SEGUN Y

GZ_Z1 = pg.DERIVADA_Zn(X,Y,GZ,1)[0]     # DERIVADA DE AB SEGUN Z

GZ_Z2 = pg.DERIVADA_Zn(X,Y,GZ,2)[0]     # DERIVADA SEGUNDA DE AB SEGUN Z

GH = np.sqrt(GZ_X**2+GZ_Y**2)           # GRADIENTE HORIZONTAL

SA = np.sqrt(GZ_X**2+GZ_Y**2+GZ_Z1**2)  # SEÑAL ANALITICA

# MAS FALOPA...
# SA = np.sqrt(np.abs(GZ_X**2+GZ_Y**2-GZ_Z1**2))  # SEÑAL ANALITICA

#===============================================================
#|||||||||||||||||||||||||| GRAFICOS ||||||||||||||||||||||||||#
#===============================================================
from matplotlib.colors import LinearSegmentedColormap
colors = [(1., 0.75, 0.875), (1., 0.5, 0.75), (1.0, 0.0, 0.5), (1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.6)][::-1]
cmap = LinearSegmentedColormap.from_list('mycmap', colors)

plt.figure(8)
#==============================================================
# GRADIENTE HORIZONTAL

plt.subplot(131)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

img = plt.imread('/home/andres/Escritorio/Codigo Gravimetria/IMGSAT.png')
    
im = plt.imshow(GH*cut,cmap=cmap, origin="lower",
           interpolation=None,
           extent=[np.min(x),np.max(x),np.min(y),np.max(y)],
           alpha=0.6)

cbar = plt.colorbar(im, orientation = 'vertical')
tick_font_size = 5
cbar.ax.tick_params(labelsize=tick_font_size)
plt.grid(True, color='black',linestyle='dotted')

plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('EJE X',fontsize=6)
plt.ylabel('EJE Y',fontsize=6)
plt.title('Gradiente Horizontal',fontsize=6)
plt.show()

#==============================================================
# SEÑAL ANALITICA

plt.subplot(132)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

im = plt.imshow(SA*cut,cmap=cmap, origin="lower",
           interpolation=None,
           extent=[np.min(x),np.max(x),np.min(y),np.max(y)],
           alpha=0.6)
    
cbar = plt.colorbar(im, orientation = 'vertical')
tick_font_size = 5
cbar.ax.tick_params(labelsize=tick_font_size)
#plt.clim(0 , 0.013)
    
plt.grid(True, color='black',linestyle='dotted')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('EJE X',fontsize=6)
plt.ylabel('EJE Y',fontsize=6)
plt.title('Señal Analitica',fontsize=6)
plt.show()

#==============================================================
# DERIVADA 2da VERTICAL.

plt.subplot(133)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

img = plt.imread('/home/andres/Escritorio/Codigo Gravimetria/IMGSAT.png')
plt.contour(X, Y, GZ_Z1*cut, 10, colors="black", linewidths=0.5)
im = plt.imshow(GZ_Z2*cut,cmap=cmap, origin="lower",
           interpolation=None,
           extent=[np.min(x),np.max(x),np.min(y),np.max(y)],
           alpha=0.6)
    
cbar = plt.colorbar(im, orientation = 'vertical')
tick_font_size = 4
cbar.ax.tick_params(labelsize=tick_font_size)

plt.clim(-0.00001 , 0.00001)
#plt.clim(-0.015 , 0.015)
    
plt.grid(True, color='black',linestyle='dotted')
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('EJE X',fontsize=6)
plt.ylabel('EJE Y',fontsize=6)
plt.title('Derivada segunda',fontsize=6)
plt.show()

#==============================================================