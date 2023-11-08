#==============================================================
# Separacion de anomalias con ajuste polinomial.
#==============================================================
#Librerias.
import matplotlib.pyplot as plt
import numpy as np

#Funciones para graficar:
import Graficas as gr
import Auxiliar as aux
import procesamiento as pg
#==============================================================
#Importamos los datos.
d = np.loadtxt("input_G.txt")
#--------------------------------------------------------------
#MEDICIONES
x, y, gz  = d[:,18],d[:,19],d[:,17]
#==============================================================
#Grillado:

nx, ny, X, Y = pg.grilla(x,y,200,200)

#Seccion sin extrapolar:
cut = aux.seccion(x,X,y,Y,gz)

#Interpolación:
GZ  = pg.interpolacion(x,y,gz,nx,ny,'linear')
#==============================================================
#Ajuste polinomial de grado 1:

AP1_REG, AP1_RES = pg.ajuste_polinomial_1(nx,ny,x,y,gz,GZ)
#==============================================================
#Ajuste polinomial de grado 2:

AP2_REG, AP2_RES = pg.ajuste_polinomial_2(nx,ny,x,y,gz,GZ)
#==============================================================
plt.figure(1)
#plt.subplot(231)
#gr.mapa(X,x,Y,y,GZ*cut,'Anomalia de Bouguer')
plt.subplot(221)
gr.mapa(X,x,Y,y,AP1_REG*cut,'Anomalia Regional (Orden 1)')
plt.subplot(222)
gr.mapa(X,x,Y,y,AP1_RES*cut,'Anomalia Residual')
#plt.subplot(222)
#gr.mapa(X,x,Y,y,GZ*cut,'Anomalia de Bouguer')
plt.subplot(223)
gr.mapa(X,x,Y,y,AP2_REG*cut,'Anomalia Regional (Orden 2)')
plt.subplot(224)
gr.mapa(X,x,Y,y,AP2_RES*cut,'Anomalia Residual')
#==============================================================