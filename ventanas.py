#==============================================================
import numpy as np
#==============================================================
#Ventana: Continuacion Ascendente.
def CA(kx,ky,h):
    kr = np.sqrt((kx**2)+(ky**2))
    W  = np.exp(-(kr)*h)
    #W  = np.fft.fftshift(W)
    nombre_filtro = "Ventana CAA"
    
    return W , nombre_filtro
#--------------------------------------------
#Ventana: Butterworth.
def BW(kx,ky,n,kc):
    kr = np.sqrt((kx**2)+(ky**2))
    W = 1.0/np.sqrt((1.+(kr/kc)**n))
    #W = np.fft.fftshift(b)
    nombre_filtro = "Ventana Butterworth"
    
    return W , nombre_filtro
#--------------------------------------------
#Ventana: Derivada de orden n.
def DN(kx,ky,n):
    kr = np.sqrt((kx**2)+(ky**2))
    W = kr**n
    nombre_filtro = "Derivada orden n"
    
    return W , nombre_filtro
#==============================================================
def GRAD(kx,ky):
    kr = np.sqrt((kx**2)+(ky**2))
    W = kr * kx * ky
    nombre_filtro = "Gradiente"
    
    return W , nombre_filtro
#==============================================================
#Espejado:
def espajado(GZ):
    CUAD1 = np.fliplr(GZ)
    CUAD2 = GZ
    CUAD3 = np.flipud(GZ)
    CUAD4 = np.flipud(CUAD1)

    preprocesado = np.concatenate((np.concatenate((CUAD3,CUAD4),axis=1),
                             np.concatenate((CUAD2,CUAD1),axis=1)), axis=0)
    
    return preprocesado
#==============================================================


