#--------------------------------------------------------------------
# PROCESAMIENTO
#--------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata 
from scipy.interpolate import Rbf
#--------------------------------------------------------------------

#GENERACION DE GRILLAS

# Estas funciones generan las tres grillas necesarias para obtrener el mapa de AB

#GRILLAS X Y

# Se genera una grilla con una extencion dada entre los valores maximos y
# minimos de latitud y longitud presentes en los datos.
#
# x son los valores de longitud de los datos
# y son los valores de latitud de los datos
# resolucion_x es el numero de columnas deseado para las grillas
# resolucion_y es el numero de filas deseado para las grillas

def grilla(x,y,resolucion_x,resolucion_y):
    nx,ny   = resolucion_x, resolucion_y
    grid_x  = np.linspace(min(x), max(x), nx)
    grid_y  = np.linspace(min(y), max(y), ny)
    X,Y     = np.meshgrid(grid_x, grid_y)

    print("Se generaron grillas de:", resolucion_x, "x", resolucion_y)

    return nx, ny, X, Y

#GRILLA DE ANOMALIA DE BOUGUER

# Se genera una grilla para la anomalia de Bouguer a partir de una 
# interpolacion de los datos obtenidos de esta.
#
# x son los valores de longitud de los datos
# y son los valores de latitud de los datos
# resolucion_x es el numero de columnas deseado para las grillas
# resolucion_y es el numero de filas deseado para las grillas
# gz son los valores de anomalia de Bouguer obtenidos
# metodo es un string que indica el tipo de interpolacion deseada.

def interpolacion(x,y,gz,resolucion_x,resolucion_y,metodo):
    
    grid_x  = np.linspace(min(x), max(x), resolucion_x)
    grid_y  = np.linspace(min(y), max(y), resolucion_y)
    X,Y     = np.meshgrid(grid_x, grid_y)

    rbf3 = Rbf(x, y, gz, function="linear", smooth=0.1)

    xnew, ynew = np.meshgrid(grid_x, grid_y)
    xnew = xnew.flatten()
    ynew = ynew.flatten()
    znew = rbf3(xnew, ynew)
    GZ = griddata(points=(xnew,ynew), values=znew, xi=(X,Y), method='linear', fill_value=np.nan)

    return GZ

#--------------------------------------------------------------------

# PROCESAMIENTO EN DOMINIO ESPACIAL

# AJUSTE POLINOMIAL DE PRIMER ORDEN

# Se obtiene una grilla propia de un ajuste polinomial de primer orden
# de los valores de anomalia de Boouguer estudiados.
#
# x son los valores de longitud de los datos
# y son los valores de latitud de los datos
# nx es el numero de columnas deseado para las grillas
# ny es el numero de filas deseado para las grillas
# gz son los valores de anomalia de Bouguer obtenidos
# GZ es la grilla de anomalia de Bouguer obtenida luego de la interpolación

def ajuste_polinomial_1(nx,ny,x,y,gz,GZ):

    nx,ny,X,Y = grilla(x,y,nx,ny)

    m      = np.zeros(shape=(3,1))      #Inicializo los coeficientes.

    #A es una matriz donde guardo las variables de cada termino del polinomio.
    A      = np.zeros(shape=(len(gz),len(m)))
    A[:,0] = x                  #Primer termino     :X
    A[:,1] = y                  #Segundo termino    :Y
    A[:,2] = np.ones(len(gz))   #Tercer termino     :cte
    
    from scipy.linalg import lstsq
    
    m,_,_,_ = lstsq(A,gz)  #Ajusta los coeficientes con minimos cuadrados
    
    A_grid = np.zeros(shape=(nx*ny,len(m)))
    
    A_grid[:,0] = X.reshape(nx*ny) 
    A_grid[:,1] = Y.reshape(nx*ny) 
    A_grid[:,2] = np.ones(nx*ny)
    
    GZ_fit1 = A_grid @ m # En Python>3, '@' es el producto entre una matriz y un vector
    GZ_res1 = GZ - GZ_fit1.reshape(nx,ny) # campo residual interpolado
    
    return GZ_fit1.reshape(nx,ny), GZ_res1

# AJUSTE POLINOMIAL DE SEGUNDO ORDEN

# Se obtiene una grilla propia de un ajuste polinomial de segundo orden
# de los valores de anomalia de Boouguer estudiados.
#
# x son los valores de longitud de los datos
# y son los valores de latitud de los datos
# nx es el numero de columnas deseado para las grillas
# ny es el numero de filas deseado para las grillas
# gz son los valores de anomalia de Bouguer obtenidos
# GZ es la grilla de anomalia de Bouguer obtenida luego de la interpolación

def ajuste_polinomial_2(nx,ny,x,y,gz,GZ):
    
    nx,ny,X,Y = grilla(x,y,nx,ny)
    
    m      = np.zeros(shape=(6,1))      #Inicializo los coeficientes.
    
    #A es una matriz donde guardo las variables de cada termino del polinomio.
    A      = np.zeros(shape=(len(gz),len(m)))
    
    A[:,0] = x**2               # Primer terminio   :X^2
    A[:,1] = y**2               # Segundo terminio  :Y^2
    A[:,2] = x*y                # Tercer termino    :X*Y
    A[:,3] = x                  # Cuarto termino    :X
    A[:,4] = y                  # Quinto termino    :Y
    A[:,5] = np.ones(len(gz))   # Sexto termino     :cte
    
    from scipy.linalg import lstsq
    
    m,_,_,_ = lstsq(A,gz)   #Ajusta los coeficientes con minimos cuadrados
    
    #Los elementos de la grilla los reordeno como un vector.
    A_grid = np.zeros(shape=(nx*ny,len(m)))     #Inicializo la matriz A.
    
    A_grid[:,0] = X.reshape(nx*ny) **2
    A_grid[:,1] = Y.reshape(nx*ny) **2
    A_grid[:,2] = X.reshape(nx*ny) * Y.reshape(nx*ny)
    A_grid[:,3] = X.reshape(nx*ny)
    A_grid[:,4] = Y.reshape(nx*ny)
    A_grid[:,5] = np.ones(nx*ny)
    
    GZ_fit2 = A_grid @ m #'@' es el producto entre una matriz y un vector
    GZ_res2 = GZ - GZ_fit2.reshape(nx,ny) # campo residual interpolado
    
    return GZ_fit2.reshape(nx,ny), GZ_res2

#--------------------------------------------------------------------

# PROCESAMIENTO EN DOMINIO DE NUMERO DE ONDA

# PREPROCESADO DE LA GRILLA DE ANOMALIA DE BOUGUER

# Esta funcion realiza un preprocesado que va a facilitar la implementacion
# de filtros en el dominio de numeros de onda, permitiendo aumentar el numero
# de muestras y por lo tanto incrementando el dominio.
# Lo que se hace es espejar la grilla de AB en sentido vertical y horizontal

# GZ es la grilla de AB (Anomalia de Bouguer)

def preprocesado(GZ):
    CUAD1 = np.fliplr(GZ)
    CUAD2 = GZ
    CUAD3 = np.flipud(GZ)
    CUAD4 = np.flipud(CUAD1)

    preprocesado = np.concatenate((np.concatenate((CUAD3,CUAD4),axis=1),
                             np.concatenate((CUAD2,CUAD1),axis=1)), axis=0)
    
    return preprocesado

# GRILLA DEL DOMINIO DE NUMEROS DE ONDA

def grilla_TDF2(X,Y,preprocesado):
    
    dx=X[1,:][1]-X[1,:][2]
    dy=Y[:,1][1]-Y[:,1][2]
    
    kx_adimensional = (np.pi)*np.fft.fftfreq(len(preprocesado)) 
    kx              = kx_adimensional / dx          # 1/longitud
    ky_adimensional = (np.pi)*np.fft.fftfreq(len(preprocesado)) 
    ky              = ky_adimensional / dy          # 1/longitud
    
    kxx,kyy = np.meshgrid(kx,ky)
    
    return kxx, kyy

# FILTRADO PASA BAJOS

def FPB_2D(kxx,kyy,W,preprocesado,nx,ny):
    
    ft = np.fft.fft2(preprocesado) 
    
    ft_w = ft * W
    
    ift = np.real(np.fft.ifft2(ft_w))   #Antitransformo el espectro filtrado
    ift = ift[0:nx, 0:ny]               #Recorto los ceros del preprocesado.
    ift = ift.real                      #Me quedo solo con la parte real.
    
    return ift, ft, ft_w

#--------------------------------------------------------------------

# METODOS DE REALCE

# DERIVADA DIRECCIONAL HORIZONTAL SEGUN X

def DERIVADA_X(X,Y,GZ):
    
    prepro = preprocesado(GZ)
    
    kxx , kyy = grilla_TDF2(X,Y,prepro)
    
    ft = np.fft.fft2(prepro) 
    
    ft_w = ft * kxx
    
    ift = np.imag(np.fft.ifft2(ft_w))   #Antitransformo el espectro filtrado
    ift = ift[0:np.shape(GZ)[1], 0:np.shape(GZ)[0]]               #Recorto los ceros del preprocesado.
    ift = ift.real                      #Me quedo solo con la parte real.
    
    ift = np.flipud(ift)
    
    return ift, ft, ft_w

# DERIVADA DIRECCIONAL HORIZONTAL SEGUN Y

def DERIVADA_Y(X,Y,GZ):
    
    prepro = preprocesado(GZ)
    
    kxx , kyy = grilla_TDF2(X,Y,prepro)
    
    ft = np.fft.fft2(prepro) 
    
    ft_w = ft * kyy
    
    ift = np.imag(np.fft.ifft2(ft_w))   #Antitransformo el espectro filtrado
    ift = ift[0:np.shape(GZ)[1], 0:np.shape(GZ)[0]]               #Recorto los ceros del preprocesado.
    ift = ift.real                      #Me quedo solo con la parte real.
    
    ift = np.flipud(ift)
    
    return ift, ft, ft_w

# DERIVADA DIRECCIONAL VERTICAL DE ORDEN N

def DERIVADA_Zn(X,Y,GZ,n):
    
    prepro = preprocesado(GZ)
    
    kxx , kyy = grilla_TDF2(X,Y,prepro)
    
    kr = np.sqrt(kxx**2+kyy**2)
    
    ft = np.fft.fft2(prepro) 
    
    ft_w = ft * kr**n
    
    ift = np.real(np.fft.ifft2(ft_w))   #Antitransformo el espectro filtrado
    ift = ift[0:np.shape(GZ)[1], 0:np.shape(GZ)[0]]               #Recorto los ceros del preprocesado.
    ift = ift.real                      #Me quedo solo con la parte real.
    
    ift = np.flipud(ift)
    
    return ift, ft, ft_w