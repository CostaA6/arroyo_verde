#--------------------------------------------------------------------
#Funciones auxiliares
#--------------------------------------------------------------------
import numpy as np
from scipy.interpolate import griddata 
from scipy.interpolate import Rbf

def seccion(x,X,y,Y,gz):
    GZ = griddata(points=(x,y), values=gz, xi=(X,Y), method='cubic', fill_value=np.nan)
    sec = GZ*0+1
    
    return sec
#--------------------------------------------------------------------

#--------------------------------------------------------------------