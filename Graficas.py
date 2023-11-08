#==============================================================
import matplotlib.pyplot as plt
import numpy as np
#-----------------

#Paleta de colores:
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
colors = [(1., 0.75, 0.875), (1., 0.5, 0.75), (1.0, 0.0, 0.5), (1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.6)][::-1]
cmap = LinearSegmentedColormap.from_list('mycmap', colors)

#==============================================================
#Grafico del grillado y de las mediciones:
def grilla(X,x,Y,y,gz,GZ,titulo):
    import matplotlib.ticker as ticker
    # Configurar el formateo de los ejes
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)

    # Aplicar el formateo a los ejes x e y
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    #plt.scatter(X, Y, c='gray', s=0.5)
    plt.scatter(x, y, s=100, c=gz , cmap = cmap,edgecolors='black')
    cbar = plt.colorbar()
    plt.clim(np.min(GZ) , np.max(GZ))
    #plt.axis([-65.36, -65.323, -42.0375, -42.014])
    plt.grid(True, color='black',linestyle='dotted')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('EJE X',fontsize=8)
    plt.ylabel('EJE Y',fontsize=8)
    plt.title(titulo,fontsize=10)
    img = plt.imread('/home/andres/Escritorio/Codigo Gravimetria/IMGSAT.png')
    #img2 = plt.imread('/home/andres/Escritorio/Codigo Gravimetria/afloramiento_a.png')
    plt.imshow(img, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], aspect='auto', alpha=0.6)

    # plt.imshow(GZ,cmap=cmap, origin="lower",
    #            interpolation=None,
    #            extent=[np.min(x),np.max(x),np.min(y),np.max(y)],
    #            alpha=1)
    plt.clim(np.min(GZ) , np.max(GZ))
    cbar.set_label('mGal')
    plt.show()
#==============================================================
#Grafico del mapa:
def mapa(X,x,Y,y,GZ, titulo):
    
    
    import matplotlib.ticker as ticker

# Dibujar tus gráficos
# ...

# Configurar el formateo de los ejes
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)

# Aplicar el formateo a los ejes x e y
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)

#     xf = 3555500  # Coordenada x de la flecha (ajustar según sea necesario)
#     yf = 5348000  # Coordenada y de la flecha (ajustar según sea necesario)
#     arrow_size = 600  # Tamaño de la flecha (ajustar según sea necesario)

# # Dibujar la flecha
#     plt.annotate('N', xy=(xf, yf), xytext=(xf, yf-arrow_size),
#              arrowprops=dict(arrowstyle='->', color='black', linewidth=5),
#              ha='center', va='center', fontsize=12, fontweight='bold')

    
    # img = plt.imread('/home/andres/Escritorio/Codigo Gravimetria/IMGSAT.png')
    #img2 = plt.imread('/home/andres/Escritorio/Codigo Gravimetria/afloramiento_a.png')
    # plt.imshow(img, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], aspect='auto', alpha=0.6)
    #plt.imshow(img2, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], aspect='auto', alpha=0.4)
    
    # plt.contour(X, Y, GZ, 10, colors="black", linewidths=0.3)
    
    im = plt.imshow(GZ,cmap=cmap, origin="lower",
           interpolation=None,
           extent=[np.min(x),np.max(x),np.min(y),np.max(y)],
           alpha=1)
    
    cbar = plt.colorbar(im, orientation = 'vertical')
    tick_font_size = 8
    cbar.ax.tick_params(labelsize=tick_font_size)
       
    # plt.grid(True, color='black',linestyle='dotted')
    #plt.scatter(x, y,s=3, color="black")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('EJE X',fontsize=6)
    plt.ylabel('EJE Y',fontsize=6)
    plt.title(titulo,fontsize=6)
    
    # plt.clim(-3.103 , 1.956)
    
    #plt.clim(-0.00001 , 0.00001)
    
    #plt.show()
#==============================================================    
def prepros(GZ, titulo):
    plt.imshow(GZ,cmap="jet", origin="lower", interpolation=None)
    plt.colorbar(orientation = 'vertical')
    plt.grid(True)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.title(titulo,fontsize=6)
    plt.show()
#==============================================================
#Grafico de los espectros:
def espectro(A, titulo):
    
    X1 = abs(A)
    X1n = X1 / np.max(X1)
    X1dB = 20 * np.log10(X1n + 0.00000001)
    A = X1dB

    #cmap = plt.get_cmap('inferno')  # Cambia 'viridis' al mapa de colores que desees

    im = plt.imshow(A, cmap=cmap)
    plt.title(titulo)

    cbar = plt.colorbar(im, spacing='proportional', orientation='horizontal', shrink=0.7, format="%.0f")
    cbar.ax.tick_params(labelsize=12)  # Establece el tamaño de la fuente en la colorbar
    cbar.set_label(' ', fontsize=12)  # Puedes cambiar el texto de la etiqueta aquí

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.title(titulo, fontsize=12)
    
#==============================================================