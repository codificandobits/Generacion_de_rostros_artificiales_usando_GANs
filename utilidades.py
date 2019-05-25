dataset = "./dataset"
ejemplos = "./ejemplos"

import os
from imageio import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt

# Creación del set de entrenamiento
def cargar_datos():
	print('Creando set de entrenamiento...',end="",flush=True)
	filelist = os.listdir(dataset)

	n_imgs = len(filelist)
	x_train = np.zeros((n_imgs,128,128,3))

	for i, fname in enumerate(filelist):
		if fname != '.DS_Store':
			imagen = imread(os.path.join(dataset,fname))
			x_train[i,:] = (imagen - 127.5)/127.5
	print('¡Listo!')

	return x_train

# Visualizar imágenes del set de entrenamiento
def visualizar_imagen(nimagen,x_train):
	imagen = (x_train[nimagen,:]*127.5) + 127.5
	imagen = np.ndarray.astype(imagen, np.uint8)
	plt.imshow(imagen.reshape(128,128,3))
	plt.axis('off')
	plt.show()

# Visualización de algunas imagenes obtenidas con el generador
def graficar_imagenes_generadas(epoch, generador, ejemplos=16, dim=(4,4), figsize=(10,10)):
    ruido = np.random.normal(0,1,[ejemplos,100])
    imagenes_generadas = generador.predict(ruido)
    imagenes_generadas.reshape(ejemplos,128,128,3)
    imagenes_generadas = imagenes_generadas*127.5 + 127.5
    plt.figure(figsize=figsize)
    for i in range(ejemplos):
        plt.subplot(dim[0],dim[1], i+1)
        plt.imshow(imagenes_generadas[i].astype('uint8'), interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('GAN_imagen_generada_%d.png' %epoch)
    plt.close()

# Generar imágenes ejemplo
def generar_imagenes(generador,nimagenes):
	ruido = np.random.normal(0,1,[nimagenes,100])
	imagenes_generadas = generador.predict(ruido)
	imagenes_generadas.reshape(nimagenes,128,128,3)
	imagenes_generadas = imagenes_generadas*127.5 + 127.5
	imagenes_generadas.astype('uint8')
	for i in range(nimagenes):
		imwrite(os.path.join(ejemplos,'ejemplo_'+str(i)+'.png'),imagenes_generadas[i].reshape(128,128,3))
