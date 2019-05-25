# Generación de rostros artificiales usando redes adversarias

Código fuente de [este](https://youtu.be/dlnA1uiWu90) video, en donde se muestra cómo utilizar las Generative Adversarial Networks (GAN) para generar imágenes artificiales de rostros humanos.

## Contenido

- *dataset*: carpeta con el set de datos utilizado durante el entrenamiento. Contiene 3,755 imágenes a color de rostros humanos reales, cada una con un tamaño de 128x128.
- *ejemplos*: carpeta con 100 ejemplos de imágenes obtenidas tras el entrenamiento de la GAN.
- *utilidades.py*: funciones para la lectura del set de entrenamiento, la visualización de las imágenes obtenidas y la generación de imágenes con el modelo entrenado.
- *generacion_de_rostros.py*: implementación de la GAN para la generación de rostros artificiales
- *generador.h5*: GAN entrenada (5000 iteraciones)

## Lecturas recomendadas

- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
- Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.

## Dependencias
matplotlib==2.0.0
numpy==1.15.4
Keras==2.2.4
imageio==2.5.0