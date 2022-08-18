# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 02:35:49 2022

@author: Villar
"""

#----------------------------PARTE 1: CREAR LA ESTRUCTURA DE CNN--------------------------------------------

#Librería para inicializar la red neuronal con pesos aleatorios
from keras.models import Sequential
#Crear capa de convolución 2D
from keras.layers import Convolution2D
#crear capa de maxpooling
from keras.layers import MaxPooling2D
#Aplanar las matrices a un solo vector(que son los datos de cada pixel)
from keras.layers import Flatten
#Crear la sinápsis
from keras.layers import Dense
#Dropout
from keras.layers import Dropout
#Librería para guardar pesos en formato .yaml
from keras.models import model_from_yaml
#Para dibujar las diferencias entre las accuracy de train y test, así como las pérdidas de ambas
from matplotlib import pyplot as plt


#Inicializar la CNN
classifier = Sequential()

#Pao 1 - Convolución: Tomar la imagen, aplicar detectores de rasgos
#Para crear varias imágenes con menos pixeles, las que seran los mapas de características

#Filter: Cantidad de mapas de características
#kernel_size: dimensión de cada kernel
#input_shape: cantidad de pixeles de filas, columnas y la cantidad de capas de color(3 = azul, rojo, amarillo)
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(128, 128, 1), activation="relu"))

#Paso 2 - Max Pooling: Obtener las características más significativas de los mapas de características
#Así se reduce la cantidad de nodos. Cabe destacar que aquí la matriz resultante es con los pixeles más significativos

#pool_size
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Aplicamos una segunda capa de convolución y Max Pooling para profundizar el aprendizaje de la red
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Paso 3 - Flattening: Se aplasta la matriz a un vector unidimensional (a, b, c, ...., N)
#Así le entregamos cada columna como nodo a la red neuronal
classifier.add(Flatten())

#Paso 4 - Full Connection
classifier.add(Dense(units=512, activation="relu"))
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

#Vemos las características de nuestra red
print(classifier.summary())

#Compilar la CNN
#optimizer: algoritmo de optimización de pesos
#loss: funcion de pérdida que se utilizará, si tuvieramos más de una clase a predecir mediante las imágenes, deberíamos usar una categórica
#metrics: métricas que quiero que me muestre
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#----------------------------PARTE 2: AJUSTAR LA CNN A LAS IMÁGENES A ENTRENAR--------------------------------------------

#Generamos más imágenes a partir de las que ya tenemos, les hará zoom, moverá pixeles, etc.
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255, #los pixeles ahora tendrán números décimales de 0 a 1
        shear_range=0.2, #
        zoom_range=0.2, #20% de zoom
        horizontal_flip=True) #se da vuelta la imágen en sentido horizontal

test_datagen = ImageDataGenerator(rescale=1./255) #reescalado en el conjunto de test

training_dataset = train_datagen.flow_from_directory( #Cargamos una carpeta de trabajo con las imágenes para entrenar
        'dataset/training_set', #nombre de la carpeta donde están las imágenes para entrenar
        target_size=(128, 128), #Tamaño en el que espero cargar las imágenes, y como en la convolución esperamos que salgan de 64x64, colocamos esto
        batch_size=32, #tamaño del bloque de carga, es la cantidad de imágenes que pasaran por la red neuronal antes de proceder a actualizar los pesos
        class_mode='binary',
        color_mode='grayscale') #como tengo 2 categorías, entonces es binario. Si tuviera una clasificación > 2, habría que cambiarlo

testing_dataset = test_datagen.flow_from_directory( #Validador para el conjunto de test 
        'dataset/test_set', 
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        color_mode='grayscale')

results = classifier.fit_generator( #Hacer el ajuste a nuestro modelo(clasificador)
        training_dataset,#conjunto de entrenamiento
        steps_per_epoch=int(8000/32), #cuantas muestras debe tomar por pasada en la red neuronal -> Cantidad de imágenes en el entrenamiento / batch size
        epochs=50, #Cantidad de épocas, es decir, la cantidad de veces que pasará el conjunto de entrenamiento por la red
        validation_data=testing_dataset, #conjunto de test
        validation_steps=int(2000/32)) # Cantidad de imágenes en el conjunto de validación / batch size

print(results.history['accuracy'])

#Guardamos los pesos
model_yaml = classifier.to_yaml()
with open("model.yaml", "w") as yaml_file:
  yaml_file.write(model_yaml)
# serializa los pesos(weights) para HDF5
classifier.save_weights("model.h5")

#-----------------------------------------------------------------------------------------------------------------------

# Finalmente hacemos el plot de los gráficos para comparar las funciones de pérdida y las precisiones
# Así vemos si el modelo se está sobreajustando o no
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('Accuracy Graph Train vs Test')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()


plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Loss Function Graph Train vs Test')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
