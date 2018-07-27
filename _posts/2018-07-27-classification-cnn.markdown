---
layout: post
title:  "Classification CNN ( Convolutional Neural Network)"
date:   2018-07-27 15:00:00 +0200
categories: jekyll update
---
Les réseaux de neurones convolutionnels (CNN) sont à ce jour les modèles les plus performants pour classer des images.  ils comportent deux parties bien distinctes. 
En entrée, une image est fournie sous la forme d’une matrice de pixels. Elle a 2 dimensions pour une image en niveaux de gris. 
La couleur est représentée par une troisième dimension, de profondeur 3 pour représenter les couleurs fondamentales [Rouge, Vert, Bleu].

Dans ce post on va classer l'ensemble de données Fashion-MNIST avec tf.keras, en utilisant une architecture CNN ( Convolutional Neural Network ).

 On a 10 catérgories à classer dans l'ensemble de données fashion_mnist:
0. T-shirt / top  
1. Pantalon  
2. Pull  
3. Robe  
4. Manteau  
5. Sandale  
6. Chemise  
7. Sneaker  
8. Sac  
9. Bottine

#### Importer le jeu de données fashion_mnist
Importons l'ensemble de données et préparons-le pour la formation, la validation et le test.

Chargez les données fashion_mnist avec l'API keras.datasets avec une seule ligne de code. Puis une autre ligne de code pour charger le train et tester le jeu de données. Chaque image en niveaux de gris est 28x28.

```python
importer tensorflow comme tf
importez numpy comme np 
import matplotlib.pyplot comme plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
``` 
`=>x_train shape: (60000, 28, 28, 1) y_train shape: (60000,)`

#### Visualisez les données
Nous pouvons visualiser une image sur jupyter à partir de l'ensemble de données d'apprentissage avec imshow () de la bibliothèque matplotlib pour voir l'une des images des jeux de données

Donc on peux afficher l'une des images de jeu des données d'apprentissage avec cette instruction :
```python
plt.imshow (x_train [img_index])
```

![alt text](https://rabebbenothmen.github.io/image/imagevisualise.png "imagevisualisé")
#### Normalisation des données
Nous normalisons ensuite les dimensions des données pour qu'elles aient approximativement la même échelle.
```python
x_train = x_train.astype ('float32') / 255 
x_test = x_test.astype ('float32') / 255
```
#### Modèle 
Il existe deux API pour définir un modèle dans Keras:

1. API de modèle séquentiel
2. API fonctionnelle
Dans ce tutoriel, nous utilisons l' API de modèle séquentiel pour créer un modèle CNN simple qui répète quelques couches d'une couche de convolution suivie d'une couche de regroupement puis d'une couche de suppression.

```python
model = tf.keras.Sequential ()
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(2,2), padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(2,2), padding='same', activation='relu')) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary ()
```
#### Compiler le modèle
```python
model.compile (loss = 'categorical_crossentropy', 
             optimizer = 'adam', 
             metrics = ['exactitude'])

```
#### Entraînez le modèle
```python
model.fit(x_train,  y_train, batch_size=10, epochs=10)
```
![alt text](https://rabebbenothmen.github.io/image/compilation.png "compilation")


#### Exactitude du test
Et nous obtenons une précision de test de plus de 90%.

# Evaluer le modèle sur l'ensemble de tests 
```python
score = model.evaluate (x_test, y_test, verbose = 0)
```
# Impression de la précision du test d' 
impression ('\ n', 'Exactitude du test:', note [1])
```python
print('\n', 'Test accuracy:', score[1])
```
![alt text](https://rabebbenothmen.github.io/image/tauxtest.png "tauxtest")


#### Visualisez les prédictions
Maintenant, nous pouvons utiliser le modèle formé pour faire des prédictions / classifications sur l'ensemble de données de test model.predict(x_test)et les visualiser. Si vous voyez l'étiquette en rouge, cela signifie que la prédiction ne correspond pas à la véritable étiquette; sinon c'est vert.

Avec le modèle formé, nous pouvons l’utiliser pour faire des prédictions sur certaines images
```python
model.predict(x_test)

predictions = model.predict(x_test)
```
stockez-les ensemble des images içi pour l'utiliser 
```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']`
```
```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(np.reshape(x_test[i],(28,28)), cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label]),
                                  color=color)
```

![alt text](https://github.com/rabebbenothmen/rabebbenothmen.github.io/tree/master/image/res.png "res")

