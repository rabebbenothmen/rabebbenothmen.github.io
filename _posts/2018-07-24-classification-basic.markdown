---
layout: post
title:  "Classification de base"
date:   2018-07-24 15:00:00 +0200
categories: jekyll update
---

Dans ce tutoriel on va présenter quelque technique des classification de base de l'intelligence artificielle .

Le procesus de classification nous aide a classer les données dans un nombre donnée de classes.
Lors de cette classification, nous classons les données dans un nombre fixe de catégorie .
Dans l'apprentissage automatique, la classification résout le problème de l'identification de la catégorie à laquelle appartient un nouveau point de données. Nous construisons le modèle de classification en fonction de l'ensemble de données d'apprentissage contenant les points de données et les étiquettes correspondantes.

## Réseaux de neurones artificiels
Dans cette partie on va former un modèle de réseau de neurones pour classer les images des vêtements.
#### jeu de données Fashion MNIST
![alt text](http://127.0.0.1:4000/image/basicfashion.png "basic fasion")


Nous utilisions 60.000 images pour formerun réseau et 10,000 image pour évaluer avec quelle précision le réseau a appris à classer les images. Nous pouvons accéder au mode MNIST directement à partir de TensorFlow, il suffit d'importer et de charger les données:

Le chargement du jeu de données renvoie quatre tableaux NumPy:

* Les tableaux `train_images` et `train_labels` sont l' ensemble de formation - les données que le modèle utilise pour apprendre.

* Le modèle est testé par rapport à l' ensemble de test , aux tableaux `test_images` et aux `test_labels` tableaux.

|  Étiquette |Classe   	      |
|---	     |---	          |
| 0  	     | T-shirt / haut |
| 1 	     | Pantalon  	  |
| 2  	     | Arrêtez-vous   |
| 3 	     | Robe  	      |
| 4  	     | Manteau        | 
| 5          | Sandale        |
| 6  	     | Chemise  	  |
| 7  	     | Sneaker        |
| 8	         | Sac  	      |
| 9  	     | Bottine	      |
                          
Chaque image est mappée sur une seule étiquette. Comme les noms de classes ne sont pas inclus dans l'ensemble de données, stockez-les ici pour les utiliser plus tard lors du traçage des images:
```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']`
```
#### Explorer les données 
Avant d'entraîner le modèle on va explorer le format de l'ensemble de données .
l'illustration suivante montre qu'il ya 60 000 images dan l'ensemble d'apprentissage et chaque image répresenter par 28 x 28 pixels:
```python
train_images.shape  => (60 000,28,28) 
```

Et cette illustration montre qu'il ya 60 000 etiquette dans l'ensemble de formation 
```python
len(train_labels) => 60 000
```

Et l'ullustration suivante présente qu'il y a 10 000 images dans l'ensemble de test et chaque image réprensenter par 28 x 28 pixels :
```python

test_images.shape => (60 000,28,28)
```
Et l'ensemble de test contient 10 000 étiquettes d'images:
```python
len(test_labels) => 10 000
```

#### Pré-traiter les données
Les données doivent être prétraitées avant d'entraîner le réseau.
* plt.figure()  : la construction de la fenêtre graphique
* plt.imshow(train_images[0])
* plt.colorbar()
* plt.gca().grid(False)

![alt text](http://127.0.0.1:4000/image/ankleboot.png "ankle boot")

Nous adaptons ces valeurs à une plage de 0 à 1 avant de les transmettre au modèle de réseau neuronal. Pour cela, transtypez le type de données des composants de l'image d'un nombre entier à un nombre entier, et divisez par 255. Voici la fonction pour prétraiter les images:
```python
train_images = train_images / 255.0

test_images = test_images / 255.0
```

#### Construire le modèle 

La construction du réseau de neurones nécessite la configuration des couches du modèle, puis la compilation du modèle
* Configuration de calque:

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

* Compiler le modèle
```python
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
#### Entraîner le modèle 
1. Envoyez les données d'apprentissage au modèle.
2. Le modèle apprend à associer des images et des étiquettes.
3. Nous demandons au modèle de faire des prédictions sur un ensemble de tests.

```python 
model.fit(train_images, train_labels, epochs=5)
```
#### Évaluer l'exactitude
Ensuite, comparez les performances du modèle sur l'ensemble de données de test:
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
```
#### Faire le prédiction
Avec le modèle formé, nous pouvons l'utiliser pour faire des prédictions sur certaines images
```python
predictions = model.predict(test_images)
```
Jetons un coup d'oeil à la première prédiction:
```python
predictions[0]
```
Nous pouvons voir quelle étiquette a la valeur de confiance la plus élevée:
```python
np.argmax(predictions[0])
```
 Et nous pouvons vérifier l'étiquette de test pour voir c'est correct:
```python
 test_labels[0]
 ```