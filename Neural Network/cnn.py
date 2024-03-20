import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dimensiones de las imágenes de entrada
batch_size = 100
img_height = 180
img_width = 180

# Directorio del dataset
data_dir = '/Users/PC HOLLOW 3000/Desktop/PIA PI/cell_images'  # Ruta al directorio del dataset local

# Rutas a las subcarpetas de entrenamiento y prueba
train_dir = os.path.join(data_dir, '/Users/PC HOLLOW 3000/Desktop/PIA PI/cell_images/train')  # Ruta a la subcarpeta de entrenamiento
test_dir = os.path.join(data_dir, '/Users/PC HOLLOW 3000/Desktop/PIA PI/cell_images/test')  # Ruta a la subcarpeta de prueba

# Carga de imágenes de entrenamiento y prueba desde el directorio
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Obtener los nombres de las clases
class_names = train_data.class_names
print(class_names)

# Visualizar algunas imágenes de entrenamiento
plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# Configuración de rendimiento del dataset
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)



# Capa de normalización para escalar los valores de píxeles a [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Aplicar normalización a los datos de entrenamiento y prueba
normalized_train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
normalized_val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

# Número de clases
num_classes = len(class_names)

# Definición del modelo
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

# Compilación del modelo
model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Entrenamiento del modelo
epochs = 5
history = model.fit(
    normalized_train_data,
    validation_data=normalized_val_data,
    epochs=epochs
)

# Gráficas de precisión y pérdida durante el entrenamiento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

