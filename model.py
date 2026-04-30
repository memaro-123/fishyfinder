from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

imgWidth = 224
imgHeight = 224
batchSize =32
epochs =20

base_model = MobileNetV2(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False,
    pooling='avg'  
)

base_model.trainable = False

TRAINING_DIR = "/root/fishyfinder/train"
VALIDATION_DIR = "/root/fishyfinder/val"

NumOfClasses = len(glob(f'{TRAINING_DIR}/*'))
print(NumOfClasses)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(imgHeight, imgWidth), 
    batch_size=batchSize,
    class_mode='categorical',
    shuffle=True  
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR, 
    target_size=(imgHeight, imgWidth), 
    batch_size=batchSize,
    class_mode='categorical',
    shuffle=False 
)

print(f"\n Training images: {train_generator.samples}")
print(f" Validation images: {validation_generator.samples}")
print(f" Classes: {train_generator.class_indices}")
print(f" Class names: {list(train_generator.class_indices.keys())}")


inputs = base_model.input
x = base_model.output
x = Dense(512, activation='relu')(x)  
x = Dropout(0.3)(x)
outputs = Dense(NumOfClasses, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)



#early stopping
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        verbose=1, 
        restore_best_weights=True,
        mode='min'
    ),
    ModelCheckpoint(
        "/root/fishyfinder/best_model.h5", 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        verbose=1,
        min_lr=1e-7
    )
]

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batchSize,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batchSize,
    callbacks=callbacks,
    verbose=1
)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/fishyfinder/training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Evaluate on validation set
print("\n" + "="*50)
print(" Final Evaluation:")
print("="*50)
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)
print(f" Validation Loss: {val_loss:.4f}")
print(f" Validation Accuracy: {val_accuracy:.2%}")

# Save final model
model.save('/root/fishyfinder/final_model.h5')
print("\n Model saved to /root/fishyfinder/final_model.h5")

validation_generator.reset()  
model.evaluate(validation_generator)

# Make predictions
predictions = model.predict(x_batch, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_batch, axis=1)

# Get class names
class_names = list(train_generator.class_indices.keys())

# Show first 5 predictions
for i in range(min(5, len(predicted_classes))):
    print(f"Sample {i+1}:")
    print(f"  True: {class_names[true_classes[i]]}")
    print(f"  Predicted: {class_names[predicted_classes[i]]}")
    print(f"  Confidence: {predictions[i][predicted_classes[i]]:.2%}")
    print()

