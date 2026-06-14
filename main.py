import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def generate_synthetic_dataset(base_dir="dataset", num_samples=200):
    """Generates synthetic Cats vs Dogs images for transfer learning."""
    print(f"Generating synthetic dataset inside '{base_dir}'...")
    os.makedirs(os.path.join(base_dir, "cats"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "dogs"), exist_ok=True)
    
    np.random.seed(42)
    
    # Generate Cats (pointy ears, whiskers)
    for i in range(num_samples):
        bg_color = tuple(np.random.randint(40, 200, size=3))
        img = Image.new("RGB", (224, 224), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Face dimensions
        r = np.random.randint(65, 80)
        cx = np.random.randint(105, 115)
        cy = np.random.randint(105, 115)
        
        cat_color = (np.random.randint(180, 230), np.random.randint(130, 170), np.random.randint(70, 110)) # Ginger cat
        if np.random.rand() > 0.5:
            cat_color = (180, 180, 180) # Grey cat
            
        # Draw pointy ears (triangles)
        # Left ear
        draw.polygon([(cx - r + 15, cy - 25), (cx - r - 10, cy - r - 15), (cx - 20, cy - 60)], fill=cat_color, outline=(0, 0, 0))
        # Right ear
        draw.polygon([(cx + r - 15, cy - 25), (cx + r + 10, cy - r - 15), (cx + 20, cy - 60)], fill=cat_color, outline=(0, 0, 0))
        
        # Draw main head circle
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=cat_color, outline=(0, 0, 0), width=2)
        
        # Draw eyes (large green/yellow ellipses)
        eye_color = (np.random.randint(50, 100), np.random.randint(180, 230), 50)
        draw.ellipse([cx - 28, cy - 20, cx - 12, cy - 4], fill=eye_color, outline=(0, 0, 0))
        draw.ellipse([cx + 12, cy - 20, cx + 28, cy - 4], fill=eye_color, outline=(0, 0, 0))
        # Pupils
        draw.ellipse([cx - 22, cy - 20, cx - 18, cy - 4], fill=(0, 0, 0))
        draw.ellipse([cx + 18, cy - 20, cx + 22, cy - 4], fill=(0, 0, 0))
        
        # Draw whiskers
        # Left side
        draw.line([cx - 20, cy + 15, cx - 80, cy + 10], fill=(255, 255, 255), width=2)
        draw.line([cx - 20, cy + 20, cx - 85, cy + 22], fill=(255, 255, 255), width=2)
        draw.line([cx - 20, cy + 25, cx - 80, cy + 35], fill=(255, 255, 255), width=2)
        # Right side
        draw.line([cx + 20, cy + 15, cx + 80, cy + 10], fill=(255, 255, 255), width=2)
        draw.line([cx + 20, cy + 20, cx + 85, cy + 22], fill=(255, 255, 255), width=2)
        draw.line([cx + 20, cy + 25, cx + 80, cy + 35], fill=(255, 255, 255), width=2)
        
        # Draw tiny nose (pink triangle)
        draw.polygon([(cx - 6, cy + 15), (cx + 6, cy + 15), (cx, cy + 22)], fill=(255, 182, 193))
        
        img.save(os.path.join(base_dir, "cats", f"cat_{i}.jpg"))
        
    # Generate Dogs (floppy ears, snout)
    for i in range(num_samples):
        bg_color = tuple(np.random.randint(40, 200, size=3))
        img = Image.new("RGB", (224, 224), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Face dimensions
        r = np.random.randint(65, 80)
        cx = np.random.randint(105, 115)
        cy = np.random.randint(105, 115)
        
        dog_color = (139, 69, 19) # Brown dog (SaddleBrown)
        if np.random.rand() > 0.5:
            dog_color = (244, 164, 96) # Sandy dog (SandyBrown)
            
        # Draw floppy ears first (so head overlaps them nicely)
        # Left ear
        draw.ellipse([cx - r - 20, cy - 25, cx - r + 15, cy + 60], fill=dog_color, outline=(0, 0, 0))
        # Right ear
        draw.ellipse([cx + r - 15, cy - 25, cx + r + 20, cy + 60], fill=dog_color, outline=(0, 0, 0))
        
        # Draw main head circle
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=dog_color, outline=(0, 0, 0), width=2)
        
        # Draw eyes
        draw.ellipse([cx - 24, cy - 22, cx - 12, cy - 10], fill=(0, 0, 0))
        draw.ellipse([cx + 12, cy - 22, cx + 24, cy - 10], fill=(0, 0, 0))
        
        # Draw snout (cream/white colored oval)
        snout_color = (255, 239, 213) # PapayaWhip
        draw.ellipse([cx - 25, cy + 5, cx + 25, cy + 45], fill=snout_color, outline=(0, 0, 0))
        
        # Draw black nose circle on top of snout
        draw.ellipse([cx - 10, cy + 10, cx + 10, cy + 25], fill=(0, 0, 0))
        
        img.save(os.path.join(base_dir, "dogs", f"dog_{i}.jpg"))
        
    print(f"Synthetic dataset generation complete: 2 classes, {num_samples} images per class.\n")

def main():
    print("====================================================")
    print("Project 20: Transfer Learning with MobileNet")
    print("Goal: Use pre-trained MobileNetV2 from TF Hub as feature extractor")
    print("====================================================\n")

    dataset_dir = "dataset"
    
    # 1. Generate Dataset
    generate_synthetic_dataset(dataset_dir, num_samples=200)

    # 2. Load Dataset using tf_keras API
    print("Loading datasets using image_dataset_from_directory...")
    
    batch_size = 32
    img_size = (224, 224)
    seed = 123

    train_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(f"Detected class labels: {class_names}")

    # Prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # 3. Create Model using MobileNetV2 Feature Extractor from TF Hub
    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    print(f"\nLoading MobileNetV2 feature extractor from: {feature_extractor_url}...")
    
    # Create the TF Hub layer and freeze it
    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_url,
        input_shape=(224, 224, 3),
        trainable=False
    )

    # Assemble sequential model in tf_keras
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Rescaling(1./255),  # Normalizes pixel values to [0.0, 1.0]
        feature_extractor_layer,
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.summary()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Custom progress logger callback
    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch+1:2d}/8 - "
                  f"loss: {logs['loss']:.4f} - acc: {logs['accuracy']:.4f} | "
                  f"val_loss: {logs['val_loss']:.4f} - val_acc: {logs['val_accuracy']:.4f}")

    # 4. Train the Classification Head
    epochs = 8
    print("\nTraining classification head (MobileNetV2 feature extractor is frozen)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=0,
        callbacks=[ProgressCallback()]
    )
    print("Training finished.\n")

    # 5. Evaluate Validation Accuracy
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Final Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}\n")

    # 6. Retrieve a batch from validation set for showcasing predictions
    print("Generating predictions on validation sample subset...")
    for images, labels in val_ds.take(1):
        showcase_images = images.numpy()[:12]
        showcase_true_labels = labels.numpy()[:12]
        break
    
    # Run predictions
    predictions = model.predict(showcase_images, verbose=0).flatten()
    pred_labels = (predictions > 0.5).astype(int)

    # 7. Plotting Results
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8])

    # Left 2 columns: Predictions Showcase Grid
    ax_showcase = fig.add_subplot(gs[:, :2])
    ax_showcase.axis('off')
    ax_showcase.set_title("Cats vs Dogs Transfer Learning Showcase", fontsize=14, fontweight='bold', pad=15)

    # Plot 12 images in a 3x4 grid
    for row in range(3):
        for col in range(4):
            idx = row * 4 + col
            if idx >= len(showcase_images):
                break
                
            x_pos = col * 0.25 + 0.01
            y_pos = (2 - row) * 0.33 + 0.01
            ax_sub = ax_showcase.inset_axes([x_pos, y_pos, 0.23, 0.30])
            
            # Show image
            img_to_show = showcase_images[idx].astype('uint8')
            ax_sub.imshow(img_to_show)
            ax_sub.axis('off')
            
            true_idx = showcase_true_labels[idx]
            pred_idx = pred_labels[idx]
            prob = predictions[idx]
            
            conf = prob if pred_idx == 1 else (1.0 - prob)
            conf_pct = conf * 100
            
            true_lbl = class_names[true_idx]
            pred_lbl = class_names[pred_idx]
            
            correct = (true_idx == pred_idx)
            border_color = '#1e8e3e' if correct else '#d93025'
            
            # Draw colored box
            rect = plt.Rectangle((0, 0), 1, 1, fill=False, color=border_color, linewidth=3, transform=ax_sub.transAxes)
            ax_sub.add_patch(rect)
            
            # Overlay label
            text_str = f"Pred: {pred_lbl} ({conf_pct:.1f}%)\nTrue: {true_lbl}"
            ax_sub.text(0.05, 0.05, text_str, color='white', fontsize=8, fontweight='bold',
                        bbox=dict(facecolor=border_color, alpha=0.85, pad=3, edgecolor='none'),
                        transform=ax_sub.transAxes, verticalalignment='bottom')

    # Top Right: Loss Curves
    ax_loss = fig.add_subplot(gs[0, 2])
    epochs_range = range(1, epochs + 1)
    ax_loss.plot(epochs_range, history.history['loss'], 'o-', label='Train Loss', color='#1a73e8')
    ax_loss.plot(epochs_range, history.history['val_loss'], 's-', label='Val Loss', color='#d93025')
    ax_loss.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax_loss.set_xlabel('Epoch', fontsize=10)
    ax_loss.set_ylabel('Loss', fontsize=10)
    ax_loss.grid(True, linestyle='--', alpha=0.5)
    ax_loss.legend()

    # Bottom Right: Accuracy Curves
    ax_acc = fig.add_subplot(gs[1, 2])
    ax_acc.plot(epochs_range, history.history['accuracy'], 'o-', label='Train Accuracy', color='#1e8e3e')
    ax_acc.plot(epochs_range, history.history['val_accuracy'], 's-', label='Val Accuracy', color='#e37400')
    ax_acc.set_title(f'Accuracy Curves (Val Acc: {val_acc*100:.1f}%)', fontsize=12, fontweight='bold')
    ax_acc.set_xlabel('Epoch', fontsize=10)
    ax_acc.set_ylabel('Accuracy', fontsize=10)
    ax_acc.grid(True, linestyle='--', alpha=0.5)
    ax_acc.legend()

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, 'transfer_learning_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {plot_path}")
    print("====================================================")

if __name__ == "__main__":
    main()
