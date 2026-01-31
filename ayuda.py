import os
import random
from collections import defaultdict
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from IPython.display import display, HTML
from PIL import Image, UnidentifiedImageError
from torchmetrics.classification import Precision, Recall
from torchvision import transforms
from tqdm.auto import tqdm


def entrenamiento_loop(modelo, entrenamiento_cargado, validacion_cargado, funcion_perdida, optimizador, dispositivo, num_epocas=3):
    
    # Crear el directorio donde se guardará el mejor modelo si no existe
    save_dir = "./mejor_modelo/"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "mejor_modelo.pth")

    # Mover el modelo al dispositivo de cálculo especificado (CPU o GPU)
    modelo.to(dispositivo)
    # Asignar la función de pérdida proporcionada
    loss_function = funcion_perdida
    # Asignar el optimizador proporcionado
    optimizer = optimizador
    # Determinar el número de clases a partir de la última capa del clasificador
    num_classes = modelo.classifier[-1].out_features
    # Inicializar las métricas de accuracy, precision y recall para validación
    val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='macro').to(dispositivo)
    val_precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(dispositivo)
    val_recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(dispositivo)
    
    # Inicializar variables para almacenar el mejor rendimiento
    best_val_accuracy = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0

    # Bucle principal de entrenamiento y validación por épocas
    for epoch in range(num_epocas):
        # Poner el modelo en modo entrenamiento
        modelo.train()
        # Variable para acumular la pérdida de entrenamiento
        running_loss = 0.0
        # Contador de predicciones correctas en entrenamiento
        total_train_correct = 0
        # Contador del número total de muestras de entrenamiento
        total_train_samples = 0

        # Barra de progreso para los batches de entrenamiento
        barra_progreso = tqdm(entrenamiento_cargado, desc=f"Epoca {epoch + 1}/{num_epocas} Entrenando", unit="batch")
        # Iterar sobre los batches del dataloader de entrenamiento
        for images, labels in barra_progreso:
            # Mover imágenes y etiquetas al dispositivo
            images, labels = images.to(dispositivo), labels.to(dispositivo)
            # Reiniciar los gradientes
            optimizer.zero_grad()
            # Forward pass del modelo
            outputs = modelo(images)
            # Calcular la pérdida
            loss = loss_function(outputs, labels)
            # Backpropagation
            loss.backward()
            # Actualizar los pesos del modelo
            optimizer.step()

            # Acumular la pérdida y los contadores
            running_loss += loss.item() * labels.size(0)
            # Obtener la clase predicha
            _, predicted = torch.max(outputs, dim=1)
            # Contar predicciones correctas
            total_train_correct += (predicted == labels).sum().item()
            # Contar muestras procesadas
            total_train_samples += labels.size(0)
            
            # Calcular la pérdida media de la época
            epoch_loss = running_loss / total_train_samples
            # Calcular la accuracy de la época
            epoch_acc = 100 * total_train_correct / total_train_samples
            # Actualizar la barra de progreso con pérdida y accuracy
            barra_progreso.set_postfix(loss=f"{epoch_loss:.4f}", accuracy=f"{epoch_acc:.2f}%")

        # ----- FASE DE VALIDACIÓN -----
        # Poner el modelo en modo evaluación
        modelo.eval()
        # Contador de muestras de validación
        total_val_samples = 0
        # Variable para acumular la pérdida de validación
        val_loss = 0.0
        
        # Reiniciar las métricas de validación
        val_accuracy_metric.reset()
        val_precision_metric.reset()
        val_recall_metric.reset()

        # Desactivar el cálculo de gradientes durante la validación
        with torch.no_grad():
            # Barra de progreso para los batches de validación
            val_progress_bar = tqdm(validacion_cargado, desc=f"Epocas {epoch + 1}/{num_epocas} Validando", unit="batch")
            # Iterar sobre los batches del dataloader de validación
            for images, labels in val_progress_bar:
                # Mover imágenes y etiquetas al dispositivo
                images, labels = images.to(dispositivo), labels.to(dispositivo)
                # Forward pass
                outputs = modelo(images)
                # Calcular la pérdida
                loss = loss_function(outputs, labels)
                # Acumular la pérdida
                val_loss += loss.item() * labels.size(0)
                # Obtener la clase predicha
                _, predicted = torch.max(outputs, dim=1)
                # Contar muestras procesadas
                total_val_samples += labels.size(0)

                # Actualizar las métricas
                val_accuracy_metric.update(predicted, labels)
                val_precision_metric.update(predicted, labels)
                val_recall_metric.update(predicted, labels)
                
                # Actualizar la barra de progreso con la accuracy actual
                val_progress_bar.set_postfix(
                    accuracy=f"{100 * val_accuracy_metric.compute():.2f}%"
                )

        # Calcular la pérdida media de validación
        avg_val_loss = val_loss / total_val_samples
        
        # Calcular las métricas finales de la época
        final_val_acc = val_accuracy_metric.compute()
        final_val_precision = val_precision_metric.compute()
        final_val_recall = val_recall_metric.compute()
        
        # Mostrar resumen de resultados de validación
        print(f'Validacion pérdida (Avg): {avg_val_loss:.4f}, Validación acierto: {final_val_acc * 100:.2f}%\n')

        # Guardar el modelo si mejora la accuracy de validación
        if final_val_acc > best_val_accuracy:
            best_val_accuracy = final_val_acc
            best_val_precision = final_val_precision
            best_val_recall = final_val_recall
            torch.save(modelo.state_dict(), best_model_path)
            print(f"Nuevo mejor modelo guardado en {best_model_path} con Validación acierto: {best_val_accuracy * 100:.2f}%\n")

    # Mensaje de finalización del entrenamiento
    print("\nEntrenamiento finalizado. Se devuelve el mejor modelo entrenado.")
    print(f"Mejor Validación acierto: {best_val_accuracy * 100:.2f}%")
    print(f"Mejor Validación Precision: {best_val_precision:.4f}")
    print(f"Mejor Validación recuperación: {best_val_recall:.4f}\n")
    
    # Cargar los pesos del mejor modelo antes de devolverlo
    modelo.load_state_dict(torch.load(best_model_path))
    
    # Devolver el mejor modelo entrenado
    return modelo


def mostrar_predicciones(modelo_entrenado, validacion_cargado, dispositivo, n=6):
    modelo_entrenado.eval()

    dataset = validacion_cargado.dataset
    clases = dataset.classes

    # Elegimos n imágenes aleatorias del dataset (cambia cada vez)
    idxs = random.sample(range(len(dataset)), k=min(n, len(dataset)))

    # Valores ImageNet (ConvNeXt)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    cols = 3 if n >= 3 else n
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 4 * rows))

    with torch.no_grad():
        for j, idx in enumerate(idxs):
            img, label = dataset[idx]                 # img ya viene transformada
            x = img.unsqueeze(0).to(dispositivo)

            pred = modelo_entrenado(x).argmax(dim=1).item()
            correcto = (label == pred)

            # Desnormalizar para mostrar “bonito”
            img_show = img.cpu().permute(1, 2, 0)
            img_show = img_show * std + mean
            img_show = img_show.clamp(0, 1)

            ax = plt.subplot(rows, cols, j + 1)
            ax.imshow(img_show)
            ax.axis("off")

            ax.set_title(
                f"Real: {clases[label]}\nPred: {clases[pred]}",
                fontsize=11,
                fontweight="bold",
                bbox=dict(
                    facecolor="lightgreen" if correcto else "salmon",
                    alpha=0.85,
                    edgecolor="none",
                    boxstyle="round,pad=0.35"
                ),
                pad=6
            )

    plt.tight_layout()
    plt.show()



def predecir_imagen_local(modelo_entrenado, ruta_imagen, transform, dispositivo, clases):
    modelo_entrenado.eval()

    # Cargar imagen original
    imagen_original = Image.open(ruta_imagen).convert("RGB")

    # Transform para el modelo
    imagen = transform(imagen_original).unsqueeze(0).to(dispositivo)

    with torch.no_grad():
        output = modelo_entrenado(imagen)
        pred = output.argmax(dim=1).item()

    # Mostrar imagen ORIGINAL (sin normalizar)
    plt.figure(figsize=(4, 4))
    plt.imshow(imagen_original)
    plt.axis("off")

    plt.title(
        f"Predicción: {clases[pred]}",
        fontsize=13,
        fontweight="bold",
        bbox=dict(
            facecolor="lightblue",
            alpha=0.85,
            edgecolor="none",
            boxstyle="round,pad=0.4"
        )
    )

    plt.show()

    return pred

