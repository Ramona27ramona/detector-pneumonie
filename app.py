import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# =======================
# FUNCȚII UTILE
# =======================
def preprocess(img):
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img





def generate_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_1"):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, original_img.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(np.array(original_img), 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(superimposed)

# =======================
# INTERFAȚĂ
# =======================
st.set_page_config(page_title="Detector de Pneumonie", layout="centered")
st.title("🩻 Detector de Pneumonie din Radiografii")

# Încarcă modelul (functional, compatibil Grad-CAM)
model = tf.keras.models.load_model("pneumonia_cnn_model (2).h5")

# =======================
# ÎNCĂRCARE INDIVIDUALĂ
# =======================
st.subheader("📥 Încărcare individuală a unei radiografii")
uploaded_file = st.file_uploader("Încarcă o radiografie toracică (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Radiografia încărcată", use_column_width=True)

    img_array = preprocess(img)
    prediction = model.predict(img_array)[0][0]
    predicted_class = "PNEUMONIE" if prediction > 0.5 else "NORMALĂ"
    confidence = round(float(prediction)*100 if predicted_class == "PNEUMONIE" else (1 - float(prediction))*100, 2)

    st.markdown(f"### ✅ Predicție: **{predicted_class}**")
    st.markdown(f"### 🔬 Încredere: **{confidence}%**")

    heatmap = generate_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_1")
    cam_result = overlay_heatmap(img, heatmap)
    st.image(cam_result, caption="🧠 Grad-CAM (zonă relevantă pentru predicție)", use_column_width=True)

# =======================
# SCANARE AUTOMATĂ
# =======================
st.subheader("📁 Scanare automată folder test_images")
folder_path = "test_images"
results = []

if st.button("🔍 Scanează toate radiografiile"):
    pneumonia_count = 0
    normal_count = 0
    y_true = []
    y_pred = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path).convert("RGB")
            img_array = preprocess(img)
            prediction = model.predict(img_array)[0][0]

            predicted_class = "PNEUMONIE" if prediction > 0.5 else "NORMALĂ"
            confidence = round(float(prediction)*100 if predicted_class == "PNEUMONIE" else (1 - float(prediction))*100, 2)

            # presupunem eticheta reală din numele fișierului (ex: normal sau person)
            true_label = "PNEUMONIE" if "person" in file.lower() else "NORMALĂ"

            y_true.append(true_label)
            y_pred.append(predicted_class)
            results.append([file, predicted_class, confidence, true_label])

            st.image(img, caption=file, width=200)
            st.write(f"**Predicție:** {predicted_class} | **Încredere:** {confidence}% | Etichetă reală: {true_label}")
            st.markdown("---")

    # Rezumat numeric
    st.subheader("📊 Clasificare generală")
    fig, ax = plt.subplots()
    ax.bar(["NORMALĂ", "PNEUMONIE"], [y_pred.count("NORMALĂ"), y_pred.count("PNEUMONIE")], color=["green", "red"])
    ax.set_ylabel("Număr imagini")
    st.pyplot(fig)

    # Matrice de confuzie + acuratețe
    st.subheader("📏 Acuratețe și matrice de confuzie")
    acc = round(accuracy_score(y_true, y_pred)*100, 2)
    st.markdown(f"**Acuratețe generală:** {acc}%")

    cm = confusion_matrix(y_true, y_pred, labels=["NORMALĂ", "PNEUMONIE"])
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORMALĂ", "PNEUMONIE"])
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # Salvare CSV
    st.subheader("📄 Export rezultate în CSV")
    df = pd.DataFrame(results, columns=["Imagine", "Predicție", "Scor (%)", "Etichetă Reală"])
    csv_path = os.path.join(folder_path, "rezultate_predictii.csv")
    df.to_csv(csv_path, index=False)
    st.success(f"✅ Rezultatele au fost salvate în: `{csv_path}`")
