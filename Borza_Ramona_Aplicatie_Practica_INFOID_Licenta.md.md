# README - Proiect Licență: Detectarea pneumoniei cu ajutorul Deep Learningului

## Link Repository
https://github.com/Ramona27ramona/detector-pneumonie.git

---

## 📦 Structura Proiectului (Livrabile)
- `app/` - Aplicație interactivă Streamlit care detectează pneumonia
- `test_images/` - Folder cu imagini de test (JPG/PNG)
- `notebook.ipynb` - Notebook Kaggle pentru antrenarea modelului CNN
- `README.md` - Acest fișier explicativ

---

## Pași de instalare
```bash
git clone https://github.com/Ramona27ramona/detector-pneumonie.git
cd detector-pneumonie
```

---

## Pași de compilare/rulare

Ai nevoie de: Python 3.9+, Streamlit, TensorFlow, PIL, OpenCV, scikit-learn

1. Instalează dependențele:
```bash
pip install streamlit tensorflow pillow matplotlib opencv-python scikit-learn pandas
```

2. Rulează aplicația:
```bash
streamlit run app/app.py
```

După rulare, aplicația va fi deschisă automat în browser (de obicei la http://localhost:8501).

---

## Descriere pe scurt
Aplicația detectează automat pneumonia din radiografii pulmonare folosind un model CNN antrenat pe datasetul Chest X-ray (Paul Mooney). Oferă predicții binare (normală / pneumonie), scor de încredere, vizualizare Grad-CAM și funcționalitate de scanare automată a unui folder de imagini.

---

 Autor: **Borza Ramona**  
 Facultatea de Automatică și Calculatoare, UPT  
 Proiect coordonat de: **Dr. Ing. Mihaela Crișan-Vida**



Actualizare README.MD pentru livrabile licenta
