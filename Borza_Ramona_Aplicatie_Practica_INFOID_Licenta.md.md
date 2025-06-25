README - Proiect Licență: Detectarea pneumoniei cu ajutorul Deep Learningului



- Link Repository

https://github.com/Ramona27ramona/detector-pneumonie.git



- Structura Proiectului (Livrabile)

app.py - Aplicație interactivă Streamlit care detectează pneumonia

pneumonia_cnn_model.h5 - Modelul CNN antrenat (Keras)

test_images/ - Folder cu imagini de test (JPG/PNG)

notebook.ipynb - Notebook Kaggle pentru antrenarea modelului CNN

Prezentare_Pneumonie_Ramona_Borza.pptx - Prezentarea PowerPoint

documentatie.pdf - Documentația scrisa

README.md - Acest fișier explicativ




- Pași de instalare

1. Clonare repository

git clone https://github.com/Ramona27ramona/detector-pneumonie.git
cd detector-pneumonie


2. Pasi Instalare - compilare

Ai nevoie de: Python 3.9+, Streamlit, TensorFlow, PIL, OpenCV, scikit-learn

Faci un clone al repository-ului

pip install -r requirements.txt

Rulezi comanda streamlit run app.py

După rulare, aplicația va fi deschisă automat în browser (implicit la http://localhost:8501).



Sau individual:

pip install streamlit tensorflow pillow matplotlib opencv-python scikit-learn pandas






- Descriere pe scurt

Aplicația detectează automat pneumonia din radiografii pulmonare folosind un model CNN antrenat pe datasetul Chest X-ray (Paul Mooney). Este capabilă să ofere predicții, scor de încredere, Grad-CAM vizual pentru explicații și o scanare automata a unui folder cu imagini multiple.



Autor: Borza RamonaFacultatea de Automatică și Calculatoare, UPT Proiect coordonat de: Dr. Ing. Mihaela Crișan-Vida