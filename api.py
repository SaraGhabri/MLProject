from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Charger les modèles
logistic_regression = joblib.load('logistic_regression.pkl')
linear_svc = joblib.load('linear_svc.pkl')
knn = joblib.load('knn.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    # Réinitialiser le pointeur du fichier
    file.seek(0)
    
    try:
        # Lire le fichier en mémoire
        file_data = file.read()
        
        # Ouvrir l'image à partir des données en mémoire
        image = Image.open(io.BytesIO(file_data)).convert('L').resize((28, 28))
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

    # Convertir l'image en tableau NumPy
    image_array = np.array(image).reshape(1, -1) / 255.0

    # Récupérer le modèle sélectionné
    model_name = request.form.get('model')
    if model_name == "Logistic Regression":
        prediction = logistic_regression.predict(image_array)
    elif model_name == "Linear SVC":
        prediction = linear_svc.predict(image_array)
    elif model_name == "KNN":
        prediction = knn.predict(image_array)
    else:
        return jsonify({"error": "Invalid model name"}), 400

    # Renvoyer uniquement la prédiction
    return jsonify({
        "prediction": int(prediction[0])  # Convertir en entier pour éviter les problèmes de sérialisation
    })

if __name__ == '__main__':
    app.run(debug=True)
