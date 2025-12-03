import torch
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile
import numpy as np
import io
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN GLOBAL ---
MODEL_ID = "facebook/mms-tts-spa"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"⏳ Cargando modelo TTS en {device}... (Esto puede tardar un poco)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = VitsModel.from_pretrained(MODEL_ID).to(device)
print("✅ Modelo cargado y listo para recibir peticiones.")

@app.route('/tts', methods=['POST'])
def text_to_speech_api():
    try:
        # 1. Obtener datos de la petición (JSON)
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Falta el campo 'text' en el JSON"}), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({"error": "El texto no puede estar vacío"}), 400

        # 2. Preprocesamiento
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # 3. Inferencia (Generar Audio)
        with torch.no_grad():
            output = model(**inputs).waveform

        # 4. Procesar la salida a numpy
        audio_data = output.cpu().float().numpy().squeeze()
        sampling_rate = model.config.sampling_rate

        # 5. Guardar en memoria (Buffer) en lugar de disco
        # Creamos un archivo 'virtual' en memoria RAM
        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, rate=sampling_rate, data=audio_data)
        
        # Rebobinamos el buffer al inicio para que pueda ser leído
        buffer.seek(0)

        # 6. Retornar el binario directamente
        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=False, # False para que el navegador/cliente intente reproducirlo
            download_name="sintesis.wav"
        )

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ejecutar en el puerto 5000
    app.run(host='0.0.0.0', port=5000, debug=False)