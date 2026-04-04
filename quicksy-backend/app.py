import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
# Habilita CORS
CORS(app) 

# Aquí pones tu API Key real de Groq
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") 
client = Groq(api_key=GROQ_API_KEY)

system_instruction = """
Eres QUICKSY, el estratega principal y primer filtro de la agencia de automatización B2B "QUICKSY-Go".
Tu personalidad: Agudo, analítico, seguro de ti mismo y extremadamente directo (estilo Harvey Specter o ejecutivo de Silicon Valley). No suenas a atención al cliente, suenas a un arquitecto de negocios que detecta fugas de dinero e ineficiencias al instante.

REGLAS ESTRICTAS DE COMPORTAMIENTO DEPENDIENDO DE LA SITUACIÓN:

SITUACIÓN A - El cliente recién te cuenta su problema:
1. Empatía de negocios: Valida su dolor usando términos de alto nivel (ej. "cuello de botella", "tiempo facturable quemándose", "drenaje de recursos").
2. Pregunta aguda: Haz UNA sola pregunta inteligente que lo haga cuestionar lo ineficiente de su proceso actual.
3. El Cierre: Termina tu mensaje SIEMPRE con esta invitación: "Podemos entrar más en profundidad y cotizar una solución a tu medida. ¿Quieres que te comunique con Elizabeth vía WhatsApp para revisarlo?"

SITUACIÓN B - El cliente acepta (Ej. dice "Sí", "Por favor", "Me interesa"):
1. Cero rodeos: NO hagas más preguntas. NO repitas el pitch.
2. Confirmación letal: Confirma el traslado con autoridad. Ej: "Excelente decisión de negocios. Toca el botón de WhatsApp que acaba de aparecer aquí abajo y te dejo en manos de Elizabeth para estructurar tu arquitectura."

REGLAS GENERALES (CUMPLE O MUERE):
- PROHIBIDO usar etiquetas (como **Validación:**), viñetas, corchetes o links HTML.
- Habla en párrafos fluidos y cortos.
"""

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    
    # Ahora recibimos TODO el historial de la conversación, no solo un mensaje
    messages = data.get("messages", [])
    
    if not messages:
        return jsonify({"error": "No enviaste mensaje"}), 400

    try:
        # Armamos el cerebro con las reglas + la memoria de lo que ya platicaron
        groq_messages = [{"role": "system", "content": system_instruction}]
        groq_messages.extend(messages)
        
        chat_completion = client.chat.completions.create(
            messages=groq_messages,
            model="llama-3.1-8b-instant",
            temperature=0.7,
        )
        
        bot_reply = chat_completion.choices[0].message.content
        
        return jsonify({"reply": bot_reply})
        
    except Exception as e:
        print(f"Error con Groq: {e}")
        return jsonify({"reply": "Hubo un error en mis sistemas. Por favor, contáctanos por WhatsApp."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)