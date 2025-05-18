from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo
modelo = pickle.load(open('modelo_lr.pkl', 'rb'))

@app.route('/')
def formulario():
    return render_template('formulario.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    valores = [float(x) for x in request.form.values()]
    datos = np.array([valores])
    prediccion = modelo.predict(datos)[0]
    return render_template('formulario.html', prediccion=prediccion)

if __name__ == '__main__':
    app.run(debug=True)