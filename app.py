from flask import Flask, flash, request, render_template, redirect, url_for
import matplotlib.pyplot as plt
import numpy as np
import math

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

Bw = 0 # en KHz
temperatura = 0 #en K

# transitores
signals = [
    {"fc": 0, "bw": 0, "p_max": 0},
    {"fc": 0, "bw": 4, "p_max": 0},
    {"fc": 0, "bw": 0, "p_max": 0}
]

def recoger_datos():
    global Bw, temperatura,signals
    Bw = float(request.args.get('Bw'))
    temperatura = float(request.args.get('temperatura'))

    # Reemplazar completamente los valores anteriores
    signals = []
    for i in range(3):
        fc = float(request.args.get(f'fc{i}'))
        bw = float(request.args.get(f'bw{i}'))
        p_max = float(request.args.get(f'p_max{i}'))
        signals.append({"fc": fc, "bw": bw, "p_max": p_max})

def signal_power(f, fc, bw, p_max):
    x = (f - fc) / (bw / 2)
    attenuation = 3 * (x ** 2)
    power = p_max - attenuation
    return power

@app.route('/generar_grafica', methods=['GET'])
def generar_grafica():
    
    global Bw, temperatura, signals
    #Calcular ruido termico
    recoger_datos()
    Bw*= 1000 #Pasar de KHz a Hz
    k = 1.38*10**-23
    noise_floor = 10 * math.log10(k*temperatura*Bw) + 30 # en dBm
    colors = ['red', 'green', 'blue']

    # rango de frecuencia
    min_left = float('inf')
    max_right = float('-inf')
    for s in signals:
        left = s["fc"] - (s["bw"] / 2)
        right = s["fc"] + (s["bw"] / 2)
        min_left = min(min_left, left)
        max_right = max(max_right, right)

    frequencies = np.linspace(min_left - 10, max_right + 10, 3000)

    np.random.seed(42)
    noise = noise_floor + np.random.normal(0, 0.5, size=frequencies.shape)

    # grafico
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, noise, linestyle='--', color='gray', label="Ruido térmico")

    max_p_max = 0
    for idx, s in enumerate(signals):
        max_p_max = max(max_p_max,s["p_max"])
        power = signal_power(frequencies, s["fc"], s["bw"], s["p_max"])
        plt.plot(frequencies, power, color=colors[idx], linewidth=2.5, label=f"Señal {idx+1} Fc={s['fc']} MHz")
        
        plt.axvline(x=s["fc"], color=colors[idx], linestyle='-', linewidth=1)
        
        # media potencia
        level = 1
        left = s["fc"] - (s["bw"]/2) * level
        right = s["fc"] + (s["bw"]/2) * level
        plt.axvline(x=left, color=colors[idx], linestyle=':', linewidth=1)
        plt.axvline(x=right, color=colors[idx], linestyle=':', linewidth=1)

    # configuracion
    plt.title("Gráfica de Espectro con Curvas Suaves y Nivel de -3 dB")
    plt.xlabel("Frecuencia (MHz)")
    plt.ylabel("Potencia (dBm)")
    plt.ylim(noise_floor - 10, max_p_max+10)
    plt.xlim(min_left - 10, max_right + 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# Endpoint para cuando se genere un error
# @app.errorhandler(404)
# def not_found_endpoint(error):
#     return redirect('index.html')

if __name__ == '__main__':
    app.run(debug=True)
