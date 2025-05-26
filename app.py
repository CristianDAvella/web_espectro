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
    global Bw, temperatura,signals, ruido_manual
    ruido_manual = None
    modo_ruido = request.args.get('modo_ruido')
    # Lee valores según el modo elegido
    if modo_ruido == 'manual':
        ruido_str = request.args.get('ruido')
        try:
            ruido_manual = float(ruido_str)
            temperatura = 0
            Bw = 0
        except (TypeError, ValueError):
            ruido_manual = None
    else:
        ruido_manual = None
        temperatura = float(request.args.get('temperatura'))
        Bw = float(request.args.get('Bw'))

    # Reemplazar completamente los valores anteriores
    signals = []
    for i in range(3):
        fc = float(request.args.get(f'fc{i}'))
        bw = float(request.args.get(f'bw{i}'))
        p_max = float(request.args.get(f'p_max{i}'))
        signals.append({"fc": fc, "bw": bw, "p_max": p_max})

def signal_power(f, fc, bw, p_max):
    x = (f - fc) / (bw / 2)
    return p_max - 3 * (x ** 2)

def calcular_snr_y_comparaciones(signals, noise_floor):
    resumen = []

    for i, s in enumerate(signals):
        snr = s["p_max"] - noise_floor
        comparaciones = []
        for j, t in enumerate(signals):
            if i != j:
                delta = s["p_max"] - t["p_max"]
                signo = "+" if delta >= 0 else "–"
                comparaciones.append(f"{signo}{abs(round(delta, 2))} vs Señal{j+1}")
        resumen.append({
            "nombre": f"{i+1}",
            "fc": s["fc"],
            "bw": s["bw"],
            "potencia": s["p_max"],
            "snr": round(snr, 2),
            "comparacion": "<br>".join(comparaciones)
        })
    return resumen

@app.route('/generar_grafica', methods=['GET'])
def generar_grafica():
    
    global Bw, temperatura, signals
    #Calcular ruido termico
    recoger_datos()
    Bw*= 1000 #Pasar de KHz a Hz
    k = 1.38*10**-23
    if ruido_manual is not None:
        noise_floor = ruido_manual
    else:
        Bw *= 1000  # kHz → Hz solo si se calcula automáticamente
        noise_floor = 10 * math.log10(k * temperatura * Bw) + 30  # en dBm

    colors = ['red', 'green', 'blue']

    # rango de frecuencia
    min_left = 1e308
    max_right= 1e-308
    for i in range(3):
        fc = signals[i]["fc"]
        bw = signals[i]["bw"]
        p_max = signals[i]["p_max"]

        power_drop = p_max - noise_floor
        if power_drop > 0:  
            x_range = (bw / 2) * math.sqrt(power_drop / 3)
            f_left  = fc - x_range
            f_right = fc + x_range
            min_left = min(min_left, f_left)
            max_right = max(max_right, f_right)
            print(f"Signal {i+1}: x from {f_left:.2f} to {f_right:.2f}")
        else:
            print(f"Signal {i+1}: too weak, always below noise floor")
    #────────────────────────────────────────────

    #────────────────────────────────────────────
    # Determinar el rango completo de frecuencias
    frequencies = np.linspace(min_left-5, max_right+5, 5000)

    #────────────────────────────────────────────
    # Calcular la curva de cada señal
    p1 = signal_power(frequencies, signals[0]["fc"], signals[0]["bw"], signals[0]["p_max"])
    p2 = signal_power(frequencies, signals[1]["fc"], signals[1]["bw"], signals[1]["p_max"])
    p3 = signal_power(frequencies, signals[2]["fc"], signals[2]["bw"], signals[2]["p_max"])

    #────────────────────────────────────────────
    # [Cruce 1] Entre la señal 1 y la señal 2
    mask_12 = (frequencies >= signals[0]["fc"]) & (frequencies <= signals[1]["fc"])
    diff_12 = p1[mask_12] - p2[mask_12]
    cross_12 = np.where(np.diff(np.sign(diff_12)) != 0)[0]
    if cross_12.size > 0:
        idx     = cross_12[0] + 1
        f_cr_12 = frequencies[mask_12][idx]
        p1[frequencies >= f_cr_12] = np.nan
        p2[frequencies <  f_cr_12] = np.nan

    #────────────────────────────────────────────
    # [Cruce 2] Entre la señal 2 y la señal 3
    mask_23 = (frequencies >= signals[1]["fc"]) & (frequencies <= signals[2]["fc"])
    diff_23 = p2[mask_23] - p3[mask_23]
    cross_23 = np.where(np.diff(np.sign(diff_23)) != 0)[0]
    if cross_23.size > 0:
        idx     = cross_23[0] + 1
        f_cr_23 = frequencies[mask_23][idx]
        p2[frequencies >= f_cr_23] = np.nan
        p3[frequencies <  f_cr_23] = np.nan

    #────────────────────────────────────────────
    # [Cruce 3] Entre la señal 1 y la señal 3
    start_13 = signals[0]["fc"] + signals[0]["bw"]/2
    end_13   = signals[2]["fc"] - signals[2]["bw"]/2

    mask_13_raw   = (frequencies >= start_13) & (frequencies <= end_13)
    mask_13_valid = mask_13_raw & (~np.isnan(p1)) & (~np.isnan(p3))

    diff_13 = p1[mask_13_valid] - p3[mask_13_valid]
    cross_13 = np.where(np.diff(np.sign(diff_13)) != 0)[0]
    if cross_13.size > 0:
        idx      = cross_13[0] + 1
        freqs13  = frequencies[mask_13_valid]
        f_cr_13  = freqs13[idx]
        p1[frequencies >= f_cr_13] = np.nan
        p3[frequencies <  f_cr_13] = np.nan

    #────────────────────────────────────────────
    # Calcular el ruido térmico y eliminar las áreas por debajo del nivel de ruido
    np.random.seed(42)
    noise = noise_floor + np.random.normal(0, 0.5, size=frequencies.shape)
    p1[p1 < noise] = np.nan
    p2[p2 < noise] = np.nan
    p3[p3 < noise] = np.nan

    #────────────────────────────────────────────
    # Eliminar señales solapadas: conservar solo la potencia más alta en cada frecuencia
    stacked_signals = np.vstack((p1, p2, p3))
    max_power = np.nanmax(stacked_signals, axis=0)
    p1 = np.where(p1 < max_power, np.nan, p1)
    p2 = np.where(p2 < max_power, np.nan, p2)
    p3 = np.where(p3 < max_power, np.nan, p3)

    #────────────────────────────────────────────

    # Eliminar lineas de ruido termico que estan debajo de las curvas
    masked_noise = np.where(
        np.isnan(p1) & np.isnan(p2) & np.isnan(p3),
        noise,
        np.nan
    )

    # ────────────────────────────────────────────
    # Dibujar grafica
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, masked_noise, linestyle='--', color='gray', label="Ruido térmico")
    plt.plot(frequencies, p1, color=colors[0], linewidth=2.5, label=f'Señal 1 (fc={signals[0]["fc"]} MHz)')
    plt.plot(frequencies, p2, color=colors[1], linewidth=2.5, label=f'Señal 2 (fc={signals[1]["fc"]} MHz)')
    plt.plot(frequencies, p3, color=colors[2], linewidth=2.5, label=f'Señal 3 (fc={signals[2]["fc"]} MHz)')
    # ... 이하 생략 ...

    plt.plot(frequencies, p1, color=colors[0], linewidth=2.5, label=f'Señal 1 (fc={signals[0]["fc"]} MHz)')
    plt.plot(frequencies, p2, color=colors[1], linewidth=2.5, label=f'Señal 2 (fc={signals[1]["fc"]} MHz)')
    plt.plot(frequencies, p3, color=colors[2], linewidth=2.5, label=f'Señal 3 (fc={signals[2]["fc"]} MHz)')

    for s, col in zip(signals, colors):
        plt.axvline(s["fc"], color=col, linestyle='-',  linewidth=1)
        plt.axvline(s["fc"]-(s["bw"]/2), color=col, linestyle=':', linewidth=1)
        plt.axvline(s["fc"]+(s["bw"]/2), color=col, linestyle=':', linewidth=1)

    plt.title("Gráfica de Espectro")
    plt.xlabel("Frecuencia (MHz)")
    plt.ylabel("Potencia (dBm)")
    plt.ylim(noise_floor - 10, max(s["p_max"] for s in signals) + 10)
    valid_x = frequencies[~np.isnan(p1) | ~np.isnan(p2) | ~np.isnan(p3)]
    if valid_x.size > 0:
        plt.xlim(valid_x.min()-5, valid_x.max()+5)
    else:
        print("No signal above noise floor. Plotting default range.")
        plt.xlim(frequencies.min(), frequencies.max())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #Guradmamos la imagen del grafico para mostrarla en otra interfaz en un archivo llamado plot.png
    plt.savefig("static/plot.png")
    resumen = calcular_snr_y_comparaciones(signals, noise_floor)
    #plt.show() #Ventana emergente mostrando la grafica
    plt.close() #PAra evitar problemas al intentar generar varias graficas
    return render_template('grafica.html',image_url='static/plot.png', resumen=resumen)
    
    
#Endpoint para cuando se genere un error
@app.errorhandler(404)
def not_found_endpoint(error):
    return redirect('index.html')

if __name__ == '__main__':
    app.run(debug=True)
