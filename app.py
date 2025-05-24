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
    return p_max - 3 * (x ** 2)

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
    # 신호 전파 모델 함수 (포물선 형태)
    #────────────────────────────────────────────
    # 전체 주파수 범위 결정
    frequencies = np.linspace(min_left, max_right, 5000)

    #────────────────────────────────────────────
    # 각 신호 곡선 계산
    p1 = signal_power(frequencies, signals[0]["fc"], signals[0]["bw"], signals[0]["p_max"])
    p2 = signal_power(frequencies, signals[1]["fc"], signals[1]["bw"], signals[1]["p_max"])
    p3 = signal_power(frequencies, signals[2]["fc"], signals[2]["bw"], signals[2]["p_max"])

    #────────────────────────────────────────────
    # [교차 처리 1] 신호 1과 신호 2 사이
    mask_12 = (frequencies >= signals[0]["fc"]) & (frequencies <= signals[1]["fc"])
    diff_12 = p1[mask_12] - p2[mask_12]
    cross_12 = np.where(np.diff(np.sign(diff_12)) != 0)[0]
    if cross_12.size > 0:
        idx     = cross_12[0] + 1
        f_cr_12 = frequencies[mask_12][idx]
        p1[frequencies >= f_cr_12] = np.nan
        p2[frequencies <  f_cr_12] = np.nan

    #────────────────────────────────────────────
    # [교차 처리 2] 신호 2와 신호 3 사이
    mask_23 = (frequencies >= signals[1]["fc"]) & (frequencies <= signals[2]["fc"])
    diff_23 = p2[mask_23] - p3[mask_23]
    cross_23 = np.where(np.diff(np.sign(diff_23)) != 0)[0]
    if cross_23.size > 0:
        idx     = cross_23[0] + 1
        f_cr_23 = frequencies[mask_23][idx]
        p2[frequencies >= f_cr_23] = np.nan
        p3[frequencies <  f_cr_23] = np.nan

    #────────────────────────────────────────────
    # [교차 처리 3] 신호 1과 신호 3 사이 (유효 구간만)
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
    # 열 잡음 (ruido térmico) 계산 및 신호 아래 영역 삭제
    np.random.seed(42)
    noise = noise_floor + np.random.normal(0, 0.5, size=frequencies.shape)
    p1[p1 < noise] = np.nan
    p2[p2 < noise] = np.nan
    p3[p3 < noise] = np.nan

    #────────────────────────────────────────────
    # ★ 덮인 신호 제거: 각 주파수에서 가장 높은 전력만 남기기 ★
    stacked_signals = np.vstack((p1, p2, p3))
    max_power = np.nanmax(stacked_signals, axis=0)
    p1 = np.where(p1 < max_power, np.nan, p1)
    p2 = np.where(p2 < max_power, np.nan, p2)
    p3 = np.where(p3 < max_power, np.nan, p3)

    #────────────────────────────────────────────
    # 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, noise, linestyle='--', color='gray', label="Ruido térmico")
    plt.plot(frequencies, p1, color=colors[0], linewidth=2.5, label="Señal 1 (fc=200 MHz)")
    plt.plot(frequencies, p2, color=colors[1], linewidth=2.5, label="Señal 2 (fc=210 MHz)")
    plt.plot(frequencies, p3, color=colors[2], linewidth=2.5, label="Señal 3 (fc=220 MHz)")

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
    #plt.show() #Ventana emergente mostrando la grafica
    plt.close() #PAra evitar problemas al intentar generar varias graficas
    return render_template('grafica.html',image_url='static/plot.png')
    
    
#Endpoint para cuando se genere un error
@app.errorhandler(404)
def not_found_endpoint(error):
     return redirect('index.html')

if __name__ == '__main__':
    app.run(debug=True)
