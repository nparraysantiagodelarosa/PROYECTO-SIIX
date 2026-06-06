import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import joblib

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

try:
    columnas_encoded = pd.read_csv(r"C:\Users\marma\OneDrive\Escritorio\prueba app\columnas_encodedBASENUEVA.txt", header=None)[0].tolist()
    columnas_originales = pd.read_csv(r"C:\Users\marma\OneDrive\Escritorio\prueba app\columnas_originalesBASENUEVA.txt", header=None)[0].tolist()
    print(f"✓ Columnas cargadas: {len(columnas_originales)} originales, {len(columnas_encoded)} codificadas")
except Exception as e:
    print(f"Error cargando columnas: {e}")
    messagebox.showerror("Error", f"No se pudieron cargar las columnas:\n{e}")
    exit()

try:
    scaler = joblib.load(r"C:\Users\marma\OneDrive\Escritorio\prueba app\scalerBASENUEVA.pkl")
    print("✓ Scaler cargado correctamente")
except Exception as e:
    print(f"Advertencia: No se encontró scalerBASENUEVA.pkl: {e}")
    scaler = None
    
input_size = len(columnas_encoded)
hidden_layers = [32]
dropout_rate = 0.2
model = FlexibleMLP(input_dim=input_size, hidden_layers=hidden_layers, dropout_rate=dropout_rate)

try:
    state_dict = torch.load(r"C:\Users\marma\OneDrive\Escritorio\prueba app\modelo_entrenadoBASENUEVA.pth", 
                           map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print(" Modelo cargado correctamente")
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")
    exit()
# Construir opciones desde las columnas encoded
print("Construyendo opciones desde las columnas codificadas...")
opciones = {}

# Extraer opciones de las columnas codificadas
for col_encoded in columnas_encoded:
    if '_' in col_encoded:
        # Verificar si la primera parte está en columnas_originales
        col_name = col_encoded.split('_')[0]
        if col_name in columnas_originales:
            valor = col_encoded.split('_')[1]
            if col_name not in opciones:
                opciones[col_name] = []
            if valor not in opciones[col_name]:
                opciones[col_name].append(valor)

for col in opciones:
    opciones[col].sort()

print(f"Opciones construidas para {len(opciones)} columnas")

#Identificar columnas num
columnas_numericas = []
for col in columnas_originales:
    if col not in opciones:
        columnas_numericas.append(col)
print(f" Columnas numéricas: {len(columnas_numericas)}")

# Mostrar todas las columnas para depurar
print("\nCOLUMNAS ORIGINALES")
for col in columnas_originales:
    if col in opciones:
        print(f"  {col} (categórica) - {len(opciones[col])} opciones")
    else:
        print(f"  {col} (numérica)")

ventana = tk.Tk()
ventana.title("Predicción de duración")
ventana.geometry("900x750")
main_frame = ttk.Frame(ventana)
main_frame.pack(fill="both", expand=True)
canvas = tk.Canvas(main_frame)
scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")
# Titulo
titulo = ttk.Label(scrollable_frame, text="Sistema de predicción de rotación de personal", 
                   font=('Arial', 14, 'bold'))
titulo.grid(row=0, column=0, columnspan=2, pady=10)
# Guardar las entradas
entradas = {}
# Crear widgets
for i, columna in enumerate(columnas_originales):
    label = ttk.Label(scrollable_frame, text=columna, width=40, anchor="w")
    label.grid(row=i+1, column=0, sticky="w", pady=5, padx=10)
    if columna in opciones:
        # Combobox
        combo = ttk.Combobox(scrollable_frame, values=opciones[columna], 
                            state="readonly", width=35)
        combo.grid(row=i+1, column=1, pady=5, padx=10)
        entradas[columna] = combo
    else:
        entry = ttk.Entry(scrollable_frame, width=37)
        entry.grid(row=i+1, column=1, pady=5, padx=10)
        entradas[columna] = entry
        # Agregar valores sugeridos para campos num
        if columna.upper() == "SALARIO DIARIO ANTERIOR":
            entry.insert(0, "250")
        elif columna.upper() == "TIEMPO DE TRASLADO A SIIX MINUTOS":
            entry.insert(0, "30")
        elif columna.upper() == "CANTIDAD":
            entry.insert(0, "0")

def predecir():
    for col, widget in entradas.items():
        valor = widget.get()
        if valor == "":
            messagebox.showwarning("Falta información", f"Falta llenar el campo: {col}")
            return
    datos_onehot = {col: 0 for col in columnas_encoded}
    
    for col, widget in entradas.items():
        valor = widget.get().strip()
        if col in columnas_numericas:
            try:
                valor_num = float(valor)
                if col in datos_onehot:
                    datos_onehot[col] = valor_num
                else:
                    print(f"Advertencia: {col} no encontrada en columnas_encoded")
            except ValueError:
                messagebox.showerror("Error", f"El campo '{col}' debe ser un número válido.\nValor ingresado: {valor}")
                return
        else:
            valor_upper = str(valor).upper().strip()
            encontrado = False
            for encoded_col in columnas_encoded:
                if encoded_col.startswith(col + "_"):
                    valor_encoded = encoded_col[len(col)+1:]
                    if valor_encoded.replace(" ", "") == valor_upper.replace(" ", ""):
                        datos_onehot[encoded_col] = 1
                        encontrado = True
                        break
            if not encontrado:
                for encoded_col in columnas_encoded:
                    if encoded_col.startswith(col + "_"):
                        valor_encoded = encoded_col[len(col)+1:]
                        if valor_upper in valor_encoded or valor_encoded in valor_upper:
                            datos_onehot[encoded_col] = 1
                            encontrado = True
                            print(f"Match flexible: {col}={valor} -> {encoded_col}")
                            break
                
                if not encontrado:
                    print(f"Advertencia: No se encontró codificación para {col}={valor}")
                    # Mostrar opciones disponibles para depuración
                    opciones_disponibles = [ec[len(col)+1:] for ec in columnas_encoded if ec.startswith(col + "_")]
                    print(f"Opciones disponibles para {col}: {opciones_disponibles}")
    
    vector_caracteristicas = [datos_onehot[col] for col in columnas_encoded]
    
    # Aplicar el mismo scaler usado en entrenamiento
    if scaler is not None:
        try:
            vector_caracteristicas = scaler.transform([vector_caracteristicas])[0]
        except Exception as e:
            print(f"Error aplicando scaler: {e}")
            messagebox.showerror("Error", f"Error al normalizar los datos:\n{e}")
            return
    
    # Convertir a tensor
    X = torch.tensor([vector_caracteristicas], dtype=torch.float32)
    
    # Predicción
    try:
        with torch.no_grad():
            pred = model(X).item()
    except Exception as e:
        messagebox.showerror("Error", f"Error durante la predicción:\n{e}")
        return
    
    # Resultado
    if pred >= 0.5:
        resultado = "PERMANECE > 90 DÍAS"
        probabilidad = pred
        recomendacion = "El empleado tiene alta probabilidad de permanecer más de 90 días."
    else:
        resultado = "RENUNCIA < 90 DÍAS"
        probabilidad = 1 - pred
        recomendacion = "El empleado tiene alto riesgo de renunciar antes de 90 días."
    
    probabilidad = pred
    mensaje = "-RESULTADO DE PREDICCIÓN- \n\n"
    mensaje += f"Predicción: {resultado}\n"
    mensaje += f"Probabilidad: {probabilidad:.1%}\n"
    mensaje += f"Interpretación:\n{recomendacion}\n"
    

    
    messagebox.showinfo("Resultado de Predicción", mensaje)


def limpiar_campos():
    for col, widget in entradas.items():
        if isinstance(widget, ttk.Combobox):
            widget.set('')
        elif isinstance(widget, ttk.Entry):
            widget.delete(0, tk.END)
            # Restaurar valores sugeridos
            if col.upper() == "SALARIO DIARIO ANTERIOR":
                widget.insert(0, "250")
            elif col.upper() == "TIEMPO DE TRASLADO A SIIX MINUTOS":
                widget.insert(0, "30")


frame_botones = ttk.Frame(scrollable_frame)
frame_botones.grid(row=len(columnas_originales)+1, column=0, columnspan=2, pady=20)

btn_predecir = ttk.Button(frame_botones, text=" Predecir", command=predecir, width=15)
btn_predecir.pack(side="left", padx=10)

btn_limpiar = ttk.Button(frame_botones, text=" Limpiar", command=limpiar_campos, width=15)
btn_limpiar.pack(side="left", padx=10)

btn_salir = ttk.Button(frame_botones, text=" Salir", command=ventana.quit, width=15)
btn_salir.pack(side="left", padx=10)


def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)

print("\n=== APLICACIÓN INICIADA CORRECTAMENTE ===")
print(f"Modelo: FlexibleMLP con capas {hidden_layers}")
print(f"Features: {input_size}")
print(f"Variables categóricas: {len(opciones)}")
print(f"Variables numéricas: {len(columnas_numericas)}")
print("=========================================\n")


ventana.mainloop()