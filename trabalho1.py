import customtkinter as ctk
import numpy as np

pontos = 24
entrada = np.full((2, pontos), -1.0)  # Matriz 2xpontos preenchida com -1.0
y = np.array([1, -1])  # Saída esperada: 1 para A e -1 para B

# Inicialização dos pesos e bias
w = np.zeros(pontos)
b = 0.0

print("\nPrograma Regra de Hebb para reconhecimento de A e B")

# Inserção dos dados da letra A (entrada)
entrada[0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 19, 20, 23]] = 1

# Inserção dos dados da letra B (entrada)
entrada[1, [0, 1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22, 23]] = 1

# Aplicação da regra de Hebb
for cont1 in range(2):
    deltaW = entrada[cont1] * y[cont1]
    deltaB = y[cont1]
    w += deltaW
    b += deltaB
    # Valores de w
    # A/B 0
    # A 2
    # B -2
    # N A/B 0

# Cria a janela principal
app = ctk.CTk()
rows, cols = 6, 4
column_width = 40
row_height = 40
checkboxes = []

for col in range(cols):
    app.grid_columnconfigure(col, weight=1, minsize=column_width)

for row in range(rows):
    app.grid_rowconfigure(row, weight=1, minsize=row_height)

# pegando valores da matrix de checkbox
def matrix_checkbox_values():
    matrix = np.zeros((rows, cols))
    for i, row_in_matrix in enumerate(checkboxes):
        for j, checkbox_in_line in enumerate(row_in_matrix):
            matrix[i, j] = 1 if checkbox_in_line.get() else -1
    return matrix


def testando_entrada(weight, bias):
    matrix_checkbox = matrix_checkbox_values()
    # matrix checkbox achatada
    matrix_1x_pontos = matrix_checkbox.flatten()

    # y = mx + b
    # np.dot -> produto escalar entre dois vetores
    delta_teste = np.dot(weight, matrix_1x_pontos) + bias
    teste = 1 if delta_teste >= 0 else -1
    if teste == 1:
        print_resultado("A", "#0000FF", "#ffffff")
    else:
        print_resultado("B", "#FF0000", "#ffffff")

    print("\nSaída esperada: 1 (A), -1 (B)")
    print(f"\nSaída encontrada: {teste}\n")

def print_resultado(resultado, cor_fundo, cor_texto):
    label.configure(text=f"Resultado: {resultado}", fg_color=cor_fundo, text_color=cor_texto)
    label.grid()

# montando matrix de checkbox
for row in range(rows):
    checkbox_row = []
    for col in range(cols):
        checkbox = ctk.CTkCheckBox(master=app, text='', width=column_width, height=row_height, corner_radius=0)
        checkbox.grid(row=row, column=col, sticky="nsew")
        checkbox_row.append(checkbox)
    checkboxes.append(checkbox_row)


button = ctk.CTkButton(master=app, text="Testar Entrada", command=lambda: testando_entrada(w, b))
button.grid(row=rows, columnspan=cols, pady=10, sticky="nsew")

label = ctk.CTkLabel(master=app, text="Resultado: ")
label.grid(row=rows+1, column=0, columnspan=cols, sticky="nsew")

app.geometry("200x350")  # Ajusta o tamanho da janela conforme necessário
app.attributes("-topmost", True)  # sempre visível
app.mainloop()