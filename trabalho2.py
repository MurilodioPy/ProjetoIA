import numpy as np
import customtkinter as ctk
from Perceptron import Perceptron
from Hebb import Hebb

# Configurações iniciais
col = 8
row = 8
entradas = col * row
my_ias = {0 : "Hebb", 1 : "Perceptron"}
p = Perceptron(entradas)

# Função para desenhar a grid com funcionalidade de ativar/desativar
def create_grid(parent_frame, col_value, row_value):
    grid = []
    for i in range(row_value):
        row_list = []
        for j in range(col_value):
            cell = ctk.CTkButton(parent_frame, width=30, height=30, text="",
            command=lambda btn=i * col_value + j: toggle_button(grid[btn // col_value][btn % col_value]))
            cell.grid(row=i, column=j, padx=1, pady=1)
            row_list.append(cell)
        grid.append(row_list)
    return grid

# Função para ativar/desativar um quadrado
def toggle_button(button):
    if button.cget("text") == "":
        button.configure(text="X")
    else:
        button.configure(text="")

# Função para mostrar a grid em formato de matriz no terminal
def print_grid_as_matrix(grid):
    matrix = []
    for row in grid:
        matrix_row = []
        for cell in row:
            if cell.cget("text") == "X":
                matrix_row.append(1)
            else:
                matrix_row.append(-1)
        matrix.append(matrix_row)

    for row in matrix:
        print(row)
    return matrix

# Função para achatar a matrix
def achatar_matrix(matrix):
    matrix = np.array(matrix)
    return matrix.flatten()

# Função de classificação
def print_result(prediction):
    if prediction == 0:
        result = "Grid 1"
    else:
        result = "Grid 2"

    label.configure(text=f"{result}")
    label.configure(fg_color='#4e8752', text_color='#ffffff', font=('Helvetica', 16, 'bold'))
    label.grid(row=3, column=0, columnspan=3, pady=20)

# Função a ser executada ao clicar no botão "Treinar"
def on_train_button_click():
    print("Grid 1:")
    grid1_matrix = print_grid_as_matrix(grid_letra1)

    print("\nGrid 2:")
    grid2_matrix = print_grid_as_matrix(grid_letra2)

    # Processando as grids para o treinamento
    A_processed = achatar_matrix(grid1_matrix)
    B_processed = achatar_matrix(grid2_matrix)

    letras_entrada = []
    letras_entrada.append(A_processed)
    letras_entrada.append(B_processed)

    # Saídas desejadas
    # 0 --> Grid 1 - A
    # 1 --> Grid 2 - B

    saidas = np.array([0, 1])

    p.train(letras_entrada, saidas)
    print("\nTreinamento Concluído!")

# Função a ser executada ao clicar no botão "Testar"
def on_test_button_click():
    print("\nGrid Teste:")
    grid_test_matrix = print_grid_as_matrix(grid_letratest)

    # Processando a grid para a previsão
    matrix_test = achatar_matrix(grid_test_matrix)

    # Realizando a previsão
    pred_test = p.predict(matrix_test)

    print_result(pred_test)

# Função para atualizar a grid com base na seleção de colunas e linhas
def update_grid():
    global col, row, grid_letra1, grid_letra2, grid_letratest, p
    col = int(column_selector.get())
    row = int(row_selector.get())

    # Atualiza o número de colunas e linhas
    col = int(column_selector.get())
    row = int(row_selector.get())

    # Atualiza o número de entradas
    entradas = col * row
    p = Perceptron(entradas)

    # Ajustar a geometria da janela principal
    root.geometry(f'{40*col*3 + 160}x{40*row + 200}')

    # Limpar grids antigos
    for widget in frame_letra1.winfo_children():
        widget.destroy()
    for widget in frame_letra2.winfo_children():
        widget.destroy()
    for widget in frame_letratest.winfo_children():
        widget.destroy()

    # Recriar grids
    grid_letra1 = create_grid(frame_letra1, col, row)
    grid_letra2 = create_grid(frame_letra2, col, row)
    grid_letratest = create_grid(frame_letratest, col, row)

# Configurações básicas da janela principal
root = ctk.CTk()
root.title("Perceptron - Frank Rosenblatt - 1958")
root.geometry(f'{40*col*3 + 160}x{40*row + 200}')

# Frame principal que centraliza os grids e botões
main_frame = ctk.CTkFrame(root)
main_frame.place(relx=0.5, rely=0.5, anchor="center")

# Frame para seleção de colunas e linhas
config_frame = ctk.CTkFrame(main_frame)
config_frame.grid(row=0, column=0, columnspan=3, pady=10)

column_selector = ctk.CTkOptionMenu(config_frame, values=list(my_ias.values()), command=lambda _: update_grid())
column_selector.set(my_ias.get(0))
column_selector.grid(row=0, column=0, padx=5)

column_selector = ctk.CTkOptionMenu(config_frame, values=[str(i) for i in range(3, 11)], command=lambda _: update_grid())
column_selector.set(str(col))
column_selector.grid(row=0, column=1, padx=5)

row_selector = ctk.CTkOptionMenu(config_frame, values=[str(i) for i in range(3, 11)], command=lambda _: update_grid())
row_selector.set(str(row))
row_selector.grid(row=0, column=2, padx=5)

# Frame para as letras e o teste
frame_letra1 = ctk.CTkFrame(main_frame, width=300, height=300, corner_radius=10)
frame_letra1.grid(row=1, column=0, padx=20, pady=20)
grid_letra1 = create_grid(frame_letra1, col, row)

frame_letra2 = ctk.CTkFrame(main_frame, width=300, height=300, corner_radius=10)
frame_letra2.grid(row=1, column=1, padx=20, pady=20)
grid_letra2 = create_grid(frame_letra2, col, row)

frame_letratest = ctk.CTkFrame(main_frame, width=300, height=300, corner_radius=10)
frame_letratest.grid(row=1, column=2, padx=20, pady=20)
grid_letratest = create_grid(frame_letratest, col, row)

label = ctk.CTkLabel(master=main_frame, text="")
label.grid(row=3, column=0, columnspan=2, sticky="nsew")

# Botão para treinar
btn_treinar = ctk.CTkButton(main_frame, text="Treino", width=120, command=on_train_button_click)
btn_treinar.grid(row=2, column=1, padx=20, pady=20)

# Botão para testar
btn_testar = ctk.CTkButton(main_frame, text="Teste", width=120, command=on_test_button_click)
btn_testar.grid(row=2, column=2, padx=20, pady=20)

# Iniciar a aplicação
root.mainloop()
