import random

# Configurações
TAMANHO_POPULACAO = 5
BITS_POR_CROMOSSOMO = 2
PROBABILIDADE_MUTACAO = 0.1
PROBABILIDADE_CRUZAMENTO = 0.8

# Função para gerar população inicial
def gerar_populacao(tamanho, bits):
    return [random.randint(0, (2 ** bits) - 1) for _ in range(tamanho)]

# Converter população decimal para binária
def converter_para_binario(pop_dec, bits):
    return [format(individuo, f'0{bits}b') for individuo in pop_dec]

# Função objetivo (aptidão)
def funcao_objetivo(x):
    return -(x**3) + 5 * x + x ** 2 + 5

# Calcular probabilidades da roleta
def calcular_probabilidades(pop_dec):
    aptidoes = [funcao_objetivo(x) for x in pop_dec]
    total = sum(aptidoes)
    return [aptidao / total for aptidao in aptidoes]

# Seleção dos melhores (via roleta)
def girar_roleta(probabilidades, pop_bin):
    selecionados = []
    for _ in range(len(pop_bin) // 2): # metade da população
        r = random.random()
        soma = 0
        for i, prob in enumerate(probabilidades):
            soma += prob
            if r <= soma:
                selecionados.append(pop_bin[i])
                break
    return selecionados

# Cruzamento
def cruzar(pais, ponto_corte):
    pai1, pai2 = pais
    filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
    filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]
    print(filho1, '  :Filho1')
    print(filho2, '  :Filho2')
    return filho1, filho2

# Aplicar mutação
def aplicar_mutacao(individuo_bin, prob_mutacao):
    mutacao =  ''.join(
        bit if random.random() > prob_mutacao else str(1 - int(bit))
        for bit in individuo_bin
    )
    print(mutacao, '  :Mutação')
    return mutacao

# Atualizar população
def atualizar_populacao(melhores, antigos, filhos):
    atualiza = melhores + antigos + filhos
    print(atualiza, '  :Atualiza')
    return atualiza
# Executar algoritmo genético
def algoritmo_genetico(iteracoes):
    pop_dec = gerar_populacao(TAMANHO_POPULACAO, BITS_POR_CROMOSSOMO)
    print(pop_dec, '  :População decimal')
    for _ in range(iteracoes):
        pop_bin = converter_para_binario(pop_dec, BITS_POR_CROMOSSOMO)
        print(pop_bin, '  :População binária')
        probabilidades = calcular_probabilidades(pop_dec)
        print(probabilidades, '  :Probabilidades')
        melhores = girar_roleta(probabilidades, pop_bin)
        print(melhores, '  :Melhores')
        # Cruzamento
        filhos = []
        for _ in range(int(len(pop_bin) * PROBABILIDADE_CRUZAMENTO // 2)):
            pais = random.sample(melhores, 2) # Seleciona 2 pais aleatórios
            print(pais, '  :Pais')
            ponto_corte = random.randint(1, BITS_POR_CROMOSSOMO - 1)
            print(ponto_corte, '  :Ponto de corte')
            filhos.extend(cruzar(pais, ponto_corte))
            print(pais, '  :Pais') 
        print(filhos, '  :Filhos')
        # Mutação
        filhos = [aplicar_mutacao(filho, PROBABILIDADE_MUTACAO) for filho in filhos]
        print(filhos, '  :Filhos com mutação')
        # Atualizar população
        pop_bin = atualizar_populacao(melhores, pop_bin, filhos)
        print(pop_bin, '  :População atualizada')
        pop_dec = [int(ind, 2) for ind in pop_bin]
        print(pop_dec, '  :População decimal atualizada')
    return pop_dec

# Execução
resultado = algoritmo_genetico(10)

print(f'\n Resultado final: \n')
print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
print(resultado)
