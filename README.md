# História Inicial das Inteligências Artificiais: Hebb e Perceptron

## Introdução
A história da inteligência artificial (IA) é repleta de marcos significativos que moldaram o desenvolvimento das tecnologias que conhecemos hoje. Dois desses marcos são a teoria de aprendizagem de Hebb e o Perceptron, ambos fundamentais para o avanço dos modelos de redes neurais.

## Teoria de Aprendizagem de Hebb

### O que é?
A teoria de aprendizagem de Hebb, proposta por Donald Hebb em 1949, sugere que as conexões entre neurônios (sinapses) se fortalecem à medida que são ativadas simultaneamente. Essa ideia pode ser resumida pela frase "neurônios que disparam juntos, se conectam".

### Importância
- **Fundamento da neurociência**: A teoria de Hebb foi crucial para a compreensão de como o cérebro pode se adaptar e aprender com base em experiências.
- **Base para as redes neurais artificiais**: A ideia de que as conexões podem ser fortalecidas por meio da experiência direta influenciou o desenvolvimento de modelos de aprendizado automático.

### Funcionamento
O princípio de Hebb pode ser representado matematicamente como uma atualização das sinapses, onde o peso da conexão entre dois neurônios aumenta proporcionalmente à ativação desses neurônios:


$`Delta w_{ij} = \eta \cdot x_i \cdot y_j`$


onde:
- $`( \Delta w_{ij} )`$ é a mudança no peso da conexão entre os neurônios $`( i )`$ e $`( j )`$,
- $`( \eta )`$ é a taxa de aprendizagem,
- $`( x_i )`$ é a entrada no neurônio $`( i )`$,
- $`( y_j )`$ é a saída do neurônio $`( j )`$.

## O Perceptron

### Origem
O Perceptron foi desenvolvido por Frank Rosenblatt em 1958, sendo um dos primeiros modelos de redes neurais artificiais. Ele foi projetado para simular o processo de aprendizagem dos neurônios humanos.

### Funcionamento
O Perceptron é um algoritmo de classificação binária que toma uma decisão com base em uma combinação linear dos pesos das entradas. Em termos simples, ele pode ser visto como um modelo de aprendizado supervisionado que ajusta os pesos das entradas para minimizar o erro na classificação de dados.

### Estrutura
- **Entradas**: Um conjunto de valores numéricos (vetor de características).
- **Pesos**: Um conjunto de valores que multiplicam as entradas.
- **Função de Ativação**: Uma função que decide se o neurônio deve "disparar" ou não, geralmente utilizando uma função degrau.
- **Saída**: Uma decisão binária (0 ou 1).

A fórmula básica do Perceptron é:

$`y = \text{sign}(\sum_{i=1}^{n} w_i x_i + b)`$

onde:
- $`( y )`$ é a saída do Perceptron,
- $`( w_i )`$ são os pesos,
- $`( x_i )`$ são as entradas,
- $`( b )`$ é o bias,
- $`( \text{sign} )`$ é a função de ativação que retorna 1 se a soma for positiva e 0 caso contrário.

### Impacto
- **Primeira aplicação prática de redes neurais**: O Perceptron foi utilizado em uma variedade de problemas de classificação, mostrando o potencial das redes neurais artificiais.
- **Limitações**: Embora poderoso, o Perceptron clássico só é capaz de resolver problemas linearmente separáveis. Isso levou ao desenvolvimento de modelos mais complexos, como as redes neurais multicamadas.

## Conclusão
A teoria de aprendizagem de Hebb e o Perceptron representam os primeiros passos no desenvolvimento das redes neurais artificiais. Ambos são fundamentais para o avanço das tecnologias de IA e continuam a influenciar pesquisas e aplicações modernas.

## Referências
- Hebb, D. O. (1949). *The Organization of Behavior: A Neuropsychological Theory*. Wiley.
- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*. Psychological Review.

[Playground tensor flow](https://playground.tensorflow.org/) - Brinque um pouco com redes neurais!!
