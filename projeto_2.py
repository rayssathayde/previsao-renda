import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Previsão de renda",
     page_icon="https://icones.pro/wp-content/uploads/2021/03/icone-de-l-argent-symbole-png-vert.png",
     layout="wide",
)

st.write('# Análise exploratória da previsão de renda')

st.write('## Método CRISP - DM')
st.write('---------------------')
st.write('## Entendimento do negócio')

st.write('''Uma instituição financeira fornece cartões de crédito para clientes, porém precisa medir de alguma forma o 
         risco de inadimplência desses clientes - a instituição precisa saber se é seguro ou não fornecer o cartão para 
         o cliente, ou seja, qual o risco desse cliente não pagar o que deve.  
         No momento em que o cliente solicita o cartão, a instituição pede alguns dados para que ocorra a avaliação de 
         crédito. A partir desses dados, a empresa quer construir um modelo preditivo para renda e identificar o risco 
         de inadimplência, para ser usado com futuros clientes. Dessa forma, o modelo fará a previsão se o futuro cliente 
         tem maior ou menor risco de ser inadimplente e se a empresa deve ou não fornecer o cartão.''')

st.write('---------------------')

st.write('## Entendimento dos dados')

st.write('''Os dados fornecidos pelos clientes estão dispostos em uma tabela com uma linha para cada cliente, e uma coluna 
         para cada variável armazenando as características desses clientes.  
         São 14 variáveis.''')



st.markdown('''
| Variável                | Descrição                                                               | Tipo         |
| ----------------------- |:-----------------------------------------------------------------------:| ------------:|
| data_ref                |  Data da coleta dos dados                                               | object       |
| id_cliente              |  Código identificador do cliente                                        | int64        |
| sexo                    |  F = feminino ou M = masculino                                          | object       |
| posse_de_veiculo        |  Indica a posse de veículo (True = possui; False = não possui)          | bool         |   
| posse_de_imovel         |  Indica a posse de imóvel  (True = possui; False = não possui)          | bool         |
| qtd_filhos              |  Quantidade de filhos do cliente                                        | int64        |
| tipo_renda              |  Tipo de renda (empresário, assalariado, servidor público, etc)         | object       |
| educacao                |  Nível de educação (secundário, superior completo ou incompleto, etc)   | object       |
| estado_civil            |  Estado civil (casado, solteiro, separado, etc)                         | object       |
| tipo_residencia         |  Tipo de residência (casa, com os pais, aluguel, etc)                   | object       |
| idade                   |  Idade do cliente (em anos)                                             | int64        |
| tempo_emprego           |  Tempo (em anos) no emprego atual                                       | float64      |
| qt_pessoas_residencia   |  Quantidade de pessoas que moram na residência                          | float64      |
| renda                   |  Renda (em reais) do cliente                                            | float64      |''')


st.write('---------------------')

st.write('## Pacotes utilizados')

st.write('''Numpy   
         Pandas  
         Seaborn   
         Matplotlib   
         Statsmodels  
         Sklearn''')

st.write('---------------------')

renda = pd.read_csv('./input/previsao_de_renda.csv')

st.write('### Análise da renda e o impacto das variáveis explicativas')


#plots
fig, ax = plt.subplots(8,1,figsize=(10,80))
renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0])
st.write('### Gráficos ao longo do tempo')
sns.lineplot(x='data_ref',y='renda', hue='posse_de_imovel',data=renda, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='posse_de_veiculo',data=renda, ax=ax[2])
ax[2].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='qtd_filhos',data=renda, ax=ax[3])
ax[3].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_renda',data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='educacao',data=renda, ax=ax[5])
ax[5].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='estado_civil',data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_residencia',data=renda, ax=ax[7])
ax[7].tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(plt)

st.write('---------------------')

st.write('### Gráficos bivariada')
fig, ax = plt.subplots(7,1,figsize=(10,80))
sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0])
sns.barplot(x='posse_de_veiculo',y='renda',data=renda, ax=ax[1])
sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[2])
sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[3])
sns.barplot(x='educacao',y='renda',data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=12)
sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[5])
sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(plt)

st.write('---------------------')
st.write('''Os gráficos acima mostram que as variáveis ``posse_de_veiculo``, ``sexo``, ``tipo_renda`` (servidor público),
            ``estado_civil`` (casado) e ``tipo_residencia`` (estúdio) parecem apresentar um valor preditivo relevate em 
            relação a ``renda``.''')

st.write('---------------------')

st.write('## Modelagem')


st.write('''Para construir um modelo de previsão de renda, a técnica escolhida foi uma análise de regressão múltipla, 
         utilizando como variável resposta ``renda`` e as variáveis explicativas ``tempo_emprego``, ``posse_de_veiculo``, 
         ``sexo``, ``tipo_renda`` (servidor público), ``estado_civil`` (casado) e ``tipo_residencia`` (estúdio).''') 

st.write('''O modelo construído apresentou coeficiente de determinação de aproximadamente 36% para a base de treino e 
         23% na base de teste.''')


#preparando dados

renda = renda.drop_duplicates() #retirar dados duplicados
renda = renda.dropna() #retirar dados faltantes
renda = renda.drop(['data_ref', 'id_cliente'], axis=1) #retirar variáveis que não serão utilizadas
renda_final = pd.get_dummies(renda) #modificar o tipo das variáveis qualitativas

#dividir base em treino e teste

renda_train, renda_test = train_test_split(renda_final, test_size=0.3, random_state=100)
renda_train.rename(columns={'tipo_renda_Servidor público': 'tipo_renda_Servidor_publico'}, 
                 inplace=True)

#rodar modelo

reg_log = smf.ols('''np.log(renda) ~ posse_de_veiculo 
                                   + sexo_M
                                   + tempo_emprego
                                   + tipo_renda_Servidor_publico
                                   + estado_civil_Casado
                                   + tipo_residencia_Estúdio''', renda_train).fit()

st.write('---------------------')

st.write('## Aplicando o modelo')

st.write('Utilize o quadro abaixo para completar os dados de um cliente e aplicar o modelo para prever sua renda.')
st.write('''As colunas devem ser preenchidas da seguinte forma:  
         * sexo_M: 0 para feminino, 1 para masculino  
         * posse_de_veiculo: 0 para não possui e 1 para possui  
         * posse_de_imovel: 0 para não possui e 1 para possui  
         * qtd_filhos: número de filhos do cliente  
         * tipo_renda_Servidor_publico: 1 para servidor público e 0 para demais profissões  
         * estado_civil_Casado: 1 para casado e 0 para demais estados civis  
         * tipo_residencia_Estúdio: 1 para estúdio e 0 para demais tipos de residência  
         * idade: idade do cliente em anos  
         * tempo_emprego: tempo no emprego atual em anos
         * qt_pessoas_residencia: número de pessoas que moram na residência ''')



entrada = pd.DataFrame([{'sexo_M': 1, 
                         'posse_de_veiculo': 1, 
                         'posse_de_imovel': 1, 
                         'qtd_filhos': 1, 
                         'tipo_renda_Servidor_publico': 1,  
                         'estado_civil_Casado': 1, 
                         'tipo_residencia_Estúdio': 1, 
                         'idade': 34, 
                         'tempo_emprego': 7.183562, 
                         'qt_pessoas_residencia': 5}])

edited_entrada = st.data_editor(entrada)

renda_pred = np.exp(reg_log.predict(edited_entrada)) # como o modelo considera log de renda, usar exp para ter valor real
st.write(f"Renda estimada: R${str(np.round(renda_pred[0], 2)).replace('.', ',')}")