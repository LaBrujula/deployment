#importar las librerias
import streamlit as st
import pandas as pd
import pickle
import requests
import json
from PIL import Image
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost.sklearn import XGBRegressor
from sklearn.pipeline import Pipeline
import re
import emoji

def clean_tweets(tweet_text,mentions=True, newlines=True, extra_whitespace=True, hyperlinks=True, punctuation=True, special=True, emojis=True):
            clean_criteria = []
            clean_text = tweet_text
            if mentions:
             clean_criteria.append((r'@\w+', ''))
            if newlines:
             clean_criteria.append((r'\n', ' '))
            if extra_whitespace:
             clean_criteria.append((r'\s{2,}', ' '))
            if hyperlinks:
             clean_criteria.append((r'https?://\S+', ''))
            if punctuation:
             clean_criteria.append((r'[\!\,\.\:\;]', ''))
            if special:
             clean_criteria.append((r'[\"\#\$\%\&\'\(\)\*\+\-\/\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]', ''))
            for criteria, repl in clean_criteria:
             clean_text = re.sub(criteria, repl, clean_text)
            if emojis:
             clean_text = emoji.demojize(clean_text, delimiters=(" ", " "), language='es')
            return clean_text.strip()

class TweetCleaner(BaseEstimator, TransformerMixin):
                def __init__(self):
                    pass
    
                def fit(self, X, y=None):
                    return self
    
                def transform(self, X, y=None):
                    X_cleaned = X.apply(clean_tweets)
                    return X_cleaned

def main():
    st.set_page_config(
    page_title="Trabajo Integrador Final",
    page_icon="????",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None)

    
    # Header
    st.header('Deployment del Trabajo Integrador Final')

    # Text

    tab1, tab2, tab3 , tab4, tab5, tab6, tab7 = st.tabs(["Presentaci??n", "Dolar Blue", "Tweets", "Modelos", "Evaluaci??n", "Predicci??n", "Conclusiones"])

    with tab1:
       st.subheader('Presentaci??n')
       st.markdown('__Trabajo Integrador Final__ del curso de Data Science de _Digital House_\n\nAlumnxs: Guadalupe __Atienza__ | Luc??a __Sauer__ | Santiago __Bezchinsky__ | Santiago __Federico__')
       image_1 = Image.open('p_1.jpeg')
       st.image(image_1, caption=None)
       image_2 = Image.open('p_2.jpeg')
       st.image(image_2, caption=None)
       image_3 = Image.open('p_3.jpeg')
       st.image(image_3, caption=None)

    with tab2:
       image_1 = Image.open('h_2.png')
       st.image(image_1, caption=None)
       st.markdown('La obtenci??n de la cotizaci??n diaria del D??lar Blue :flag-us: se ha logrado gracias a la utilizaci??n de una API conectada a Bluelytics, lo que permiti?? generar un DataFrame b??sico con valores de compra y venta del billete de moneda estadounidense desde el d??a 01/01/19 hasta el 30/12/22.')
       st.markdown('Posteriormente se realizaron operaciones de DataWrangling y EDA. A continuaci??n se presentan los elementos mas representativos de dicho proceso.')
       st.markdown('_Para m??s informaci??n se puede consultar la p??gina de Bluelytics: https://bluelytics.com.ar/_')
       
       st.subheader('Evoluci??n del D??lar Blue')
       with open('serie_dolar_historico.pkl', 'rb') as f_fig:
        serie_dolar_historico = pickle.load(f_fig)
        st.plotly_chart(serie_dolar_historico, use_container_width=True)
       st.subheader('Tendencias')
       st.markdown('Como sabemos, una serie de tiempo est?? compuesta por los siguientes componentes: la tendencia, la estacionalidad, el ciclo y el error. En el gr??fico debajo podemos apreciar la descomposici??n de la serie del d??lar venta')
       with open('dolar_tendencia.pkl', 'rb') as f_fig2:
        dolar_tendencia = pickle.load(f_fig2)
        st.plotly_chart(dolar_tendencia, use_container_width=True)
       st.subheader('Estacionariedad del D??lar Blue')
       st.markdown('Por otro lado, para poder estimar y hacer inferencia a partir de un proceso generador de datos necesitamos que la serie de tiempo sea ESTACIONARIA. Claramente, al analizar la serie vemos que la misma posee un componente tendencial, raz??n suficiente para que no se cumpla la estacionariedad.')
       with open('serie_dolar_estacionariedad.pkl', 'rb') as f_fig3:
        serie_dolar_estacionariedad = pickle.load(f_fig3)
        st.plotly_chart(serie_dolar_estacionariedad, use_container_width=True)
       st.markdown('Al diferenciar la serie, podemos notar que el componente tendencial fuertemente marcado desaparece, y la serie de la variaci??n del d??lar pasa a ser estacionaria. A partir de ahora esta ser?? nuestra variable target en los modelos que m??s adelante implementaremos.')
       st.info('_cada gr??fico tiene un encabezado donde se despliegan varias opciones de visualizaci??n_', icon="???????") 

    with tab3:
        image_1 = Image.open('h_3.jpeg')
        st.image(image_1, caption=None)
        st.markdown('Las cuentas institucionales, personales y de medios de comunicaci??n _influyentes_ seleccionadas se agrupa en 5 tipos, caracterizados en la siguiente manera:')
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Economistas", "81", None)
        col2.metric("Medios", "51", None)
        col3.metric("Gobierno", "23", None)
        col4.metric("Empresarios", "6", None)
        col5.metric("Escritores", "2", None)

        st.markdown('Seleccionadas las cuentas y generado el scrappeo, se pre-procesan los tweets, ya que muchas veces traen una cantidad importante de menciones, hiperv??nculos y signos de puntuaci??n, que solo contribuyen a agregar ruido (y tiempo de procesamiento) a los diversos modelos de NLP. Este proceso fue realizado a partir de diversas t??cnicas de _text minning_ y otros recursos __muy copados__ que ya conocen.\n\n')
        st.subheader('En total, el dataset se compone de m??s de 1.5 millones de tweets :scream:\n\n')
        
        st.header('Nube de palabras')
        st.markdown('Como se desarroll?? en en la secci??n de __"Presentaci??n"__, el scrappeo de las cuentas de los _influyentes_ se almacen para su posterior analisis.\n\nA continuaci??n se muestra la __Nube de palabras__ donde se muestran los tweets que aparecen con mayor frecuencia.')
        with open('wordcloud2.pkl', 'rb') as f_fig3:
         wordcloud = pickle.load(f_fig3)
         st.image(wordcloud, caption=None)
        
        st.header('Limpieza de Tweets')
        st.markdown('Para un mejor procesamiento, se cre?? una funci??n que permite realizar la limpieza de __tweets__. Para testearla, introducir un tweet a continuaci??n.')
        st.caption('Se deja este __TERRIBLE__ tweet a modo de ejemplo: ????AHORA | ????????Tras la reuni??n de Chiqui #Tapia y Lionel #Scaloni en la previa de los premios #TheBest en Par??s, hubo acuerdo total y firma hasta el mundial de 2026. La #Scaloneta contin??a su viaje...????????')
        with st.form("clean_tweets"):
            dirty_tweet = st.text_input('introducir un __tweet__ _complejo_, con emojis :hot_pepper::tada:	:poop: y cositas (#@//*&)...:')
            submitted = st.form_submit_button("quiero ver la magia.")
        if submitted:
            st.write()
        tweet_text = dirty_tweet       
        resultado = clean_tweets(tweet_text)
        st.write('Asi se ve __despu??s__ de la limpieza: \n\n',resultado,'\n\n_...claramente se nota la magia..._')
        st.info('_vale aclarar que se pueden modificar los par??metros de limpieza. vale tanto para la eliminaci??n de hiperv??nculos, hashtags o emojis..._', icon="???????")   

    with tab4:
        image_1 = Image.open('h_4.jpeg')
        st.image(image_1, caption=None)
        st.subheader('i) CountVectorizer + Logistic Regression')
        st.markdown('Los tweets se limpian y concatenan en un ??nico string por d??a, y luego se preprocesan con CountVectorizer para generar la matriz de features. Finalmente se modela utilizando Logistic Regression, para predecir la categor??a: 1, 0 o -1, seg??n si ese d??a la variaci??n del precio del Blue fue positiva, nula o negativa.')
        st.subheader('ii) CountVectorizer + Time Series Features + XGBoost')
        st.markdown('Consiste en un modelos de regresi??n que tiene como variable target la variaci??n del d??lar absoluta y como features posee la variaci??n del d??lar rezagada un per??odo, los tweets emitidos del d??a anterior, e introduce como dummies el mes, el d??a de la semana y el quarter del a??o. El preprocesamiento de los tweets es igual al del primer modelo.')
        st.subheader('iii) HuggingFace Transformer (AutoTrain + Disponibilizaci??n con API)')
        st.markdown('Al no poder realizar f??cilmente el deploy de nuestro modelo con embeddings, realizamos como prueba extra un modelo de Transformers entrenado con la plataforma AutoTrain de HuggingFace, donde especificando el problema (TextClassification) y el dataset, la plataforma selecciona y entrena el mejor modelo para resolverlo. Esto se encuentra disponibilizado mediante una API, que nuestro Streamlit consume para realizar una predicci??n equivalente al modelo 1, pero con otro procesamiento de los datos.')
        st.markdown('_Otro modelo realizado pero que no se encuentra en esta p??gina..._')
        st.subheader('iv) Embeddings + logistic regression')
        st.markdown('En vez de concatenar los strings de un mismo d??a, procesamos los tweets con un modelo de transformers en espa??ol, ajustado para trabajar con tweets. El modelo genera embeddings a nivel de cada palabra, y mediante el mecanismo de self-attention, combina esos embeddings al nivel de cada Tweet. Para obtener un embedding de cada d??a, se realiz?? un mean pooling. Luego estas 256 dimensiones generadas por el embedding fueron tomadas como features para entrenar diversos modelos de clasificaci??n, de los cuales nos quedamos con Logistic Regression')
        image_2 = Image.open('embeddings.jpeg')
        st.image(image_2, caption=None)

    with tab5:
       image_1 = Image.open('h_5.jpeg')
       st.image(image_1, caption=None)
       st.subheader('_CountVectorizer + Logistic Regression_')
       st.markdown('El _accuracy_ haciendo predicciones con la clase mayoritaria (0): 0.4831932773109244')
       image_1 = Image.open('im_1.jpg')
       st.image(image_1, caption=None)
       image_2 = Image.open('im_2.jpg')
       st.image(image_2, caption=None)
       col1, col2 = st.columns(2)
       with col1:
        st.caption("Resultados con 'CountVectorizer'")
        st.image("im_3.jpg")
       with col2:
        st.caption("Resultados con 'TF-IdF'")
        st.image("im_4.jpg")
       image_5 = Image.open('im_5.jpg')
       st.caption("Scatter Plot")
       st.image(image_5, caption=None)
       image_6 = Image.open('im_6.jpg')
       st.caption("Dimensiones de los t??rminos m??s recurrentes")
       st.image(image_6, caption=None)
       image_7 = Image.open('im_7.jpg')
       st.caption("Resultados del _accuracy_ de los modelos logit, forest y xgboost")
       st.image(image_7, caption=None)
       image_8 = Image.open('im_8.jpg')
       st.caption("Scatter Plot")
       st.image(image_8, caption=None)
       st.subheader('_CountVectorizer + Time Series Features + XGBoost_')
       image_9 = Image.open('im_9.jpg')
       st.caption("Resultados del modelo")
       st.image(image_9, caption=None)
       with open('ts_graph.pkl', 'rb') as f_4:
        ts_graph = pickle.load(f_4)
        st.plotly_chart(ts_graph, use_container_width=True)

    with tab6:
        image_1 = Image.open('h_6.jpeg')
        st.image(image_1, caption=None)
        st.markdown('La siguiente secci??n busca predecir de qu?? manera explotar?? el __d??lar blue__ en :flag-ar:. Cada uno de los modelos predice algo distinto por lo que resulta __necesario__ que elija en primera instancia uno de los modelos y, _posteriormente_, ingrese los datos requeridos.')
        st.markdown('_Recordemos los modelos..._\n\nel __modelo 1__ est?? bastante piola,\n\n utiliza un _CountVectorizer + Logistic Regression_\n\n el __modelo 2__ es __IMPECABLE__,\n\n _predice utilizando Time Series [CountVectorizer + Time Series Features + XGBoost]_ \n\n finalmente el __modelo 3__ es una :bomb::heavy_exclamation_mark:, \n\n que se trajo desde una __API__ llamada _HuggingFace Transformer [AutoTrain + Disponibilizaci??n con API]_')
        
        with st.form("my_form"):
            option = st.selectbox('Se requiere elegir uno de los __increibles__ modelos predictivos',('modelo 1', 'modelo 2', 'modelo 3'))
            submitted = st.form_submit_button("Como dir??a Google... __Voy a tener suerte!__ :four_leaf_clover:")
        if submitted:
            st.write("__seleccionado:__", option)

        modelo = option  
        if modelo == 'modelo 1':   
            with st.form("my_form_1"):
             tweet = st.text_input('Escriba un Tweet:')
             submitted_2 = st.form_submit_button("Dale, __quiero saber!__ Realmente no aguanto m??s... :scream:")
            if submitted_2:
             st.write("__tweet:__", tweet)
            pred = pd.Series([tweet])
            
            with open('pipe-cv.pkl', 'rb') as m_1:
                pipe_cv = pickle.load(m_1)

            pred_variacion = pipe_cv.predict(pred)
            st.write('El resultado de la prediccion es:',pred_variacion[0])
            st.caption('[*] el valor [-1] refiere a una variaci??n negativa, el [1] a una positiva y el [0], _obviamente_, a que no que no haya cambios')
                                
        elif modelo == 'modelo 2': 
            with st.form("my_form_3"):
             text_lag = st.text_input('Escriba un Tweet:')
             month = st.selectbox('Seleccione un mes:',(1,2,3,4,5,6,7,8,9,10,11,12))
             quarter = st.selectbox('Selecciones un cuatrimestre:',(1,2,3,4))
             variation_lag = st.number_input('ingresar la diferencia con respecto al dia anterior: ')
             weekday = st.selectbox('Elija el numero de la semana:',(1,2,3,4))
             features = {'text_lag':text_lag, 'month':month, 'quarter':quarter, 'variation_lag':variation_lag, 'weekday':weekday}
             input = pd.DataFrame(features, index=[0])
             submitted = st.form_submit_button("seleccionar")
             
             if submitted:
               st.write("_me la juego con:_", input)
                          
            input = pd.DataFrame(features, index=[0])
            with open('pipemodel3.pkl', 'rb') as m3:
                model3 = pickle.load(m3)
            pred_serie = model3.predict(input)
            st.write('Seg??n la predicci??n, se espera que el D??lar Blue tenga una variaci??n de ',pred_serie[0],' :flag-ar:')
            

        elif modelo == 'modelo 3':
            st.markdown('A modo de yapa, incorporamos un modelo desde _Hugging Face_ :hugging_face:, __pa joder nomas!__')
            tweet = 'a la espera de que pase algo...'
            with st.form("my_form_8"):
             tweet = st.text_input('Escriba un Tweet:')
             submitted = st.form_submit_button("Dale, __quiero saber!__ Realmente no aguanto m??s... :scream:")
            if submitted:
             st.write("__tweet:__", tweet)
            API_URL = "https://api-inference.huggingface.co/models/tizan25/autotrain-twitter-classifier-v2-37780100209"
            headers = {"Authorization": "Bearer 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"}
            def query(payload):
	            response = requests.post(API_URL, headers=headers, json=payload)
	            return response.json()
            output = query({"inputs":tweet})
            data = output
            label = []
            score = []
            for features in data[0]:
                label.append(features['label'])
                score.append(features['score'])
            datos = {'Variacion':label,'Probabilidad':score}
            prediccion = pd.DataFrame(data = datos)
            prediccion['Variacion'] = prediccion['Variacion'].replace({'0': 'Sin cambios', '1': 'Positiva', '-1': 'Negativa'})
            st.dataframe(prediccion)    
                    
        st.info('Este mensaje es puramente informativo, pero... __creo que nos vamos al tacho__', icon="???????")

    

    with tab7:
       image_1 = Image.open('h_7.jpeg')
       st.image(image_1, caption=None)
       st.markdown(':heavy_exclamation_mark: Las clases mayoritarias cuando se analiza la serie historica del dolar blue corresponden a los d??as con variaci??n 0, d??as con NaN (correspondientes a aquellos donde no hay registro de variaci??n porque el mercado estaba cerrado, es decir fines de semana y feriados), o d??as con baja variaci??n (1 o 2 pesos de variaci??n positiva o negativa. Con estas 6 categor??as ya abarcamos el 88% de los casos del per??odo observado. Esto representar?? un desaf??o a la hora de entrenar modelos con m??tricas de predicci??n aceptables.')
       st.markdown(':heavy_exclamation_mark: Las diferencias a??o a a??o pueden hacer que resulte complejo generar un modelo ??nico.')
       st.markdown(':heavy_exclamation_mark: La evaluaci??n de los primeros modelos present?? resultados magros. Al concatenar todo se pierde algo de la informaci??n contextual "intra-tweets" o "intra-user", mientras que por otro lado, haciendo BOW o TFIDF se tienen datos demasiado escasos, con baja informaci??n para entrenar modelos. En el caso de TFIDF, tambi??n se suma el hecho de que otorga mayor importancia a palabras ??nicas a un documento, y resta importancia a palabras comunes al corpus.')
       st.markdown(':heavy_exclamation_mark: Parece haber un importante peso de los t??rminos asociados a la pandemia de COVID-19, algo esperable en un dataset que atraviesa los a??os 2020 y 2021. Luego aparecen programas de actualidad, y finalmente temas de agenda como la guerra de Ucrania, el caso de Fernando Baez Sosa y algunos personajes p??blicos.')
       st.header('Posibles mejoras al __TIF__')
       st.markdown(':question: Hacer comprobaciones para ver si los pesos de los t??rminos principales de cada a??o no afectan la capacidad predictiva del modelo.')

if __name__ == '__main__':
    main()
