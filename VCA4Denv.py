import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from PIL import Image

header = st.container()
VC = st.container() #Cassava Value Chain (agrégée)
PR = st.container() #Production de racines de manioc
TR = st.container() #Cassava Value Chain (désagrégée)
IN = st.container() #inventary of data


with header:

    st.header('Analyse environnementale ACV de la chaîne de valeur du manioc en RDC',divider='blue')
    st.markdown("Cet outil permet de visualiser les résultats de l'analyse du cycle de vie environnemental de la chaîne de valeur du manioc en République démocratique du Congo à l'aide de :blue[SIMAPRO]. Les résultats de l'analyse de la chaîne de valeur de deux manières : d'abord, sous forme agrégée, en montrant les impacts globaux ; ensuite, sous forme désagrégée, en analysant chaque étape du processus séparément ")
    
    st.divider()
    st.write(' 👈Sur la gauche, vous trouverez un menu proposant différentes options de visualisation des données.')
    st.divider()
    
    DD = st.sidebar.radio(
        "Sélectionnez le type de données que vous souhaitez afficher",
        ["**Inventaire de Cycle de Vie (ICV)**","**Cassava Value Chain (agrégée)**", "**Cassava Value Chain (désagrégée)**","**Production de racines de manioc**" ],
        captions = ["Intrants Agricoles et Unités de transformation du manioc en fufu","Montrant les impacts globaux", "Analysant chaque étape du processus séparément", "Production de manioc dans 8 sites différents en RDC"])

with IN:

    if DD == "**Inventaire de Cycle de Vie (ICV)**":


        # Datos de la primera tabla
        data1 = {
            'Parcelle': ['Parcelle 1', 'Parcelle 2', 'Parcelle 3', 'Parcelle 4', 'Parcelle 5', 'Parcelle 6', 'Parcelle 7', 'Parcelle 8'],
            'Lieu': ['Menkao', 'Madimba (LAYUCA)', 'Mbanza-Ngungu -CKID', 'Seeke-BanzaAGRIKCOM', 'Kwango / Plateau Bateke (ECOSAC)', 'Kwilu', 'Sud-Kivu', 'Tshopo'],
            'Description': ['Parcelle collective', 'Parcelle collective', 'Parcelle individuelle', 'Parcelle collective', 'Parcelle collective', 'Parcelle collective', 'Parcelle collective', 'Parcelle collective'],
            'Surface déclarée (ha)': [10, 90, 60, 35, 300, 1.6, 0.4, 7],
            'Production (t racines)': [35, 1350, 600, 700, 4500, 7.2, 1.5, 100],
            'Rendement (t racines.ha-1)': [3.5, 15, 10, 20, 15, 4.5, 3.7, 14.2],
            'Cycle de culture (mois)': [12, 12, 9, 12, 12, '-', '-', '-'],
            'Densité de plantation (m x m)': ['1 x 1'] * 8,
            'Densité de plantation (plantes/ha)': [10000] * 8,
            'Résidus de récolte laissés au champ (t résidus.ha-1)': [0.77, 3.3, 2.2, 4.4, 3.3, 0.99, 0.84, 3.12],
            'Semences': ['Tiges récolte précédente'] * 8,
            'Organic fertilizer': [0] * 8,
            'Mineral fertilizer (Kg fertilizer/ha)': [0, 0, 0, 0, 120, 0, 0, 0],
            'Glyphosate (herbicide) (L herbicide/ha)': [0, 0, 0, 0, 20, 0, 0, 0],
            'Pesticides': [0] * 8,
            'Tracteur: défricher et labourer (L diesel/ha)': [35, 25, 23, 20, 60, 29.03, 23, 26],
            'Phosphore dû à l’érosion (eau) (kg P/ha/an)': [11.31] * 8,
            'N2O des résidus de récolte (air) (kg N2O/ha/an)': [0.18, 0.78, 0.52, 1.44, 0.78, 0.23, 0.19, 0.74],
            'NO2 des résidus de récolte (air) (kg NO2/ha/an)': [0.04, 0.16, 0.11, 0.22, 0.16, 0.05, 0.04, 0.15],
            'N2 des résidus de récolte (air) (kg N2/ha/an)': [1.04, 4.46, 2.97, 5.94, 4.46, 1.34, 1.10, 4.22],
            'Centre de transformation': ['Menkao', 'Madimba', 'Mbanza-Ngungu', 'Seeke-Banza', 'Kwango', 'Kwilu', 'Sud-Kivu', 'Tshopo'],
            'Distance (km)': [1, 25, 1, 12, 10, '-', '-', '-'],
            'Type de transport / Capacité (t)': ['Moto 200kg', 'Camion 10ton', 'Tracteur 4 ton', 'Moto 200kg', 'Camion 10ton', '-', '-', '-']
        }

        # Creación del DataFrame para la primera tabla
        df1 = pd.DataFrame(data1)
        df1.set_index('Parcelle', inplace=True)

        # Datos de la segunda tabla
        data2 = {
            'Unité de transformation': ['Unité transfo 1', 'Unité transfo 2', 'Unité transfo 3', 'Unité transfo 4'],
            'Lieu': ['Madimba', 'Kinshasa', 'Plateau Bateke', 'Matadi'],
            'Capacité de production (t racines/jour)': [3.7, 5.7, 3.8, 1],
            'Capacité de production (t fufu/jour)': [1.0, 1.5, 1.0, 0.26],
            'Eau (fermentation racines) (m3/t fufu)': [3.7, 5, 3.8, 1.0],
            'Peaux (t/t fufu)': [0.38, 0.85, 0.39, 0.15],
            'Eaux usées (m3/t fufu)': [3.7, 57, 3.8, 1.0],
            'Matière organique dans les eaux usées (kg/t fufu)': [74, 114, 76, 20],
            'Électricité (râpage) (kWh/t fufu)': [68, 73, 68, '-'],
            'Carburant (râpé) (L/ t fufu)': ['-', '-', '-', 7.7],
            'Diesel (brûleur) (L/t fufu)': [83, 83, 84, '-'],
            'Centre de distribution/vente': ['Kinshasa'] * 4,
            'Distance (km)': [100, 10, 138, 350],
            'Type de transport': ['Camion', 'pick-up', 'Camion', 'Bus'],
            'Capacité (ton)': [10, 2, 10, 1]
        }

        # Creación del DataFrame para la segunda tabla
        df2 = pd.DataFrame(data2)
        df2.set_index('Unité de transformation', inplace=True)

        # Despliegue de los DataFrames en Streamlit
        st.title('Données d’inventaire de cycle de vie (ICV)')

        st.subheader('Tableau 1: ICV pour la culture du manioc et le transport du champ vers l’unité de transformation')
        st.dataframe(df1.T)

        st.subheader('Tableau 2: ICV pour la transformation du manioc en fufu et le transport vers le centre de distribution')
        st.dataframe(df2.T)

        st.subheader('Figure 1. Quatre scénarios envisagés pour l’analyse du cycle de vie à l’aide de SimaPro sont représentés')
        st.image('Data/Figure2.png', caption="Quatre scénarios envisagés")

with VC:
        
    if DD == "**Cassava Value Chain (agrégée)**":

            # Definir los datos en un diccionario
            data = {
                "Category": ["Climate change", "Ozone depletion", "Terrestrial acidification", "Freshwater eutrophication", "Marine eutrophication", 
                             "Human toxicity", "Photochemical oxidant formation", "Particulate matter formation", "Terrestrial ecotoxicity", 
                             "Freshwater ecotoxicity", "Marine ecotoxicity", "Ionising radiation", "Agricultural land occupation", "Urban land occupation", 
                             "Natural land transformation", "Water depletion", "Metal depletion", "Fossil depletion"],
                "Unit": ["kg CO2 eq", "kg CFC-11 eq", "kg SO2 eq", "kg P eq", "kg N eq", 
                         "kg 1,4-DB eq", "kg NMVOC", "kg PM10 eq", "kg 1,4-DB eq", 
                         "kg 1,4-DB eq", "kg 1,4-DB eq", "kBq U235 eq", "m2a", "m2a", 
                         "m2", "m3", "kg Fe eq", "kg oil eq"],
                "VC_Kinshasa": [613, 7.69e-5, 3.37, 3.29, 0.167, 75.5, 4.86, 1.78, 0.106, 0.341, 0.782, 27.9, 78.6, 21.7, 0.156, 5.66, 48.8, 173],
                "VC_Madimba": [529, 6.27e-5, 2.92, 3.28, 0.144, 60.3, 4.2, 1.55, 0.0795, 0.29, 0.573, 22.8, 74.3, 18, 0.126, 5.43, 44.2, 144],
                "VC_Matadi": [213, 4.3e-5, 0.715, 2.46, 0.0281, 16.7, 1.42, 0.305, 0.0303, 0.122, 0.269, 14.6, 4.57, 3.42, 0.0821, 4.86, 5.17, 83.6],
                "VC_Plateau_Bateke": [568, 6.89e-5, 3.14, 3.29, 0.155, 66.8, 4.53, 1.67, 0.09, 0.312, 0.652, 25, 78.1, 19.7, 0.139, 5.53, 47, 157]
            }

            # Crear DataFrame
            df = pd.DataFrame(data)

            # Combinar categoría con unidad
            df['Category'] = df['Category'] + " (" + df['Unit'] + ")"

            # Eliminar la columna 'Unit' ya que ya está combinada
            df.drop('Unit', axis=1, inplace=True)

            # Normalizar los valores al máximo de cada fila
            df_normalized = df.set_index("Category")
            df_normalized = df_normalized.div(df_normalized.max(axis=1), axis=0) * 100

            # Configuración de Streamlit
            st.title("Analyse comparative de la chaîne de valeur (agrégée)")
            st.write("Impacts environnementaux potentiels de la production d'une tonne de fufu à partir de quatre unités de transformation.")



            # Mostrar el DataFrame original y normalizado
            st.subheader("Données de sortie SIMAPRO")
            st.dataframe(df)
            st.subheader("Données normalisées")
            st.write("Pour chaque catégorie d’impact, l’unité de transformation avec l’impact le plus élevé est représentée avec un indice 100 (au lieu des unités d’origine), pour faciliter les comparaisons.")

            st.dataframe(df_normalized)

            # Selector múltiple para las columnas
            options = st.multiselect("Sélectionnez les lieux à afficher:", ["VC_Kinshasa", "VC_Madimba", "VC_Matadi", "VC_Plateau_Bateke"], default=["VC_Kinshasa","VC_Madimba", "VC_Matadi", "VC_Plateau_Bateke"])
            # Filtrar el dataframe para incluir solo las columnas seleccionadas
            if options:
                filtered_data = df_normalized[options]

                # Crear y mostrar el gráfico
                fig, ax = plt.subplots(figsize=(10, 8))
                filtered_data.plot(kind='barh', ax=ax)
                ax.set_xlabel("Percentage of Max Value (%)")
                ax.set_title("Impact Assessment by Category and Location")
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
                st.pyplot(fig)
            else:
                st.write("No locations selected. Please select at least one location.")

with TR:
    if DD == "**Cassava Value Chain (désagrégée)**":


            # Función para cargar los datos desde un archivo CSV con delimitador ','
            def load_data():
                df = pd.read_csv('Data/graphvc2.csv', delimiter=',', header=0)
                return df

            # Crear el gráfico de barras apiladas horizontal con categorías agrupadas
            def create_stacked_bar_chart(df, selected_categories, selected_locations):
                fig = go.Figure()

                # Definir colores para cada tipo de impacto
                colors = {
                    'Production': 'blue',
                    'Transport à l\'usine': 'orange',
                    'transformation': 'green',
                    'Transport au marché': 'red'
                }

                # Filtrar las categorías y localidades seleccionadas
                df_filtered = df[(df['Category of impact'].isin(selected_categories)) & (df['Location'].isin(selected_locations))]

                categories = df_filtered['Category of impact'].unique()
                locations = df_filtered['Location'].unique()

                # Crear etiquetas en el eje Y con categorías y localidades combinadas
                y_labels = []
                for category in categories:
                    for location in locations:
                        y_labels.append(f"{category} - {location}")

                # Añadir barras apiladas para cada categoría y localidad
                for column in ['Production', 'Transport à l\'usine', 'transformation', 'Transport au marché']:
                    values = []
                    for category in categories:
                        for location in locations:
                            value = df_filtered[(df_filtered['Category of impact'] == category) & (df_filtered['Location'] == location)][column].values
                            values.append(value[0] if len(value) > 0 else 0)
                    fig.add_trace(go.Bar(
                        y=y_labels,
                        x=values,
                        name=column,
                        orientation='h',
                        marker_color=colors[column]
                    ))

                # Agregar anotaciones (etiquetas) al lado de las barras más grandes
                for i, y_label in enumerate(y_labels):
                    total_value = sum([df_filtered[(df_filtered['Category of impact'] == y_label.split(" - ")[0]) & (df_filtered['Location'] == y_label.split(" - ")[1])][col].values[0] for col in colors.keys()])
                    location = y_label.split(" - ")[1]
                    fig.add_annotation(
                        x=total_value + 0.5,  # Coloca la etiqueta al lado de la barra
                        y=y_label,
                        text=location,
                        showarrow=False,
                        font=dict(size=12),
                        xanchor="left",
                        yanchor="middle"
                    )

                # Mejora la disposición de las etiquetas en el eje Y
                fig.update_layout(
                    barmode='stack',
                    title="Impacts environnementaux classés par catégorie et par lieu",
                    xaxis=dict(title="Impacto"),
                    yaxis=dict(
                        title="Categoría de Impacto",
                        tickmode="array",
                        tickvals=[i * len(locations) + len(locations) / 2 - 0.5 for i in range(len(categories))],
                        ticktext=categories
                    ),
                    showlegend=True,
                    height=1500  # Ajustar según sea necesario
                )

                return fig, df_filtered

            # Función principal para ejecutar la aplicación Streamlit
            def main():
                st.title('Visualisation des impacts environnementaux de la production de 1 ton de fufu dans quatre localités')
                df = load_data()

                st.subheader("Impacts environnementaux potentiels de la production d'une tonne de fufu à partir de quatre unités de transformation.")

                # Obtener listas de todas las categorías y localidades
                all_categories = df['Category of impact'].unique().tolist()
                all_locations = df['Location'].unique().tolist()

                # Crear selectores multiselect para categorías de impacto y localidades
                selected_categories = st.multiselect("Sélectionner les catégories d'impact", all_categories, default=all_categories)
                selected_locations = st.multiselect('Sélectionner les lieux', all_locations, default=all_locations)

                # Crear la gráfica con las categorías y localidades seleccionadas
                fig, df_filtered = create_stacked_bar_chart(df, selected_categories, selected_locations)
                st.plotly_chart(fig, use_container_width=True)

                # Reorganizar las columnas para que "Category of impact" sea la primera
                columns_order = ['Category of impact', 'Location', 'Production', 'Transport à l\'usine', 'transformation', 'Transport au marché']
                df_filtered = df_filtered[columns_order]

                # Mostrar la tabla con los datos filtrados
                st.subheader("Données normalisées et désagrégées par catégorie d'impact et par lieu.")
                st.dataframe(df_filtered)

            if __name__ == "__main__":
                main()

with PR:
    if DD == "**Production de racines de manioc**":


            # Definir los datos en un diccionario
            data = {
                "Category": [
                    "Climate change", "Ozone depletion", "Terrestrial acidification", 
                    "Freshwater eutrophication", "Marine eutrophication", "Human toxicity", 
                    "Photochemical oxidant formation", "Particulate matter formation", 
                    "Terrestrial ecotoxicity", "Freshwater ecotoxicity", "Marine ecotoxicity", 
                    "Ionising radiation", "Agricultural land occupation", "Urban land occupation", 
                    "Natural land transformation", "Water depletion", "Metal depletion", "Fossil depletion"
                ],
                "Unit": [
                    "kg CO2 eq", "kg CFC-11 eq", "kg SO2 eq", "kg P eq", "kg N eq", 
                    "kg 1,4-DB eq", "kg NMVOC", "kg PM10 eq", "kg 1,4-DB eq", 
                    "kg 1,4-DB eq", "kg 1,4-DB eq", "kBq U235 eq", "m2a", "m2a", 
                    "m2", "m3", "kg Fe eq", "kg oil eq"
                ],
                "P1 Menkao": [
                    46.8, 5.83e-6, 3.12, 3.23, 0.0165, 5.76, 0.474, 0.168,
                    0.00715, 0.0224, 0.0418, 2.17, 8.18, 1.84, 0.0126, 0.0946,
                    4.76, 13.2
                ],
                "P2 Madimba": [
                    22.7, 9.71e-7, 0.532, 0.754, 0.00283, 0.961, 0.081, 0.0284,
                    0.00119, 0.00374, 0.00802, 0.362, 1.36, 0.307, 0.00211, 0.066,
                    0.793, 2.2
                ],
                "P3 Mbanza Ngungu": [
                    25.5, 1.34e-6, 0.713, 1.13, 0.00376, 1.33, 0.108, 0.0384,
                    0.00165, 0.00516, 0.0111, 0.499, 1.88, 0.424, 0.00291, 0.0793,
                    1.09, 3.04
                ],
                "P4 Seeke-Banza": [
                    25.8, 5.82e-7, 0.546, 0.566, 0.00187, 0.576, 0.0532, 0.0181,
                    0.000715, 0.00224, 0.0048, 0.217, 0.817, 0.184, 0.00337, 0.0993,
                    0.475, 1.32
                ]
            }

            # Crear DataFrame
            df = pd.DataFrame(data)

            # Combinar categoría con unidad
            df['Category'] = df['Category'] + " (" + df['Unit'] + ")"

            # Eliminar la columna 'Unit' ya que ya está combinada
            df.drop('Unit', axis=1, inplace=True)

            # Normalizar los valores al máximo de cada fila
            df_normalized = df.set_index("Category")
            df_normalized = df_normalized.div(df_normalized.max(axis=1), axis=0) * 100

            # Configuración de Streamlit
            st.title("Analyse Comparative de l'Impact Environnemental Normalisé")
            st.write("Visualisation des impacts environnementaux normalisés pour diverses localités.")

            # Mostrar el DataFrame original y normalizado
            st.subheader("Données Originales")
            st.dataframe(df)
            st.subheader("Données Normalisées")
            st.dataframe(df_normalized)

            # Selector múltiple para las columnas
            options = st.multiselect("Sélectionnez les localités à afficher:", list(df_normalized.columns), default=list(df_normalized.columns))

            # Filtrar el dataframe para incluir solo las columnas seleccionadas
            if options:
                filtered_data = df_normalized[options]

                # Crear y mostrar el gráfico
                fig, ax = plt.subplots(figsize=(10, 8))
                filtered_data.plot(kind='barh', ax=ax)
                ax.set_xlabel("Pourcentage de la valeur maximale (%)")
                ax.set_title("Évaluation de l'Impact par Catégorie et Localité")
                st.pyplot(fig)
            else:
                st.write("Aucune localité sélectionnée. Veuillez en sélectionner au moins une.")

st.markdown('*Copyright (C) 2024 CIRAD, AGRINATURA*')
st.caption('**Authors: Alejandro Taborda, (latabordaa@unal.edu.co), Thierry Tran**')

