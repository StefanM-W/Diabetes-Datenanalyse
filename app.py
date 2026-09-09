import joblib
import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title='Diabetes Dashboard', page_icon='💉', layout='wide')


# Caching für Daten + ML Model
@st.cache_data
def load_data():
    return pd.read_csv('diabetesdata_cleaned.csv')

df = load_data()

@st.cache_resource
def load_model():
    return joblib.load('baseline_model.pkl')

model = load_model()



# Spaltenreihenfolge aus dem Modell 
BASELINE_FEATURES = list(model.feature_names_in_)


# Funktion zur Berechnung einer marginalisierten Wahrscheinlichkeit für fehlende Parameter
def marginalisierte_wahrscheinlichkeit(bekannte_werte):
    basis = df[BASELINE_FEATURES].copy()
    for spalte, wert in bekannte_werte.items():
        basis[spalte] = wert

    return model.predict_proba(basis)[:, 1].mean()


# Konfiguration der Eingabefelder
ALLE_FEATURES = {
    'Pregnancies': 'Schwangerschaften',
    'Glucose': 'Glukose (mg/dL)',
    'BloodPressure': 'Blutdruck (mm Hg)',
    'SkinThickness': 'Hautfaltendicke (mm)',
    'Insulin': 'Insulin (mu U/ml)',
    'BMI': 'BMI (kg/m²)',
    'DiabetesPedigreeFunction': 'Diabetes Pedigree Function',
    'Age': 'Alter',
}

# Min/Max/Default fürs jeweilige number_input-Widget
FEATURE_WIDGET_CONFIG = {
    'Pregnancies': (0, 20, 3),
    'Glucose': (10, 1000, 120),
    'BloodPressure': (20, 140, 70),
    'SkinThickness': (0, 100, 20),
    'Insulin': (0, 900, 80),
    'BMI': (10.0, 80.0, 32.0),
    'DiabetesPedigreeFunction': (0.078, 2.42, 0.5),
    'Age': (18, 100, 22),
}


# Eingabemaske
def eingabe_maske(features):
    eingaben = {}
    spalten = st.columns(3)
    for i, feature in enumerate(features):
        min_wert, max_wert, default_wert = FEATURE_WIDGET_CONFIG[feature]
        with spalten[i % 3]:
            eingaben[feature] = st.number_input(ALLE_FEATURES[feature], min_wert, max_wert, default_wert)

    return eingaben

# ERgebnisanzeige
def zeige_ergebnis(p_diabetes):
    st.write('---')
    st.subheader('Ergebnis:')

    if p_diabetes > 0.5:
        st.error(f'Risiko für Diabetes! (Wahrscheinlichkeit: {p_diabetes:.1%})')
    else:
        st.success(f'Kein Diabetes (Wahrscheinlichkeit: {1 - p_diabetes:.1%})')

    col1, col2 = st.columns(2)
    col1.metric('Kein Diabetes', f'{1 - p_diabetes:.1%}')
    col2.metric('Diabetes', f'{p_diabetes:.1%}')


# Main Function
def main():
    st.markdown("<h1 style='text-align: center;margin-bottom: 25px;'>Diabetes Dashboard 💉</h1>", unsafe_allow_html=True)


    # Navigationsleiste
    seite = st.sidebar.radio('Navigation', ['Übersicht', 'Daten', 'Visualisierungen','Prognose'])

    # Datenübersicht und Statistik
    if seite == 'Übersicht':
        st.write(f'Datensatz enthält **{len(df)} Patienten** und **{len(df.columns)} Features**')
        st.dataframe(df.head(10), width='stretch')

        st.subheader('Statistiken')
        st.write(df.describe())

    # Datenexploration nach einzelnen Features
    elif seite == 'Daten':
        st.markdown("<h2 style='text-align: center;margin-bottom: 50px;'>Feature Explorer🔍</h2>", unsafe_allow_html=True)
        # Feature auswählen
        feature = st.sidebar.selectbox(
            'Welches Feature willst du sehen?',
            options=['Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Pregnancies'],
            help='Wähle ein numerisches Feature zur Visualisierung'
        )

        # Info anzeigen
        st.write(f'**Gewähltes Feature:** {feature}')
        st.write(f'**Durchschnitt:** {df[feature].mean():.2f}')
        st.write(f'**Median:** {df[feature].median():.2f}')

        # Histogram
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(df[feature], bins=20, edgecolor='black', color='skyblue')
        ax.set_xlabel(feature)
        ax.set_ylabel('Häufigkeit')
        ax.set_title(f'Verteilung: {feature}')
        st.pyplot(fig, width='content')

    # Visualisierungen
    elif seite == 'Visualisierungen':

        st.markdown("<h2 style='text-align: center;margin-bottom: 50px;'>Visualisierungen 📊</h2>", unsafe_allow_html=True)

        # Auswahl in der Sidebar
        abbildung = st.sidebar.radio(
            "📍 Abbildung wählen",
            [
                "Zielvariable",
                "BMI vs. Alter",
                "Glukose nach Outcome",
                "Adipositas-Kategorien",
                "Glukose vs. BMI"
            ]
        )

        # Visualisierung basierend auf der Auswahl

        if abbildung == "Zielvariable":
            st.subheader('Verteilung der Zielvariable')
            col1, col2 = st.columns(2)
            with col1:
                counts = df['Outcome'].value_counts()
                labels = ['Kein Diabetes (0)', 'Diabetes (1)']
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.pie(counts.values, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'forestGreen'],
                   textprops={'fontsize': 8})
                st.pyplot(fig, width='content')
            with col2:
                st.markdown("<br> </br>", unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                ax2.hist(df['Age'], bins=20, edgecolor='black', color='skyblue')
                ax2.set_xlabel('Alter', fontsize=12)
                ax2.set_ylabel('Häufigkeit', fontsize=12)
                ax2.set_title('Altersverteilung der Patientinnen')
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2, width='content')


        elif abbildung == "BMI vs. Alter":
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            sns.scatterplot(data=df, x='Age', y='BMI', hue='Outcome', ax=ax3)
            ax3.set_title('BMI vs. Alter')
            st.pyplot(fig3, width='content')

        elif abbildung == "Glukose nach Outcome":
            fig4, ax4 = plt.subplots(1, 3, figsize=(14, 6))

            df[df['Outcome'] == 0]['Glucose'].hist(ax=ax4[0], bins=50, color='green', alpha=0.5, edgecolor='black', label='Kein Diabetes')
            df[df['Outcome'] == 1]['Glucose'].hist(ax=ax4[0], bins=50, color='red', alpha=0.5, edgecolor='black', label='Diabetes')
            ax4[0].set_xlabel('Glukose (mg/dL)')
            ax4[0].set_ylabel('Häufigkeit')
            ax4[0].set_title('Glukose nach Outcome')
            ax4[0].legend()

            gruppe0 = df[df['Outcome'] == 0]['Glucose']
            gruppe1 = df[df['Outcome'] == 1]['Glucose']

            ax4[1].boxplot(gruppe0)
            ax4[1].set_title('Glukoseverteilung kein Diabetes')
            ax4[1].set_ylabel('Glukose (mg/dL)')

            ax4[2].boxplot(gruppe1)
            ax4[2].set_title('Glukoseverteilung Diabetes')
            ax4[2].set_ylabel('Glukose (mg/dL)')
            st.pyplot(fig4, width='content')

        elif abbildung == "Adipositas-Kategorien":
            adipositas_order = [
                'untergewichtig', 'normalgewichtig', 'praeadipositas',
                'adipositas_grade_I', 'adipositas_grade_II', 'adipositas_grade_III'
            ]

            fig5, ax5 = plt.subplots(1, 2, figsize=(22, 6))

            counts_adip = df['Adipositas'].value_counts().reindex(adipositas_order)

            ax5[0].barh(range(len(counts_adip)), counts_adip.values, color='lightblue', edgecolor='black')
            ax5[0].set_yticks(range(len(counts_adip)))
            ax5[0].set_yticklabels(adipositas_order)
            ax5[0].set_xlabel('Anzahl Patientinnen')
            ax5[0].set_title('Verteilung der Adipositas-Kategorien')

            diabetes_rate = df.groupby('Adipositas')['Outcome'].mean().reindex(adipositas_order) * 100
            ax5[1].barh(range(len(diabetes_rate)), diabetes_rate.values, color='forestgreen', edgecolor='black')
            ax5[1].set_yticks(range(len(diabetes_rate)))
            ax5[1].set_yticklabels(adipositas_order)
            ax5[1].set_xlabel('Diabetes-Rate (%)')
            ax5[1].set_title('Diabetes-Rate je Adipositas-Kategorie')
            st.pyplot(fig5)

        elif abbildung == "Glukose vs. BMI":
            fig6, ax6 = plt.subplots(1, 1, figsize=(16, 6))

            colors = ['green' if c == 0 else 'red' for c in df['Outcome']]
            ax6.scatter(df['BMI'], df['Glucose'], c=colors, alpha=0.5)

            ax6.set_xlabel('BMI (kg/m²)')
            ax6.set_ylabel('Glukose (mg/dL)')
            ax6.set_title('Glukose vs. BMI – nach Diabetes-Status')
            st.pyplot(fig6)

    # Prognose anhand ML mit allen Parametern oder frei wählbaren Parametern
    elif seite == 'Prognose':
        st.markdown("<h2 style='text-align: center;margin-bottom: 50px;'>Diabetes-Prognose 🎓</h2>", unsafe_allow_html=True)

        st.sidebar.subheader('🧠 Eingabe auswählen')
        modus = st.sidebar.radio(
            'Womit soll die Prognose erstellt werden?',
            ['Alle Parameter', 'Nur ausgewählte Parameter'],
            help='In beiden Fällen rechnet das im ML-Notebook trainierte Modell. '
                 'Bei einer Teilauswahl wird über die nicht angegebenen Merkmale gemittelt.'
        )

        if modus == 'Alle Parameter':
            st.info(
                'Verwendet wird das im ML-Notebook trainierte und gespeicherte Modell mit allen Parametern'
            )

            st.write('Gib die Patientendaten ein:')
            eingaben = eingabe_maske(BASELINE_FEATURES)

            if st.button('Vorhersage', type='primary'):
                zeige_ergebnis(marginalisierte_wahrscheinlichkeit(eingaben))

        else:
            st.sidebar.subheader('🧬 Parameter auswählen')
            ausgewaehlt = st.sidebar.multiselect(
                'Welche Patientendaten sollen für die Prognose genutzt werden?',
                options=list(ALLE_FEATURES.keys()),
                default=['Glucose', 'Pregnancies'],
                format_func=lambda f: ALLE_FEATURES[f],
            )

            if len(ausgewaehlt) == 0:
                st.warning('Bitte wähle mindestens einen Parameter in der Sidebar aus.')
            else:
                gewaehlte_namen = ', '.join(ALLE_FEATURES[f] for f in ausgewaehlt)
                st.info(
                    f'Es rechnet weiterhin das Modell aus dem ML-Notebook. Angegeben werden '
                    f'**{len(ausgewaehlt)} von {len(BASELINE_FEATURES)} Parametern**: {gewaehlte_namen}  \n'
                    f'Über die übrigen Merkmale wird gemittelt'
                )
                if len(ausgewaehlt) < 4:
                    st.caption('⚠️ Je weniger Parameter angegeben sind, desto näher liegt das '
                               'Ergebnis an der Grundrate des Datensatzes.')

                st.write('Gib die Patientendaten ein:')
                eingaben = eingabe_maske(ausgewaehlt)

                if st.button('Vorhersage', type='primary'):
                    zeige_ergebnis(marginalisierte_wahrscheinlichkeit(eingaben))

        # CSV-Upload für Prognose
        st.write('---')
        st.subheader('📁 Eigene CSV-Datei hochladen')

        uploaded_file = st.file_uploader('CSV-Datei wählen', type='csv')

        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f'Datei konnte nicht gelesen werden: {e}')
            else:
                st.success(f'Datei geladen: {uploaded_file.name}')
                st.write(f'Anzahl Zeilen: {len(upload_df)}')
                st.dataframe(upload_df, width='stretch')

                upload_features = [f for f in ALLE_FEATURES if f in upload_df.columns]

                # Warnung, wenn keine bekannten Features in der Datei enthalten sind
                if len(upload_features) == 0:
                    st.error(
                        'Die Datei enthält keine bekannten Patientendaten-Spalten. '
                        'Mögliche Spalten: ' + ', '.join(ALLE_FEATURES.keys())
                    )

                # Berechnung wie bei der Einzeleingabe

                else:
                    st.info(f'Verwendete Spalten: {", ".join(upload_features)}')

                    if st.button('Prognose für alle Zeilen erstellen'):
                        wahrscheinlichkeiten = [
                            marginalisierte_wahrscheinlichkeit(zeile)
                            for zeile in upload_df[upload_features].to_dict('records')
                        ]

                        ergebnis_df = upload_df.copy()
                        ergebnis_df['Prognose'] = ['Diabetes' if p > 0.5 else 'Kein Diabetes' for p in wahrscheinlichkeiten]git remote add origin https://github.com/<DEIN-NAME>/<REPO>.git
                        ergebnis_df['Wahrscheinlichkeit_Diabetes'] = wahrscheinlichkeiten

                        st.write('Ergebnis:')
                        st.dataframe(ergebnis_df, width='stretch')


# Run
if __name__ == '__main__':
    main()
