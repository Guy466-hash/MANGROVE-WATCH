import os
from datetime import date, datetime

import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MANGROVE WATCH",
    page_icon="🌿",
    layout="wide"
)

# Connexion persistante à Supabase
@st.cache_resource
def obtenir_client_supabase():
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"]
    )

supabase = obtenir_client_supabase()


# ==========================================
# 2. AUTHENTIFICATION DES UTILISATEURS
# ==========================================

URL_APPLICATION = st.secrets.get("APP_URL", "http://localhost:8501")

# Récupération du retour de Google après authentification
code_oauth = st.query_params.get("code")

if code_oauth and not st.session_state.get("auth_code_processed"):
    try:
        resultat = supabase.auth.exchange_code_for_session(
    {"auth_code": code_oauth}
)
        session = resultat.session

        st.session_state["access_token"] = session.access_token
        st.session_state["refresh_token"] = session.refresh_token
        st.session_state["auth_code_processed"] = True

        st.query_params.clear()
        st.rerun()

    except Exception as erreur:
        st.error(f"Erreur de connexion Google : {erreur}")

# Restauration de la session utilisateur
if st.session_state.get("access_token"):
    try:
        supabase.auth.set_session(
            st.session_state["access_token"],
            st.session_state["refresh_token"]
        )
        utilisateur = supabase.auth.get_user().user

    except Exception:
        st.session_state.pop("access_token", None)
        st.session_state.pop("refresh_token", None)
        st.session_state.pop("auth_code_processed", None)
        utilisateur = None
else:
    utilisateur = None

# Affichage de la connexion si aucun utilisateur n'est connecté
if utilisateur is None:
    st.title("🌿 MANGROVE WATCH")
    st.subheader("Connexion à la plateforme")

    if st.button("🔐 Se connecter avec Google", use_container_width=True):
        resultat = supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": URL_APPLICATION
            }
        })
        st.link_button("Continuer avec Google", resultat.url)

    st.stop()

# Affichage du compte connecté
st.sidebar.success(f"Connecté : {utilisateur.email}")

if st.sidebar.button("Se déconnecter"):
    supabase.auth.sign_out()
    st.session_state.clear()
    st.rerun()

FICHIER_MODELE = "modele_mangrove_genipa.joblib"
FICHIER_HISTORIQUE = "historique_evaluations.csv"
CENTRE_GENIPA = [14.5504759, -60.9934514]

CLASSES = {0: "Faible", 1: "Modérée", 2: "Forte"}

VARIABLES = [
    "couverture_vegetale",
    "fragmentation",
    "pression_anthropique",
    "turbidite",
    "biodiversite",
    "salinite",
    "oxygene_dissous",
    "regeneration"
]

LIBELLES = {
    "couverture_vegetale": "Couverture végétale (%)",
    "fragmentation": "Fragmentation (%)",
    "pression_anthropique": "Pression anthropique (%)",
    "turbidite": "Turbidité (NTU)",
    "biodiversite": "Indice de biodiversité (0 à 10)",
    "salinite": "Salinité (PSU)",
    "oxygene_dissous": "Oxygène dissous (mg/L)",
    "regeneration": "Régénération végétale (%)",
    "indice_synthetique": "Indice synthétique de dégradation (0 à 100)",
    "classe_num": "Classe de dégradation (0 à 2)"
}

# L'indice reprend la formule utilisée pour créer les classes
# du jeu de données expérimental. La salinité n'y figurait pas.
def calculer_indice_synthetique(ligne):
    indice = (
        0.20 * (100 - float(ligne["couverture_vegetale"]))
        + 0.15 * float(ligne["fragmentation"])
        + 0.20 * float(ligne["pression_anthropique"])
        + 0.10 * (float(ligne["turbidite"]) / 60 * 100)
        + 0.15 * (10 - float(ligne["biodiversite"])) * 10
        + 0.10 * (10 - float(ligne["oxygene_dissous"])) * 10
        + 0.10 * (100 - float(ligne["regeneration"]))
    )
    return float(np.clip(indice, 0, 100))


def classe_vers_nombre(valeur):
    correspondance = {"Faible": 0, "Modérée": 1, "Forte": 2}
    return correspondance.get(str(valeur), np.nan)


# ============================================================
# 2. CHARGEMENT DU MODELE
# ============================================================

@st.cache_resource
def charger_modele():
    return joblib.load(FICHIER_MODELE)


try:
    modele = charger_modele()
except Exception as erreur:
    st.error("Impossible de charger le modèle de prédiction.")
    st.exception(erreur)
    st.stop()


# ============================================================
# 3. HISTORIQUE : LECTURE ET ENREGISTREMENT
# ============================================================

def charger_historique():
    if os.path.exists(FICHIER_HISTORIQUE):
        try:
            historique = pd.read_csv(FICHIER_HISTORIQUE)
            return historique
        except Exception as erreur:
            st.error(f"Impossible de lire l'historique : {erreur}")
    return pd.DataFrame()



def enregistrer_evaluation(donnees):
    """
    Enregistre une observation dans le CSV local et dans Supabase.
    La sauvegarde locale est conservée même si l'envoi distant échoue.
    """

    # 1. Calcul de l'indice synthétique avec la fonction commune
    indice = calculer_indice_synthetique(donnees)

    # 2. Sauvegarde locale dans le fichier CSV
    try:
        historique = charger_historique()
        nouvelle_ligne = pd.DataFrame([donnees])
        historique = pd.concat(
            [historique, nouvelle_ligne],
            ignore_index=True
        )
        historique.to_csv(FICHIER_HISTORIQUE, index=False)
    except Exception as erreur:
        st.error(f"Impossible d'enregistrer l'historique local : {erreur}")
        return

    # 3. Préparation des données à envoyer vers Supabase
    donnees_supabase = {
        "user_id": utilisateur.id,
        "code_station": donnees["code_station"],
        "nom_station": donnees["nom_station"],
        "secteur": donnees["secteur"],
        "date_observation": donnees["date_observation"],
        "latitude": donnees["latitude"],
        "longitude": donnees["longitude"],
        "source_position": donnees["source_position"],
        "couverture_vegetale": donnees["couverture_vegetale"],
        "fragmentation": donnees["fragmentation"],
        "pression_anthropique": donnees["pression_anthropique"],
        "turbidite": donnees["turbidite"],
        "biodiversite": donnees["biodiversite"],
        "salinite": donnees["salinite"],
        "oxygene_dissous": donnees["oxygene_dissous"],
        "regeneration": donnees["regeneration"],
        "indice_synthetique": indice,
        "classe_degradation": donnees["classe_predite"]
    }

    # 4. Envoi dans la table observations de Supabase
    try:
        supabase.table("observations").insert(
            donnees_supabase
        ).execute()
        st.success("Observation enregistrée localement et dans Supabase.")
    except Exception as erreur:
        st.warning(
            "Observation enregistrée dans le fichier local, "
            "mais l'envoi vers Supabase a échoué."
        )
        st.error(f"Détail de l'erreur Supabase : {erreur}")


def preparer_historique(historique):
    """Ajoute les champs nécessaires aux graphiques sans modifier le CSV."""
    if historique.empty:
        return historique

    historique = historique.copy()

    colonnes_numeriques = VARIABLES + ["latitude", "longitude"]
    for colonne in colonnes_numeriques:
        if colonne in historique.columns:
            historique[colonne] = pd.to_numeric(
                historique[colonne], errors="coerce"
            )

    if "date_observation" in historique.columns:
        historique["date_observation"] = pd.to_datetime(
            historique["date_observation"], errors="coerce"
        )

    if all(c in historique.columns for c in VARIABLES if c != "salinite"):
        historique["indice_synthetique"] = historique.apply(
            calculer_indice_synthetique, axis=1
        )

    if "classe_predite" in historique.columns:
        historique["classe_num"] = historique["classe_predite"].map(
            classe_vers_nombre
        )

    return historique


# ============================================================
# 4. INITIALISATION DE LA POSITION
# ============================================================

if "latitude" not in st.session_state:
    st.session_state.latitude = CENTRE_GENIPA[0]
if "longitude" not in st.session_state:
    st.session_state.longitude = CENTRE_GENIPA[1]
if "precision_gps" not in st.session_state:
    st.session_state.precision_gps = None
if "source_position" not in st.session_state:
    st.session_state.source_position = "Position initiale"


# ============================================================
# 5. EN-TETE
# ============================================================

st.title("🌿 MANGROVE WATCH")
st.subheader("Plateforme de surveillance de la dégradation des mangroves")
st.markdown("**Zone d'étude : Baie de Génipa – Martinique**")
st.caption(
    "Suivi environnemental | Géolocalisation | Machine Learning | "
    "Analyse temporelle et territoriale"
)
st.divider()


# ============================================================
# 6. IDENTIFICATION ET LOCALISATION
# ============================================================

st.header("1. Identification et localisation de la station")

st.subheader("📍 A. Géolocalisation automatique")
st.write(
    "Sur le terrain, utilisez le bouton GPS et autorisez le navigateur "
    "à accéder à votre position."
)

position = streamlit_geolocation()

if position and position.get("latitude") is not None:
    latitude_gps = float(position["latitude"])
    longitude_gps = float(position["longitude"])
    precision = position.get("accuracy")

    st.session_state.latitude = latitude_gps
    st.session_state.longitude = longitude_gps
    st.session_state.precision_gps = precision
    st.session_state.source_position = "GPS automatique"

    st.success("Position GPS récupérée.")
    st.write(f"Latitude : {latitude_gps:.7f}")
    st.write(f"Longitude : {longitude_gps:.7f}")

    if precision is not None:
        st.write(f"Précision annoncée par le GPS : {float(precision):.1f} m")
        if float(precision) <= 5:
            st.success("Précision annoncée satisfaisante (≤ 5 m).")
        elif float(precision) <= 15:
            st.warning(
                "Précision moyenne. Attendez si possible une meilleure réception."
            )
        else:
            st.warning(
                "Précision actuellement insuffisante pour un relevé très précis. "
                "La précision dépend du matériel et des conditions de réception."
            )

st.caption(
    "La précision affichée est celle annoncée par l'appareil : "
    "L’application ne peut pas garantir une précision de 4 à 5 mètres."
)

st.divider()

st.subheader("✏️ B. Saisie manuelle des coordonnées GPS")
col1, col2 = st.columns(2)

with col1:
    latitude_manuelle = st.number_input(
        "Latitude GPS",
        min_value=-90.0,
        max_value=90.0,
        value=float(st.session_state.latitude),
        format="%.7f",
        key="latitude_manuelle"
    )

with col2:
    longitude_manuelle = st.number_input(
        "Longitude GPS",
        min_value=-180.0,
        max_value=180.0,
        value=float(st.session_state.longitude),
        format="%.7f",
        key="longitude_manuelle"
    )

if st.button("Appliquer les coordonnées manuelles"):
    st.session_state.latitude = latitude_manuelle
    st.session_state.longitude = longitude_manuelle
    st.session_state.source_position = "Saisie manuelle"
    st.success("Coordonnées manuelles appliquées.")
    st.rerun()

st.divider()

st.subheader("🗺️ C. Sélectionner un point sur la carte")
st.write("Zoomez sur la baie de Génipa, puis cliquez sur l'emplacement de la station.")

carte_selection = folium.Map(
    location=CENTRE_GENIPA,
    zoom_start=12,
    tiles="OpenStreetMap"
)

folium.Marker(
    location=[st.session_state.latitude, st.session_state.longitude],
    tooltip="Position actuellement retenue",
    popup="Position de la station",
    icon=folium.Icon(color="green", icon="leaf")
).add_to(carte_selection)

resultat_carte = st_folium(
    carte_selection,
    height=480,
    width=None,
    key="carte_selection_station"
)

if resultat_carte and resultat_carte.get("last_clicked"):
    point = resultat_carte["last_clicked"]
    st.session_state.latitude = float(point["lat"])
    st.session_state.longitude = float(point["lng"])
    st.session_state.source_position = "Sélection sur la carte"
    st.success(
        "Point sélectionné : "
        f"{st.session_state.latitude:.7f}, "
        f"{st.session_state.longitude:.7f}"
    )
    st.info("Les coordonnées seront utilisées au prochain enregistrement.")

col1, col2, col3 = st.columns(3)
col1.metric("Latitude retenue", f"{st.session_state.latitude:.7f}")
col2.metric("Longitude retenue", f"{st.session_state.longitude:.7f}")
col3.write(f"**Origine :** {st.session_state.source_position}")

st.divider()


# ============================================================
# 7. IDENTIFICATION DE LA STATION
# ============================================================

st.header("2. Informations sur la station")
col1, col2 = st.columns(2)

with col1:
    code_station = st.text_input(
        "Code de la station", placeholder="Exemple : GEN-001"
    )
    nom_station = st.text_input(
        "Nom de la station", placeholder="Exemple : Station mangrove nord"
    )

with col2:
    secteur = st.text_input("Secteur géographique", value="Baie de Génipa")
    date_observation = st.date_input(
        "Date de l'observation", value=date.today()
    )

st.divider()


# ============================================================
# 8. PARAMETRES ENVIRONNEMENTAUX
# ============================================================

st.header("3. Paramètres environnementaux")
st.write("Renseignez les paramètres mesurés ou estimés sur le terrain.")

col1, col2 = st.columns(2)

with col1:
    couverture_vegetale = st.number_input(
        "Couverture végétale (%)", 0.0, 100.0, 70.0
    )
    fragmentation = st.number_input(
        "Fragmentation (%)", 0.0, 100.0, 30.0
    )
    pression_anthropique = st.number_input(
        "Pression anthropique (%)", 0.0, 100.0, 30.0
    )
    turbidite = st.number_input("Turbidité (NTU)", 1.0, 60.0, 15.0)

with col2:
    biodiversite = st.number_input(
        "Indice de biodiversité (0 à 10)", 0.0, 10.0, 7.0
    )
    salinite = st.number_input("Salinité (PSU)", 5.0, 40.0, 25.0)
    oxygene_dissous = st.number_input(
        "Oxygène dissous (mg/L)", 1.0, 10.0, 7.0
    )
    regeneration = st.number_input(
        "Régénération végétale (%)", 0.0, 100.0, 70.0
    )

st.divider()


# ============================================================
# 9. EVALUATION ET ENREGISTREMENT
# ============================================================

st.header("4. Évaluation de la station")

if st.button(
    "🌿 Enregistrer et évaluer la station",
    type="primary",
    use_container_width=True
):
    if not code_station.strip():
        st.error("Veuillez renseigner le code de la station.")
    elif not nom_station.strip():
        st.error("Veuillez renseigner le nom de la station.")
    elif not secteur.strip():
        st.error("Veuillez renseigner le secteur géographique.")
    else:
        donnees_modele = pd.DataFrame(
            [[
                couverture_vegetale,
                fragmentation,
                pression_anthropique,
                turbidite,
                biodiversite,
                salinite,
                oxygene_dissous,
                regeneration
            ]],
            columns=VARIABLES
        )

        try:
            prediction = int(modele.predict(donnees_modele)[0])
            resultat = CLASSES[prediction]

            nouvelle_evaluation = {
                "date_enregistrement": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date_observation": str(date_observation),
                "code_station": code_station.strip().upper(),
                "nom_station": nom_station.strip(),
                "secteur": secteur.strip(),
                "couverture_vegetale": couverture_vegetale,
                "fragmentation": fragmentation,
                "pression_anthropique": pression_anthropique,
                "turbidite": turbidite,
                "biodiversite": biodiversite,
                "salinite": salinite,
                "oxygene_dissous": oxygene_dissous,
                "regeneration": regeneration,
                "classe_predite": resultat,
                "latitude": st.session_state.latitude,
                "longitude": st.session_state.longitude,
                "source_position": st.session_state.source_position
            }

            enregistrer_evaluation(nouvelle_evaluation)
            st.session_state.derniere_evaluation = nouvelle_evaluation
            st.success("Évaluation enregistrée avec succès !")
            st.rerun()

        except Exception as erreur:
            st.error("Une erreur est survenue pendant l'évaluation.")
            st.exception(erreur)


# ============================================================
# 10. RESULTAT DE LA DERNIERE EVALUATION
# ============================================================

if "derniere_evaluation" in st.session_state:
    evaluation = st.session_state.derniere_evaluation
    st.header("5. Résultat de la dernière évaluation")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Station :** {evaluation['code_station']}")
        st.write(f"**Nom :** {evaluation['nom_station']}")
        st.write(f"**Secteur :** {evaluation['secteur']}")
    with col2:
        st.write(f"**Latitude :** {evaluation['latitude']:.7f}")
        st.write(f"**Longitude :** {evaluation['longitude']:.7f}")
        st.write(f"**Date :** {evaluation['date_observation']}")

    if evaluation["classe_predite"] == "Faible":
        st.success(f"Niveau de dégradation estimé : {evaluation['classe_predite']}")
    elif evaluation["classe_predite"] == "Modérée":
        st.warning(f"Niveau de dégradation estimé : {evaluation['classe_predite']}")
    else:
        st.error(f"Niveau de dégradation estimé : {evaluation['classe_predite']}")

    indice_recent = calculer_indice_synthetique(evaluation)
    st.metric("Indice synthétique de dégradation", f"{indice_recent:.1f} / 100")
    st.caption(
        "Indice expérimental calculé avec la formule de génération des classes. "
        "Il ne constitue pas encore un indicateur écologique validé."
    )

    st.info(
        "Le modèle a été entraîné sur des données simulées. "
        "Ses résultats ne constituent pas un diagnostic écologique réel "
        "de la baie de Génipa."
    )

st.divider()


# ============================================================
# 11. HISTORIQUE DES EVALUATIONS
# ============================================================

st.header("6. Historique des évaluations")
historique_brut = charger_historique()
historique = preparer_historique(historique_brut)

if not historique.empty:
    st.metric("Nombre total d'évaluations", len(historique))
    st.dataframe(
        historique_brut,
        use_container_width=True,
        hide_index=True
    )

    csv_export = historique_brut.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Télécharger l'historique CSV",
        data=csv_export,
        file_name="historique_mangrove_watch.csv",
        mime="text/csv"
    )
else:
    st.info("Aucune évaluation enregistrée pour le moment.")

st.divider()


# ============================================================
# 12. ANALYSE TEMPORELLE ET INDICATEUR SYNTHETIQUE
# ============================================================

st.header("7. Évolution temporelle et tableau de bord des stations")

if not historique.empty and "code_station" in historique.columns:
    historique_graph = historique.dropna(
        subset=["code_station", "date_observation"]
    ).copy()

    if not historique_graph.empty:
        codes = sorted(historique_graph["code_station"].astype(str).unique())
        station_choisie = st.selectbox(
            "Sélectionner la station à analyser",
            codes,
            key="station_analyse"
        )

        donnees_station = historique_graph[
            historique_graph["code_station"].astype(str) == station_choisie
        ].copy()

        donnees_station = donnees_station.sort_values("date_observation")

        nom_affiche = (
            donnees_station["nom_station"].dropna().iloc[-1]
            if "nom_station" in donnees_station.columns
            and donnees_station["nom_station"].notna().any()
            else station_choisie
        )
        st.subheader(f"Station {station_choisie} — {nom_affiche}")

        derniere_ligne = donnees_station.iloc[-1]
        indice_actuel = float(derniere_ligne["indice_synthetique"])

        if indice_actuel < 35:
            libelle_indice = "Dégradation faible"
        elif indice_actuel < 65:
            libelle_indice = "Dégradation modérée"
        else:
            libelle_indice = "Dégradation forte"

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Indice synthétique actuel",
            f"{indice_actuel:.1f} / 100"
        )
        col2.metric(
            "Classe prédite",
            str(derniere_ligne.get("classe_predite", "Non disponible"))
        )
        col3.metric(
            "Nombre d'observations",
            len(donnees_station)
        )

        st.progress(int(round(indice_actuel)))
        st.caption(
            "Lecture de l'indice : 0 = pression/dégradation faible selon la formule, "
            "100 = pression/dégradation élevée. Seuils indicatifs : "
            "moins de 35, 35 à moins de 65, 65 et plus."
        )

        st.markdown("#### Évolution de l'indice synthétique")
        serie_indice = donnees_station.set_index("date_observation")[
            ["indice_synthetique"]
        ].rename(columns={"indice_synthetique": "Indice (0–100)"})
        st.line_chart(serie_indice, y="Indice (0–100)", use_container_width=True)

        st.markdown("#### Évolution d'un paramètre environnemental")
        choix_variables = list(VARIABLES)
        variable_choisie = st.selectbox(
            "Choisir le paramètre",
            choix_variables,
            format_func=lambda x: LIBELLES[x],
            key="variable_analyse"
        )

        serie_variable = donnees_station.dropna(
            subset=["date_observation", variable_choisie]
        ).set_index("date_observation")[[variable_choisie]]
        serie_variable = serie_variable.rename(
            columns={variable_choisie: LIBELLES[variable_choisie]}
        )

        if not serie_variable.empty:
            st.line_chart(
                serie_variable,
                y=LIBELLES[variable_choisie],
                use_container_width=True
            )
        else:
            st.info("Pas encore de valeurs disponibles pour cet indicateur.")

        st.markdown("#### Évolution de la classe prédite")
        classe_graph = donnees_station.dropna(
            subset=["date_observation", "classe_num"]
        ).set_index("date_observation")[["classe_num"]]
        if not classe_graph.empty:
            st.line_chart(
                classe_graph,
                y="classe_num",
                use_container_width=True
            )
            st.caption("0 = Faible | 1 = Modérée | 2 = Forte")
        else:
            st.info("Aucune classe prédite exploitable dans l'historique.")

        st.markdown("#### Comparaison des deux dernières observations")
        if len(donnees_station) >= 2:
            precedente = donnees_station.iloc[-2]
            actuelle = donnees_station.iloc[-1]

            indice_precedent = float(precedente["indice_synthetique"])
            variation = indice_actuel - indice_precedent

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Indice précédent",
                f"{indice_precedent:.1f} / 100"
            )
            c2.metric(
                "Indice le plus récent",
                f"{indice_actuel:.1f} / 100"
            )
            c3.metric(
                "Variation de l'indice",
                f"{variation:+.1f} point(s)",
                delta=f"{variation:+.1f}",
                delta_color="inverse"
            )

            if variation > 0:
                st.warning(
                    "L'indice augmente : selon la formule expérimentale, "
                    "les paramètres indiquent une dégradation plus élevée."
                )
            elif variation < 0:
                st.success(
                    "L'indice diminue : selon la formule expérimentale, "
                    "les paramètres indiquent une amélioration."
                )
            else:
                st.info("L'indice est stable entre les deux dernières observations.")
        else:
            st.info(
                "Il faut au moins deux observations à des dates différentes "
                "pour calculer une évolution."
            )

        with st.expander("Afficher les données de cette station"):
            colonnes_affichees = [
                c for c in [
                    "date_observation", "code_station", "nom_station",
                    "indice_synthetique", "classe_predite"
                ] + VARIABLES if c in donnees_station.columns
            ]
            st.dataframe(
                donnees_station[colonnes_affichees],
                use_container_width=True,
                hide_index=True
            )

        st.warning(
            "Important : l'indice synthétique reprend la formule du jeu de données "
            "simulé d'entraînement. La salinité est affichée comme variable "
            "individuelle, mais n'entre pas dans cet indice initial. "
            "Les tendances ne deviennent interprétables écologiquement qu'après "
            "validation avec des observations de terrain."
        )
    else:
        st.info(
            "Les dates ou codes de station ne permettent pas encore "
            "de construire les graphiques."
        )
else:
    st.info("Enregistrez des observations pour afficher les analyses temporelles.")

st.divider()


# ============================================================
# 13. CARTOGRAPHIE DES STATIONS
# ============================================================

st.header("8. Cartographie des stations")

if not historique.empty and {"latitude", "longitude"}.issubset(historique.columns):
    historique_geo = historique.copy()
    historique_geo["latitude"] = pd.to_numeric(
        historique_geo["latitude"], errors="coerce"
    )
    historique_geo["longitude"] = pd.to_numeric(
        historique_geo["longitude"], errors="coerce"
    )
    historique_geo = historique_geo.dropna(subset=["latitude", "longitude"])

    # La baie de Génipa se trouve approximativement entre ces bornes.
    # Cela évite notamment d'afficher des longitudes positives erronées.
    historique_geo = historique_geo[
        historique_geo["latitude"].between(14.0, 15.5)
        & historique_geo["longitude"].between(-62.0, -60.0)
    ]

    if not historique_geo.empty:
        carte_stations = folium.Map(
            location=CENTRE_GENIPA,
            zoom_start=12,
            tiles="OpenStreetMap"
        )

        couleurs = {"Faible": "green", "Modérée": "orange", "Forte": "red"}

        # Un marqueur par observation : les observations répétées d'une station
        # peuvent se superposer si les coordonnées sont identiques.
        for _, station in historique_geo.iterrows():
            couleur = couleurs.get(station.get("classe_predite"), "blue")
            folium.Marker(
                location=[float(station["latitude"]), float(station["longitude"])],
                tooltip=str(station.get("code_station", "Station")),
                popup=(
                    f"Station : {station.get('nom_station', '')}<br>"
                    f"Secteur : {station.get('secteur', '')}<br>"
                    f"Dégradation : {station.get('classe_predite', '')}<br>"
                    f"Date : {station.get('date_observation', '')}"
                ),
                icon=folium.Icon(color=couleur, icon="leaf")
            ).add_to(carte_stations)

        st_folium(
            carte_stations,
            height=540,
            width=None,
            key="carte_historique_stations"
        )
        st.caption(
            "Vert : dégradation faible | Orange : dégradation modérée | "
            "Rouge : dégradation forte. Les points hors de la zone géographique "
            "attendue sont exclus de cette carte."
        )
        st.metric(
            "Nombre de stations géolocalisées",
            historique_geo["code_station"].nunique()
        )
    else:
        st.info(
            "Aucune coordonnée valide dans la zone de la baie de Génipa. "
            "Vérifiez notamment le signe négatif de la longitude."
        )
else:
    st.info("La carte apparaîtra après le premier enregistrement géolocalisé.")


# ============================================================
# 14. PIED DE PAGE
# ============================================================

st.divider()
st.caption(
    "MANGROVE WATCH | Prototype de surveillance environnementale | "
    "Baie de Génipa – Martinique"
)
