import os
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

#___________________________________________PARKINSON PREDICTION__________________________________________

# Load the model
parkinson_model = pickle.load(open(r"F:\vscode\Multiple Disease Prediction\models\xgb_parkinson_model.pkl", "rb"))

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Kidney Prediction', 'Liver Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        menu_icon='hospital-fill',
        default_index=2  # Default to Parkinsons
    )

# Parkinson's Disease Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    # Input form
    st.subheader("Please enter the following values:")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        jitter_abs = st.text_input('MDVP:Jitter(Abs)')

    col6, col7, col8, col9, col10 = st.columns(5)

    with col6:
        rap = st.text_input('MDVP:RAP')
    with col7:
        ppq = st.text_input('MDVP:PPQ')
    with col8:
        ddp = st.text_input('Jitter:DDP')
    with col9:
        shimmer = st.text_input('MDVP:Shimmer')
    with col10:
        shimmer_db = st.text_input('MDVP:Shimmer(dB)')

    col11, col12, col13, col14, col15 = st.columns(5)

    with col11:
        apq3 = st.text_input('Shimmer:APQ3')
    with col12:
        apq5 = st.text_input('Shimmer:APQ5')
    with col13:
        apq = st.text_input('MDVP:APQ')
    with col14:
        dda = st.text_input('Shimmer:DDA')
    with col15:
        nhr = st.text_input('NHR')

    col16, col17, col18, col19, col20 = st.columns(5)

    with col16:
        hnr = st.text_input('HNR')
    with col17:
        rpde = st.text_input('RPDE')
    with col18:
        dfa = st.text_input('DFA')
    with col19:
        spread1 = st.text_input('spread1')
    with col20:
        spread2 = st.text_input('spread2')

    col21, col22, col23 = st.columns(3)

    with col21:
        d2 = st.text_input('D2')
    with col22:
        ppe = st.text_input('PPE')

    # Prediction
    parkinsons_diagnosis = ''
    if st.button("Predict Parkinson's"):
        try:
            input_data = np.array([
                float(fo), float(fhi), float(flo), float(jitter_percent), float(jitter_abs),
                float(rap), float(ppq), float(ddp), float(shimmer), float(shimmer_db),
                float(apq3), float(apq5), float(apq), float(dda), float(nhr),
                float(hnr), float(rpde), float(dfa), float(spread1), float(spread2),
                float(d2), float(ppe)
            ]).reshape(1, -1)

            prediction = parkinson_model.predict(input_data)[0]
            prediction_proba = parkinson_model.predict_proba(input_data)[0]

            risk_level = "High Risk" if prediction == 1 else "Low Risk"

            st.success(f"Prediction: {'Parkinsons Detected' if prediction == 1 else 'No Parkinsons'}")
            st.info(f"Probability of Parkinsons: {prediction_proba[1]*100:.2f}%")
            st.warning(f"Risk Level: {risk_level}")

        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

#___________________________________________LIVER PREDICTION__________________________________________
# Load the model
liver_model = pickle.load(open(r"F:\vscode\Multiple Disease Prediction\models\liver_best_rf_model.pkl", "rb"))

# Liver Disease Prediction Page
if selected == "Liver Prediction":
    st.title("Liver Disease Prediction using ML")

     # Input form
    st.subheader("Please enter the following values:")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        age = st.text_input('Age')
    with col2:
        gender = st.selectbox('Gender', ['Male', 'Female'])
    with col3:
        total_bilirubin = st.text_input('Total Bilirubin')
    with col4:
        direct_bilirubin = st.text_input('Direct Bilirubin')
    with col5:
        alkaline_phosphotase = st.text_input('Alkaline Phosphotase')

    col6, col7, col8, col9, col10 = st.columns(5)

    with col6:
        alamine_aminotransferase = st.text_input('Alamine Aminotransferase')
    with col7:
        aspartate_aminotransferase = st.text_input('Aspartate Aminotransferase')
    with col8:
        total_proteins = st.text_input('Total Proteins')
    with col9:
        albumin = st.text_input('Albumin')
    with col10:
        albumin_globulin_ratio = st.text_input('Albumin and Globulin Ratio')

    # Prediction
    liver_diagnosis = ''
    if st.button("Predict Liver Disease"):
        try:
            # Encode gender (Male=1, Female=0)
            gender_encoded = 1 if gender == 'Male' else 0

            # Collect inputs in order
            input_data = np.array([
                float(age), gender_encoded, float(total_bilirubin), float(direct_bilirubin),
                float(float(alkaline_phosphotase)), float(alamine_aminotransferase), float(aspartate_aminotransferase),
                float(total_proteins), float(albumin), float(albumin_globulin_ratio)
            ]).reshape(1, -1)


            # Make prediction
            prediction = liver_model.predict(input_data)[0]
            prediction_proba = liver_model.predict_proba(input_data)[0]

            # Output
            risk_level = "High Risk" if prediction == 1 else "Low Risk"

            st.success(f"Prediction: {'Liver Disease Detected' if prediction == 1 else 'No Liver Disease'}")
            st.info(f"Probability of Liver Disease: {prediction_proba[1]*100:.2f}%")
            st.warning(f"Risk Level: {risk_level}")

        except ValueError:
            st.error("Please enter valid numeric values for all fields.")


#___________________________________________KIDNEY PREDICTION__________________________________________

# Load the model
kidney_model = pickle.load(open(r"F:\vscode\Multiple Disease Prediction\models\rf_kidney_model.pkl", "rb"))

# Kidney Disease Prediction Page
if selected == "Kidney Prediction":
    st.title("Chronic Kidney Disease Prediction using ML")

    # Input form
    st.subheader("Please enter the following values:")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        age = st.text_input('Age')
    with col2:
        bp = st.text_input('Blood Pressure')
    with col3:
        sg = st.text_input('Specific Gravity')
    with col4:
        al = st.text_input('Albumin')
    with col5:
        su = st.text_input('Sugar')

    col6, col7, col8, col9, col10 = st.columns(5)

    with col6:
        rbc = st.text_input('Red Blood Cells (0=normal, 1=abnormal)')
    with col7:
        pc = st.text_input('Pus Cell (0=normal, 1=abnormal)')
    with col8:
        pcc = st.text_input('Pus Cell Clumps (0=notpresent, 1=present)')
    with col9:
        ba = st.text_input('Bacteria (0=notpresent, 1=present)')
    with col10:
        bgr = st.text_input('Blood Glucose Random')

    col11, col12, col13, col14, col15 = st.columns(5)

    with col11:
        bu = st.text_input('Blood Urea')
    with col12:
        sc = st.text_input('Serum Creatinine')
    with col13:
        sod = st.text_input('Sodium')
    with col14:
        pot = st.text_input('Potassium')
    with col15:
        hemo = st.text_input('Hemoglobin')

    col16, col17, col18, col19, col20 = st.columns(5)

    with col16:
        pcv = st.text_input('Packed Cell Volume')
    with col17:
        wc = st.text_input('White Blood Cell Count')
    with col18:
        htn = st.text_input('Hypertension (0=no, 1=yes)')
    with col19:
        dm = st.text_input('Diabetes Mellitus (0=no, 1=yes)')

    col21, col22, col23, col24 = st.columns(4)

    with col21:
        cad = st.text_input('Coronary Artery Disease (0=no, 1=yes)')
    with col22:
        appet = st.text_input('Appetite (0=good, 1=poor)')
    with col23:
        pe = st.text_input('Pedal Edema (0=no, 1=yes)')
    with col24:
        ane = st.text_input('Anemia (0=no, 1=yes)')

    # Prediction
    kidney_diagnosis = ''
    if st.button("Predict Kidney Disease"):
        try:
            input_data = np.array([
                float(age), float(bp), float(sg), float(al), float(su),
                float(rbc), float(pc), float(pcc), float(ba), float(bgr),
                float(bu), float(sc), float(sod), float(pot), float(hemo),
                float(pcv), float(wc), float(htn), float(dm),
                float(cad), float(appet), float(pe), float(ane)
            ]).reshape(1, -1)

            prediction = kidney_model.predict(input_data)[0]
            prediction_proba = kidney_model.predict_proba(input_data)[0]

            risk_level = "High Risk" if prediction == 1 else "Low Risk"

            st.success(f"Prediction: {'Chronic Kidney Disease Detected' if prediction == 1 else 'No Kidney Disease'}")
            st.info(f"Probability of CKD: {prediction_proba[1]*100:.2f}%")
            st.warning(f"Risk Level: {risk_level}")

        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
