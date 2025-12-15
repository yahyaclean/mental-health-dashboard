import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import time
import google.generativeai as genai

# ===============================================
# 0️⃣ Streamlit Page Configuration
# ===============================================
st.set_page_config(
    page_title="🧠 Mental Health Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================
# 1️⃣ Load & Preprocess Data
# ===============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Mental Health Dataset.zip")
    except FileNotFoundError:
        try:
            df = pd.read_csv("Mental Health Dataset.csv")
        except FileNotFoundError:
            st.error("⚠️ Error: File not found! Please ensure 'Mental Health Dataset.zip' or '.csv' is uploaded.")
            return pd.DataFrame()

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Date"] = df["Timestamp"].dt.date
        df["Hour"] = df["Timestamp"].dt.hour
        df["DayOfWeek"] = df["Timestamp"].dt.day_name()
    
    return df

df = load_data()

if df.empty:
    st.stop()

target = "treatment"

# ===============================================
# 🔍 SIDEBAR FILTERS & API KEY
# ===============================================
st.sidebar.header("🔑 AI Configuration")
# 1. Try to find the key in Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

st.sidebar.divider()
st.sidebar.header("🔍 Filter Data")

def get_options(col):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

sel_gender = st.sidebar.multiselect("Select Gender", get_options("Gender"), default=get_options("Gender"))
sel_stress = st.sidebar.multiselect("Growing Stress", get_options("Growing_Stress"), default=get_options("Growing_Stress"))
sel_mood = st.sidebar.multiselect("Mood Swings", get_options("Mood_Swings"), default=get_options("Mood_Swings"))
sel_days = st.sidebar.multiselect("Days Indoors", get_options("Days_Indoors"), default=get_options("Days_Indoors"))

filtered_df = df.copy()
if sel_gender: filtered_df = filtered_df[filtered_df["Gender"].isin(sel_gender)]
if sel_stress: filtered_df = filtered_df[filtered_df["Growing_Stress"].isin(sel_stress)]
if sel_mood: filtered_df = filtered_df[filtered_df["Mood_Swings"].isin(sel_mood)]
if sel_days: filtered_df = filtered_df[filtered_df["Days_Indoors"].isin(sel_days)]

# ===============================================
# 2️⃣ Feature Lists
# ===============================================
rel_cols = [
    "Gender", "Country", "Occupation", "self_employed", "family_history",
    "Days_Indoors", "Growing_Stress", "Changes_Habits", "Mental_Health_History",
    "Mood_Swings", "Coping_Struggles", "Work_Interest", "Social_Weakness",
    "mental_health_interview", "care_options"
]
geo_col = "Country"

# ===============================================
# 3️⃣ Main Layout & Tabs
# ===============================================
st.title("🧠 Mental Health Dashboard & Hybrid AI")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Stats", "⏰ Time", "😷 Behavioral", "🌍 Geographic", "🔍 Deep Dive", "🤖 Interactive AI Clinic"
])

# ===============================================
# TABS 1-5 (EXACTLY AS FIRST CODE - FULL VISUALIZATIONS)
# ===============================================

# --- TAB 1: Stats & Correlation ---
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Target Distribution")
        st.plotly_chart(px.pie(filtered_df, names=target, title='Target Distribution', color_discrete_sequence=px.colors.sequential.RdBu), use_container_width=True)
    with col2:
        st.subheader("Feature Breakdown")
        sel_col = st.selectbox("Select Feature:", rel_cols, index=0)
        temp = filtered_df.copy()
        if temp[sel_col].nunique() > 25:
            temp = temp[temp[sel_col].isin(temp[sel_col].value_counts().nlargest(20).index)]
        temp = temp.groupby([sel_col, target]).size().reset_index(name="count")
        temp["percent"] = temp.groupby(sel_col)["count"].transform(lambda x: 100 * x / x.sum())
        st.plotly_chart(px.bar(temp, x=sel_col, y="percent", color=target, barmode="stack", title=f"Treatment % by {sel_col}"), use_container_width=True)
    
    st.divider()
    st.subheader("Statistical Correlation Heatmap")
    df_enc = filtered_df.copy().drop(columns=['Timestamp'], errors='ignore')
    le = LabelEncoder()
    for col in df_enc.columns:
        if df_enc[col].dtype == 'object': df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    st.plotly_chart(px.imshow(df_enc.corr(), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title='Correlation Heatmap'), use_container_width=True)

# --- TAB 2: Time ---
with tab2:
    if "Timestamp" in filtered_df.columns:
        daily = filtered_df.groupby(["Date", target]).size().reset_index(name="count")
        st.plotly_chart(px.line(daily, x="Date", y="count", color=target, title="Trends Over Time"), use_container_width=True)
        heat = filtered_df[filtered_df[target] == "Yes"].pivot_table(index="DayOfWeek", columns="Hour", values=target, aggfunc="count", fill_value=0)
        if not heat.empty: st.plotly_chart(px.imshow(heat, title="Treatment (Yes) Heatmap by Day & Hour", aspect="auto", color_continuous_scale="Viridis"), use_container_width=True)

# --- TAB 3: Behavioral ---
with tab3:
    st.subheader("Multivariate Behavioral Analysis")
    b_df = filtered_df.copy()
    b_df["treatment_score"] = b_df[target].map({"Yes": 1, "No": 0}).fillna(0)
    st.plotly_chart(px.parallel_categories(b_df, dimensions=["Gender", "family_history", "care_options", target], color="treatment_score", color_continuous_scale=px.colors.sequential.Inferno), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.sunburst(filtered_df, path=["Country", "family_history", target], title="Country → Family History → Treatment"), use_container_width=True)
    with c2:
        cols_r = ["care_options", "family_history", "Gender", "self_employed", "mental_health_interview"]
        temp_r = filtered_df.copy()
        enc = LabelEncoder()
        for c in cols_r:
            if c in temp_r.columns: temp_r[c] = enc.fit_transform(temp_r[c].astype(str))
        if all(c in temp_r.columns for c in cols_r):
            r_data = temp_r.groupby(target)[cols_r].mean().reset_index()
            st.plotly_chart(px.line_polar(r_data.melt(id_vars=target), r="value", theta="variable", color=target, line_close=True, title="Radar Chart: Feature Averages"), use_container_width=True)

# --- TAB 4: Geographic ---
with tab4:
    if geo_col in filtered_df.columns:
        g_data = filtered_df.groupby([geo_col, target]).size().reset_index(name="count")
        if not g_data.empty:
            piv = g_data.pivot(index=geo_col, columns=target, values="count").fillna(0)
            piv["Rate"] = piv["Yes"] / (piv["Yes"] + piv["No"]) if "Yes" in piv.columns and "No" in piv.columns else 0
            st.plotly_chart(px.choropleth(piv.reset_index(), locations=geo_col, locationmode="country names", color="Rate", title="Global Treatment Rates"), use_container_width=True)

# --- TAB 5: Deep Dive ---
with tab5:
    st.header("🔍 Deep Dive: Specific Relations")
    psych_pairs = [('Growing_Stress','Mood_Swings'), ('Changes_Habits','Coping_Struggles'), ('Mental_Health_History','family_history'), ('Social_Weakness','Work_Interest')]
    c1, c2 = st.columns(2)
    for i, (x, y) in enumerate(psych_pairs):
        if x in filtered_df.columns and y in filtered_df.columns:
            fig = px.histogram(filtered_df, x=x, color=y, barmode='group', title=f"{x} vs {y}", text_auto=True)
            fig.update_traces(textposition='outside')
            (c1 if i%2==0 else c2).plotly_chart(fig, use_container_width=True)
            
    st.divider()
    st.subheader("2. Work & Environment Relations")
    work_pairs = [('mental_health_interview','care_options'), ('Occupation','care_options'), ('self_employed','mental_health_interview')]
    for x, y in work_pairs:
        if x in filtered_df.columns and y in filtered_df.columns:
            st.plotly_chart(px.histogram(filtered_df, x=x, color=y, barmode='group', title=f"{x} vs {y}", text_auto=True), use_container_width=True)

# ===============================================
# 🤖 TAB 6: FAST STEP-BY-STEP CHATBOT (FROM SECOND CODE)
# ===============================================
with tab6:
    st.header("🤖 Interactive AI Clinic")
    st.markdown("Use this tool to analyze your status step-by-step and then **chat** with the AI Assistant.")

    # --- Session State Management ---
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_inputs" not in st.session_state:
        st.session_state.user_inputs = {}
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = {}
    if "current_step_idx" not in st.session_state:
        st.session_state.current_step_idx = 0

    # 1. Load XGBoost Model
    model_path = "mental_health_xgboost.pkl"
    xgb_model = None
    if os.path.exists(model_path):
        try:
            xgb_model = joblib.load(model_path)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
    else:
        st.warning(f"⚠️ Model '{model_path}' not found. Please upload it.")
    # 🔍 DEBUG: check model class order
   # 🔍 DEBUG: check model class order
    if xgb_model:
        st.sidebar.subheader("🧪 Model Debug")
        st.sidebar.write("Model classes:", xgb_model.classes_)



    # ===============================================
    # STEP 1: WIZARD DATA ENTRY (Divided Logic)
    # ===============================================
    if not st.session_state.analysis_done:
        if xgb_model:
            
            # Define Question Groups
            question_groups = {
                "Step 1: Basics": ["Gender", "Country", "Occupation", "self_employed"],
                "Step 2: Medical History": ["family_history", "Mental_Health_History", "care_options", "mental_health_interview"],
                "Step 3: Lifestyle": ["Days_Indoors", "Growing_Stress", "Changes_Habits", "Mood_Swings"],
                "Step 4: Social & Coping": ["Coping_Struggles", "Work_Interest", "Social_Weakness"]
            }
            group_names = list(question_groups.keys())
            
            # Get current step
            current_group_name = group_names[st.session_state.current_step_idx]
            current_cols = question_groups[current_group_name]
            
            # Progress Bar
            progress = (st.session_state.current_step_idx + 1) / len(group_names)
            st.progress(progress, text=f"Assessment Progress: {current_group_name}")
            
            with st.container():
                with st.form(key=f"form_step_{st.session_state.current_step_idx}"):
                    st.subheader(current_group_name)
                    c1, c2 = st.columns(2)
                    step_responses = {}
                    
                    for i, col in enumerate(current_cols):
                        target_col = c1 if i % 2 == 0 else c2
                        if df[col].dtype == 'object':
                            options = sorted(df[col].dropna().unique().tolist())
                            step_responses[col] = target_col.selectbox(f"{col}", options)
                        else:
                            min_v, max_v = float(df[col].min()), float(df[col].max())
                            step_responses[col] = target_col.number_input(f"{col}", min_value=min_v, max_value=max_v, value=float(df[col].mean()))

                    # Determine Button Text
                    is_last_step = st.session_state.current_step_idx == len(group_names) - 1
                    btn_label = "🚀 Run Diagnosis & Chat" if is_last_step else "Next Step ➡"
                    submit_step = st.form_submit_button(label=btn_label)
            
            if submit_step:
                # Save answers
                st.session_state.user_inputs.update(step_responses)
                
                if not is_last_step:
                    st.session_state.current_step_idx += 1
                    st.rerun()
                else:
                    # --- FINAL STEP: RUN XGBOOST LOGIC ---
                    try:
                        input_data = st.session_state.user_inputs
                        input_df = pd.DataFrame([input_data])
                        
                        # Preprocessing
                        input_df['Gender'] = input_df['Gender'].map({'Male': 0, 'Female': 1})
                        input_df['self_employed'] = input_df['self_employed'].map({'No': 0, 'Yes': 1}).fillna(0)
                        input_df['family_history'] = input_df['family_history'].map({'No': 0, 'Yes': 1})
                        input_df['Coping_Struggles'] = input_df['Coping_Struggles'].map({'No': 0, 'Yes': 1})
                        input_df['Mood_Swings'] = input_df['Mood_Swings'].map({'Low': 0, 'Medium': 1, 'High': 2})
                        input_df['Days_Indoors'] = input_df['Days_Indoors'].map({'Go out Every day': 0, '1-14 days': 1, '15-30 days': 2, '31-60 days': 3, 'More than 2 months': 4})
                        
                        for col in ['Growing_Stress', 'Changes_Habits', 'Mental_Health_History', 'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options']:
                            le = LabelEncoder().fit(df[col].astype(str))
                            val = str(input_data[col])
                            input_df[col] = le.transform([val]) if val in le.classes_ else 0

                        input_df['Stress_Score'] = input_df[['Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Coping_Struggles', 'Mood_Swings']].mean(axis=1)
                        input_df['Social_Function_Score'] = input_df['Work_Interest'] - input_df['Social_Weakness']
                        input_df['SelfEmployment_Risk'] = input_df['self_employed'] * (1 - input_df['care_options'])
                        input_df['Family_Support_Impact'] = input_df['family_history'] * input_df['Coping_Struggles']

                        now = pd.Timestamp.now()
                        input_df['Year'], input_df['Month'], input_df['Day'], input_df['Hour'] = now.year, now.month, now.day, now.hour
                        input_df['Is_Winter'] = input_df['Month'].isin([12, 1, 2]).astype(int)
                        input_df['Is_MidYear'] = input_df['Month'].between(5, 8).astype(int)
                        input_df['Is_Night'] = ((input_df['Hour'] >= 20) | (input_df['Hour'] <= 6)).astype(int)

                        all_occ = sorted(df['Occupation'].unique())
                        user_occ = input_data['Occupation']
                        for occ in all_occ: input_df[f"Occupation_{occ}"] = 1 if occ == user_occ else 0
                        if 'Occupation' in input_df.columns: input_df.drop(columns=['Occupation'], inplace=True)
                        
                        if 'Country' in input_df.columns:
                            le_c = LabelEncoder().fit(df['Country'].astype(str))
                            val_c = str(input_data['Country'])
                            input_df['Country'] = le_c.transform([val_c]) if val_c in le_c.classes_ else 0

                        # Prediction
                        prediction = xgb_model.predict(input_df)[0]
                        prob = xgb_model.predict_proba(input_df)[0] if hasattr(xgb_model, "predict_proba") else [0,0]
                        
                        is_risk = (prediction == 1 or prediction == "Yes")
                        confidence = prob[1] if is_risk else prob[0]
                        
                        st.session_state.prediction_result = {
                            "text": "Treatment Likely Needed" if is_risk else "No Immediate Treatment Needed",
                            "color": "red" if is_risk else "green",
                            "confidence": f"{confidence*100:.1f}%",
                            "is_risk": is_risk
                        }
                        
                        # --- INITIALIZE CHAT ---
                        risk_desc = "High Risk" if is_risk else "Low Risk"
                        intro_prompt = f"""
                        System Context: You are a compassionate mental health AI.
                        User Data: Stress={input_data['Growing_Stress']}, Indoors={input_data['Days_Indoors']}, Mood={input_data['Mood_Swings']}.
                        Diagnosis: The XGBoost model predicts **{risk_desc}** with {confidence*100:.1f}% confidence.
                        Your Task: Greet user warmly based on result. Validate feelings if high risk, encourage if low. Ask open question. Keep short.
                        """
                        
                        if api_key:
                            try:
                                genai.configure(api_key=api_key)
                                model_name = "gemini-pro"
                                for m in genai.list_models():
                                    if 'generateContent' in m.supported_generation_methods:
                                        if 'flash' in m.name: model_name = m.name; break
                                model = genai.GenerativeModel(model_name)
                                response = model.generate_content(intro_prompt)
                                initial_msg = response.text
                            except:
                                initial_msg = "Hello. I have analyzed your profile. How are you feeling right now?"
                        else:
                            initial_msg = "Analysis Complete. Please enter API Key to chat. (Standard Mode: How can I help?)"

                        st.session_state.messages = [{"role": "assistant", "content": initial_msg}]
                        st.session_state.analysis_done = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Analysis Error: {e}")

    # ===============================================
    # STEP 2: RESULTS & CHAT (AFTER WIZARD)
    # ===============================================
    else:
        res = st.session_state.prediction_result
        
        with st.container():
            col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
            with col_r2:
                if res["is_risk"]:
                    st.error(f"🚨 **Analysis Result:** {res['text']}")
                else:
                    st.success(f"✅ **Analysis Result:** {res['text']}")
                st.metric("Model Confidence", res["confidence"])
            
            if st.button("🔄 Start New Assessment"):
                st.session_state.analysis_done = False
                st.session_state.current_step_idx = 0
                st.session_state.messages = []
                st.rerun()
        
        st.divider()
        
        st.subheader("💬 Chat with AI Assistant")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    model_name = "gemini-pro"
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            if 'flash' in m.name: model_name = m.name; break
                    
                    chat_model = genai.GenerativeModel(model_name)
                    
                    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
                    input_summary = str(st.session_state.user_inputs)
                    risk_info = str(st.session_state.prediction_result)
                    
                    full_prompt = f"""
                    Context: You are a helpful mental health assistant.
                    User Profile: {input_summary}
                    Risk Analysis: {risk_info}
                    History: {history_text}
                    Last Message: "{prompt}"
                    Reply empathetically and helpfully.
                    """
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = chat_model.generate_content(full_prompt)
                            st.markdown(response.text)
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                            
                except Exception as e:
                    st.error(f"API Error: {e}")
            else:
                st.warning("Please enter Gemini API Key in the sidebar to enable chat.")
