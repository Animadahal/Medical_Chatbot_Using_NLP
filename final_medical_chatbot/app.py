# app.py
import streamlit as st
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ctransformers import AutoModelForCausalLM

# ----------------- Config -----------------
BERT_FOLDER = "bert_classifier"
LABEL_MAP_FILE = "label_map_top50.json"
MISTRAL_GGUF = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # must exist in same folder
FOLLOWUP_COUNT = 3

# ----------------- Load models (cached) -----------------
@st.cache_resource
def load_bert():
    tok = AutoTokenizer.from_pretrained(BERT_FOLDER)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_FOLDER)
    model.eval()
    return tok, model

@st.cache_resource
def load_mistral():
    llm = AutoModelForCausalLM.from_pretrained(
        str(MISTRAL_GGUF),
        model_type="mistral",
        gpu_layers=0  # CPU-only
    )
    return llm

bert_tokenizer, bert_model = load_bert()
llm = load_mistral()

# ----------------- Label map -----------------
try:
    with open(LABEL_MAP_FILE, "r") as f:
        label_map = json.load(f)
except Exception:
    label_map = None

def get_label_name(index):
    """Return human readable label for predicted index."""
    if label_map is None:
        return f"Label {index}"
    if isinstance(label_map, dict):
        if str(index) in label_map:
            return label_map[str(index)]
        if index in label_map:
            return label_map[index]
    if isinstance(label_map, list):
        if 0 <= index < len(label_map):
            return label_map[index]
    return f"Label {index}"

# ----------------- Red Flag Detection -----------------
RED_FLAG_SYMPTOMS = {
    "chest pain": "⚠️ URGENT: Chest pain can indicate a heart attack or other serious cardiac condition.",
    "difficulty breathing": "⚠️ URGENT: Severe breathing difficulty requires immediate medical attention.",
    "shortness of breath": "⚠️ URGENT: Severe breathing difficulty requires immediate medical attention.",
    "can't breathe": "⚠️ URGENT: Severe breathing difficulty requires immediate medical attention.",
    "severe headache": "⚠️ URGENT: Sudden severe headache could indicate a stroke or brain emergency.",
    "sudden headache": "⚠️ URGENT: Sudden severe headache could indicate a stroke or brain emergency.",
    "confusion": "⚠️ URGENT: Confusion or altered mental state needs immediate evaluation.",
    "disoriented": "⚠️ URGENT: Confusion or altered mental state needs immediate evaluation.",
    "unconscious": "⚠️ EMERGENCY: Loss of consciousness requires calling emergency services immediately.",
    "passed out": "⚠️ EMERGENCY: Loss of consciousness requires calling emergency services immediately.",
    "seizure": "⚠️ EMERGENCY: Active seizures require immediate emergency care.",
    "convulsion": "⚠️ EMERGENCY: Active seizures require immediate emergency care.",
    "heavy bleeding": "⚠️ URGENT: Uncontrolled bleeding needs immediate medical attention.",
    "bleeding heavily": "⚠️ URGENT: Uncontrolled bleeding needs immediate medical attention.",
    "severe bleeding": "⚠️ URGENT: Uncontrolled bleeding needs immediate medical attention.",
    "severe abdominal pain": "⚠️ URGENT: Severe abdominal pain could indicate internal emergency.",
    "severe stomach pain": "⚠️ URGENT: Severe abdominal pain could indicate internal emergency.",
    "suicidal": "⚠️ CRISIS: Please contact emergency services or a crisis helpline immediately.",
    "want to die": "⚠️ CRISIS: Please contact emergency services or a crisis helpline immediately.",
    "kill myself": "⚠️ CRISIS: Please contact emergency services or a crisis helpline immediately.",
    "stroke": "⚠️ EMERGENCY: Signs of stroke require immediate emergency care.",
    "paralysis": "⚠️ URGENT: Sudden paralysis or weakness needs immediate medical evaluation.",
    "can't move": "⚠️ URGENT: Sudden inability to move needs immediate medical evaluation.",
    "slurred speech": "⚠️ URGENT: Slurred speech could indicate a stroke.",
    "coughing blood": "⚠️ URGENT: Coughing up blood requires immediate medical attention.",
    "vomiting blood": "⚠️ URGENT: Vomiting blood requires immediate medical attention.",
    "severe burn": "⚠️ URGENT: Severe burns require immediate medical attention.",
}

def check_red_flags(text):
    """Check if user input contains emergency symptoms."""
    text_lower = text.lower()
    alerts = []
    detected_symptoms = []
    
    for symptom, alert in RED_FLAG_SYMPTOMS.items():
        if symptom in text_lower:
            if alert not in alerts:  # Avoid duplicate alerts
                alerts.append(alert)
                detected_symptoms.append(symptom)
    
    if alerts:
        alert_message = "\n\n".join(alerts)
        alert_message += "\n\n🚨 **Please seek immediate medical attention or call emergency services:**"
        alert_message += "\n- **Dial: 102 (Ambulance)**"
        alert_message += "\n- **Go to the nearest hospital emergency room**"
        alert_message += "\n- **Do not delay seeking professional medical care**"
        return alert_message
    return None

# ----------------- Helpers -----------------
def get_followups_text(symptom_input, n=FOLLOWUP_COUNT):
    prompt = f"""You are a careful medical assistant.
The user said: "{symptom_input}".
Generate {n} short follow-up questions (each under 8 words)."""
    out = llm(prompt, max_new_tokens=200, temperature=0.2)
    lines = [ln.strip(" -•1234567890.") for ln in out.splitlines() if ln.strip()]
    if len(lines) < n:
        lines += ["How long have you had these symptoms?",
                  "Do you have a fever?",
                  "Are you taking any medicines?"]
    return lines[:n]

def predict_disease_and_confidence(text):
    enc = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = bert_model(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy().squeeze()
    if probs.ndim == 0:
        probs = np.array([probs.item()])
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    name = get_label_name(top_idx)
    return top_idx, name, confidence, probs

def generate_explanation(symptoms_with_answers, diagnosis_name, confidence):
    prompt = f"""
User reported: {symptoms_with_answers}
Predicted disease: {diagnosis_name} (confidence: {confidence:.2f})

Write a short, empathetic explanation (3–5 sentences) why this fits, 
include 3 next steps (self-care, when to see doctor). 
End with: "⚠ This is not medical advice."
"""
    out = llm(prompt, max_new_tokens=300, temperature=0.3)
    return out

def generate_free_response(context, user_message):
    prompt = f"""
You are a knowledgeable, safe medical assistant.
Continue this ongoing conversation naturally and accurately.

Chat so far:
{context}

User's new message: "{user_message}"

Reply clearly and empathetically. Avoid repeating previous info. 
End with a gentle disclaimer if needed.
"""
    out = llm(prompt, max_new_tokens=300, temperature=0.4)
    return out.strip()

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="💬 Medical Chatbot", layout="wide")
st.title("💬 Medical Chatbot")
st.caption("Educational demo only — not medical advice.")

# ----------------- Session state -----------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "context" not in st.session_state:
    st.session_state.context = ""
if "phase" not in st.session_state:
    st.session_state.phase = "symptom"  # symptom → followup → result → chat
if "followups" not in st.session_state:
    st.session_state.followups = []
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "emergency_mode" not in st.session_state:
    st.session_state.emergency_mode = False

# Sidebar
with st.sidebar:
    st.markdown("### Medical Chatbot")
    st.markdown("---")
    st.markdown("**Emergency Numbers:**")
    st.markdown("- 🚑 Ambulance: **102**")
    st.markdown("- 🚓 Police: **103**")
    st.markdown("- 🔥 Fire: **101**")
    st.markdown("---")
    if st.button("🔄 Reset chat"):
        st.session_state.clear()
        st.rerun()
    st.markdown("---")
    st.caption("⚠️ This chatbot is for educational purposes only and does not replace professional medical advice.")

# ----------------- Chat Display -----------------
if not st.session_state.chat:
    welcome_message = (
        "**Welcome to the Medical Chatbot!**\n\n"
        "I'm here to help you understand your symptoms.\n"
        "Please describe how you're feeling or list your symptoms below, and I'll ask a few follow-up questions to assist you.\n\n"
    )
    st.session_state.chat.append(("bot", welcome_message))

for role, text in st.session_state.chat:
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

# ----------------- Input Area -----------------
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.chat.append(("user", user_input))
    st.session_state.context += f"User: {user_input}\n"

    # CHECK FOR RED FLAGS FIRST (in all phases) 
    red_flag_alert = check_red_flags(user_input)
    if red_flag_alert:
        st.session_state.chat.append(("bot", red_flag_alert))
        st.session_state.context += f"Assistant: [Emergency alert displayed]\n"
        st.session_state.emergency_mode = True
        st.rerun()

    # PHASE 1: Initial Symptom 
    if st.session_state.phase == "symptom":
        with st.spinner("Analyzing symptoms..."):
            followups = get_followups_text(user_input)
        st.session_state.followups = followups
        qtext = "\n".join([f"{i+1}. {q}" for i, q in enumerate(followups)])
        reply = f"Thanks! I just need a few clarifications:\n\n{qtext}\n\n*Please answer these questions to help me better understand your condition.*"
        st.session_state.chat.append(("bot", reply))
        st.session_state.context += f"Assistant: {reply}\n"
        st.session_state.phase = "followup"
        st.rerun()
        
    # PHASE 2: Follow-up Answers 
    elif st.session_state.phase == "followup":
        combined = st.session_state.context + f"\nUser follow-up: {user_input}"
        with st.spinner("Predicting condition..."):
            top_idx, name, conf, probs = predict_disease_and_confidence(combined)
            explanation = generate_explanation(combined, name, conf)

        st.session_state.prediction = (top_idx, name, conf)

        if conf < 0.5:
            # If model not confident enough
            fallback_message = (
                f"😕 **I'm not confident enough to determine a specific condition based on your input.**\n\n"
                f"Your symptoms don't seem to clearly match any disease in my knowledge base. "
                f"However, please monitor your symptoms carefully and consult a qualified healthcare professional if they persist or worsen.\n\n"
                f"  **General Care Tips:**\n"
                f"- Stay hydrated and get enough rest\n"
                f"- Eat a balanced, nutritious diet\n"
                f"- Avoid stress and maintain good hygiene\n"
                f"- Keep track of your symptoms\n"
                f"- Visit a doctor if symptoms continue or get worse\n\n"
                f"⚠️ **This is not a medical diagnosis. Please consult a healthcare professional for proper evaluation.**"
            )
            st.session_state.chat.append(("bot", fallback_message))
        else:
            # Normal confident response
            result_message = f"**Predicted Condition:** *{name}*\n\n**Confidence Score:** {conf:.2f}\n\n{explanation}"
            st.session_state.chat.append(("bot", result_message))

        st.session_state.context += f"Assistant: {explanation}\n"
        st.session_state.phase = "chat"  # switch to free conversation

    # PHASE 3: Continuous Chat
    else:
        with st.spinner("Thinking..."):
            response = generate_free_response(st.session_state.context, user_input)
        st.session_state.chat.append(("bot", response))
        st.session_state.context += f"Assistant: {response}\n"

    st.rerun()