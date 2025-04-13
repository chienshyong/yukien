import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
import joblib
from feature_extraction import feature_extraction
import tempfile
import plotly.graph_objects as go

# Load model and scaler
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

label_map = {
    0: "mery", 1: "torino", 2: "yukien", 3: "ningen_mame",
    4: "cocoballking", 5: "fuzichoco", 6: "maccha", 7: "ilya_kuvshinov"
}

st.set_page_config(layout="wide")  # Use wide layout
st.title("🎨 Artist Style Classifier")

uploaded_file = st.file_uploader("Upload an artwork", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    # Temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Preprocess image
    img_resized = img.resize((128, 128))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    user_features = feature_extraction(tmp_path)
    user_features = scaler.transform([user_features])

    pred = model.predict([img_array, user_features])[0]
    top_idx = np.argmax(pred)

    df = pd.DataFrame({
        "Artist": [label_map[i] for i in range(len(pred))],
        "Confidence": pred
    }).sort_values(by="Confidence", ascending=True)

    # Layout: image | prediction + chart
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(img, caption="Uploaded Artwork", use_container_width=True)

    with col2:
        st.markdown(f"### 🧑‍🎨 Predicted Artist: **{label_map[top_idx]}**")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Confidence"],
            y=df["Artist"],
            orientation='h',
            marker=dict(color='mediumvioletred'),
            text=[f"{v:.2f}" for v in df["Confidence"]],
            textposition='outside'
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 1], title="Confidence"),
            yaxis=dict(title="Artist"),
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor='white',
            transition={'duration': 500},
        )
        st.plotly_chart(fig, use_container_width=True)


# to run:
# streamlit run ./app.py