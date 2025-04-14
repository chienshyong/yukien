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
from scipy.stats import percentileofscore

# Load model and scaler
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

# Compute metrics
df = pd.read_csv('dataset.csv')

# Drop non-feature columns
df_numeric = df.drop(columns=['image_path'])
feature_names = [col for col in df_numeric.columns if col != 'artist_label']

# Recompute summary statistics using only feature columns
summary = df_numeric[feature_names].agg(['mean', 'std'])
summary_by_artist = df_numeric.groupby('artist_label')[feature_names].agg(['mean', 'std'])

label_map = {
    0: "mery", 1: "torino", 2: "yukien", 3: "ningen_mame",
    4: "cocoballking", 5: "fuzichoco", 6: "maccha", 7: "ilya_kuvshinov"
}

# Feature names
feature_names = [
    "edge_density", "laplacian_variance", "shannon_entropy", "hs_colorfulness", "color_spread",
    "color_entropy", "temp_mean", "temp_stddev", "gray_mean", "gray_stddev",
    "lbp_0", "lbp_1", "lbp_2", "lbp_3", "lbp_4", "lbp_5", "lbp_6", "lbp_7", "lbp_8", "lbp_9"
]

st.set_page_config(layout="wide")
st.title("🎨 Artist Style Classifier")

uploaded_file = st.file_uploader("Upload an artwork", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Preprocess image
    img_resized = img.resize((128, 128))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    # Extract and transform features
    raw_features = feature_extraction(tmp_path)
    user_features = scaler.transform([raw_features])

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

    # Display user features in two rows
    st.markdown("### 🔍 Extracted & Scaled Features")

    # Calculate estimated percentiles
    percentiles = []
    for i, feature_name in enumerate(feature_names):
        feature_values = df_numeric[feature_name].dropna().values
        percentile = percentileofscore(feature_values, raw_features[i], kind='mean')
        percentiles.append(percentile)

    # Extract averages
    artist_avg = summary_by_artist.xs('mean', axis=1, level=1).loc[top_idx].values
    global_avg = summary.loc['mean'].values

    # Build full comparison DataFrame
    feature_df = pd.DataFrame(
        [raw_features, artist_avg, global_avg, percentiles],
        columns=feature_names,
        index=["score", "artist_avg", "global_avg", "global_percentile"]
    )
    first_row = feature_df.iloc[:, :10]
    second_row = feature_df.iloc[:, 10:]

    st.dataframe(first_row)
    st.dataframe(second_row)    

# to run:
# streamlit run ./app.py