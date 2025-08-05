import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
import io

# Load dataset
df_full = sns.load_dataset("titanic") # Load the full dataset first

# Initialize LabelEncoders on ALL possible categories from the *original* data
# This ensures they learn all labels that might appear in the Gradio inputs
le_sex = LabelEncoder()
le_sex.fit(df_full['sex'].dropna().unique()) # Fit on all unique non-NA sex values

le_embarked = LabelEncoder()
le_embarked.fit(df_full['embarked'].dropna().unique()) # Fit on all unique non-NA embarked values

# Now, filter and clean the dataframe for model training
df = df_full[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].dropna()

# Apply the transformations using the already fitted LabelEncoders
df['sex'] = le_sex.transform(df['sex'])
df['embarked'] = le_embarked.transform(df['embarked'])

# Split and train
X = df.drop(columns=["survived"])
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# --- Functions to generate plots and save them to a temporary file ---

def plot_heatmap():
    plt.figure(figsize=(6, 4)) 
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu") 
    plt.tight_layout()  #improves layout

    buf = io.BytesIO()  #required to keep in memory
    plt.savefig(buf, format="png") 
    plt.close() # clears memory
    buf.seek(0) #required so PIL reads from beginning

    return Image.open(buf) 

# --- Prediction function ---
def predict(pclass, sex, age, sibsp, parch, fare, embarked):
    try:
        # Use the fitted LabelEncoders to transform input strings to numerical values
        # These LabelEncoders were fitted on all *possible* categories, so they should not
        # encounter unseen labels from the Gradio dropdowns.
        sex_val = le_sex.transform([sex])[0]
        embarked_val = le_embarked.transform([embarked])[0]

        # Features must be in the same order as trained: pclass, sex, age, sibsp, parch, fare, embarked
        row = [[pclass, sex_val, age, sibsp, parch, fare, embarked_val]]
        
        # Make prediction
        prediction = model.predict(row)[0]
        
        return "🛟 Survived" if prediction == 1 else "💀 Did not survive"
    except ValueError as e:
        if "not in list" in str(e):
            return "⚠️ Invalid 'Embarked Port' selected. Please choose from C, Q, or S."
        return f"❌ Prediction error: {e}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

# # --- Generate image files ONCE before the Gradio interface starts ---

with gr.Blocks() as demo:
    gr.Markdown("# 🚢 Titanic Survival Prediction")
    gr.Markdown(f"**Model Accuracy:** {acc:.2%}")

    with gr.Tab("📊 Data Exploration"):
        gr.Markdown("#### 🔹 Survival Rate by Gender")
        gr.BarPlot(df.assign(sex=le_sex.inverse_transform(df["sex"]), color="blue"), x="sex", y="survived", y_aggregate="mean")

        gr.Markdown("#### 🔹 Survival Rate by Class")
        gr.BarPlot(
    df.assign(pclass=df["pclass"].map({1: "1st", 2: "2nd", 3: "3rd"})),
    x="pclass",
    y="survived",
    y_aggregate="mean",
    color="pclass")

        gr.Markdown("#### 🔹 Feature Correlation Heatmap")
        gr.Image(value=plot_heatmap, type="pil", label="Feature Correlation")
      

    with gr.Tab("🔮 Predict Survival"):
        gr.Markdown("### Fill Passenger Info")

        with gr.Row():
            pclass = gr.Dropdown([1, 2, 3], value=3, label="Passenger Class (1=1st, 3=3rd)")
            sex = gr.Radio(["male", "female"], value="female", label="Sex") 
            age = gr.Slider(1, 80, value=29, step=1, label="Age")

        with gr.Row():
            sibsp = gr.Slider(0, 5, value=0, step=1, label="Siblings/Spouses Aboard")
            parch = gr.Slider(0, 5, value=0, step=1, label="Parents/Children Aboard")
            fare = gr.Slider(0, 500, value=32, step=1, label="Fare")

        # Ensure these match the categories LabelEncoder was fitted on
        embarked = gr.Radio(
            ["C", "Q", "S"], value="S",
            label="Embarked Port (C = Cherbourg, Q = Queenstown, S = Southampton)"
        )
        predict_btn = gr.Button("Predict Survival")
        prediction_output = gr.Markdown()

        predict_btn.click(
            fn=predict,
            inputs=[pclass, sex, age, sibsp, parch, fare, embarked],
            outputs=prediction_output
        )

demo.launch()