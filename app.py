import joblib
import pandas as pd

try:
    import gradio as gr
except ImportError:  # pragma: no cover - fallback for test environments
    class _DummyComponent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _DummyInterface:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def launch(self):
            return None

    class _DummyGradioModule:
        Slider = _DummyComponent
        Textbox = _DummyComponent
        Interface = _DummyInterface

    gr = _DummyGradioModule()

# Load your trained model
model = joblib.load("model.joblib")


# Define the prediction function
def predict_churn(
    frequency,
    monetary,
    total_items,
    avg_basket_size,
    n_unique_products,
    country,
    tenure_days,
    avg_days_between_purchases,
):
    # Create a DataFrame from the inputs
    df = pd.DataFrame(
        [{
            "frequency": frequency,
            "monetary": monetary,
            "total_items": total_items,
            "avg_basket_size": avg_basket_size,
            "n_unique_products": n_unique_products,
            "country": country,
            "tenure_days": tenure_days,
            "avg_days_between_purchases": avg_days_between_purchases,
        }]
    )

    # Get prediction probability
    proba = model.predict_proba(df)[0][1]

    # Return a nice readable result
    if proba >= 0.5:
        return f"⚠️ HIGH RISK: {proba:.2%} chance of churning."
    return f"✅ SAFE: {proba:.2%} chance of churning."


# Build the Gradio Web UI
inputs = [
    gr.Slider(1, 200, value=10, label="Frequency (Total Orders)"),
    gr.Slider(1, 50000, value=500, label="Monetary (Total Spent $)"),
    gr.Slider(1, 10000, value=50, label="Total Items Bought"),
    gr.Slider(1, 5000, value=50, label="Avg Basket Size ($)"),
    gr.Slider(1, 1000, value=10, label="Unique Products Bought"),
    gr.Textbox(value="United Kingdom", label="Country"),
    gr.Slider(1, 2000, value=300, label="Tenure (Days as Customer)"),
    gr.Slider(1, 500, value=30, label="Avg Days Between Purchases"),
]

demo = gr.Interface(
    fn=predict_churn,
    inputs=inputs,
    outputs="text",
    title="🛒 Online Retail Customer Churn Predictor",
    description="Enter customer transaction details to predict if they will churn.",
)

# Launch the app
demo.launch()
