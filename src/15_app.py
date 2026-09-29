import pandas as pd
import joblib
import gradio as gr


# =========================================================
# 1. Load model
# =========================================================

model = joblib.load(
    "models/house_price_model.pkl"
)


# =========================================================
# 2. Load dataset
# =========================================================

df = pd.read_csv("data/train.csv")

X_reference = df.drop(
    "SalePrice",
    axis=1
)


# =========================================================
# 3. Prediction function
# =========================================================

def predict_price(
    overall_qual,
    gr_liv_area,
    garage_cars,
    year_built,
    full_bath
):

    # Start with a valid row containing all 80 features
    house = X_reference.iloc[[0]].copy()

    # Replace user inputs
    house["OverallQual"] = overall_qual
    house["GrLivArea"] = gr_liv_area
    house["GarageCars"] = garage_cars
    house["YearBuilt"] = year_built
    house["FullBath"] = full_bath

    # Prediction
    prediction = model.predict(house)[0]

    return f"${prediction:,.2f}"


# =========================================================
# 4. Gradio Interface
# =========================================================

with gr.Blocks(title="House Price Prediction") as app:

    gr.Markdown(
        """
        # 🏠 House Price Prediction

        Enter the main characteristics of the house
        and the machine learning model will estimate
        its sale price.
        """
    )

    with gr.Row():

        overall_qual = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Overall Quality"
        )

        gr_liv_area = gr.Number(
            value=1500,
            label="Living Area (sq ft)"
        )

    with gr.Row():

        garage_cars = gr.Slider(
            minimum=0,
            maximum=4,
            value=2,
            step=1,
            label="Garage Capacity"
        )

        year_built = gr.Number(
            value=2000,
            label="Year Built"
        )

    full_bath = gr.Slider(
        minimum=0,
        maximum=4,
        value=2,
        step=1,
        label="Full Bathrooms"
    )

    predict_button = gr.Button(
        "Predict Price"
    )

    result = gr.Textbox(
        label="Estimated Sale Price"
    )

    predict_button.click(
        fn=predict_price,
        inputs=[
            overall_qual,
            gr_liv_area,
            garage_cars,
            year_built,
            full_bath
        ],
        outputs=result
    )


# =========================================================
# 5. Launch
# =========================================================

if __name__ == "__main__":
    app.launch()