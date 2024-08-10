import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np


bigmart_loaded_encoder = pickle.load(open("big_mart_encoder.sav", "rb"))
bigmart_loaded_regresser = pickle.load(open("big_mart_regresser.sav", "rb"))

breast_cancer_loaded_model = pickle.load(open("breast_cancer_model.sav", "rb"))
breast_cancer_loaded_std = pickle.load(open("breast_cancer_standard_scaler.sav", "rb"))

calories_burnt_loaded_model = pickle.load(open("Calories_Burnt_Prediction_model.sav", 'rb'))
calories_burnt_loaded_std = pickle.load(open("Calories_Burnt_Prediction_std.sav", 'rb'))

loan_approv_loaded_model = pickle.load(open("loan_prediction.sav", "rb"))

with st.sidebar:
    predictions = [ 'Big Mart Sales','Breast cancer','Calories Burnt','Loan Approval']
    selected = option_menu("Multiple Prediction", predictions, default_index=0)


if(selected == 'Big Mart Sales'):
    def prediction(input_data):
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = bigmart_loaded_regresser.predict(input_data_reshaped)
        return f'Predicted Sales: {prediction[0]}'

    def main():
        st.title("Big Mart Sales Prediction Model")

        # Organize input fields into two columns
        col1, col2 = st.columns(2)

        with col1:
            Item_Identifier = st.text_input("Item Identifier (numeric)", "1")
            Item_Weight = st.text_input("Item Weight (numeric)", "9.3")
            Item_Fat_Content = st.text_input("Item Fat Content (0: Low Fat, 1: Regular, 2: Non-Edible)", "0")
            Item_Visibility = st.text_input("Item Visibility (numeric)", "0.016")
            Item_Type = st.text_input("Item Type (numeric)", "0")
            Item_MRP = st.text_input("Item MRP (numeric)", "249.8092")

        with col2:
            Outlet_Identifier = st.text_input("Outlet Identifier (numeric)", "1")
            Outlet_Establishment_Year = st.text_input("Outlet Establishment Year (numeric)", "1985")
            Outlet_Size = st.text_input("Outlet Size (0: Small, 1: Medium, 2: High)", "1")
            Outlet_Location_Type = st.text_input("Outlet Location Type (0: Tier 1, 1: Tier 2, 2: Tier 3)", "0")
            Outlet_Type = st.text_input("Outlet Type (0: Grocery Store, 1: Supermarket Type1, 2: Supermarket Type2, 3: Supermarket Type3)", "1")

        # Predict when button is pressed
        if st.button("Predict"):
            try:
                input_data = [
                    int(Item_Identifier),
                    float(Item_Weight),
                    int(Item_Fat_Content),
                    float(Item_Visibility),
                    int(Item_Type),
                    float(Item_MRP),
                    int(Outlet_Identifier),
                    int(Outlet_Establishment_Year),
                    int(Outlet_Size),
                    int(Outlet_Location_Type),
                    int(Outlet_Type)
                ]

                result = prediction(input_data)
                st.success(result)
            
            except ValueError:
                st.error("Please enter valid numerical values for all inputs.")
    if __name__ == '__main__':
        main()

    
if(selected == 'Breast cancer'):
    def prediction(input):
        input_as_numpy_array = np.asarray(input)
        input_reshaped = input_as_numpy_array.reshape(1,-1)
        std_input = breast_cancer_loaded_std.transform(input_reshaped)
        prediction = breast_cancer_loaded_model.predict(std_input)
        print(prediction)
        if prediction[0] == 0:
            return "The Breast Cancer is Malignant"
        else:
            return "The Breast Cancer is Benign"
    def main():
        st.title("Breast Cancer Prediction")

        # Organize input fields into three columns
        col1, col2, col3 = st.columns(3)

        with col1:
            mean_radius = st.text_input("Mean Radius", "14.0")
            mean_texture = st.text_input("Mean Texture", "19.0")
            mean_perimeter = st.text_input("Mean Perimeter", "90.0")
            mean_area = st.text_input("Mean Area", "650.0")
            mean_smoothness = st.text_input("Mean Smoothness", "0.1")
            mean_compactness = st.text_input("Mean Compactness", "0.05")
            mean_concavity = st.text_input("Mean Concavity", "0.1")
            mean_concave_points = st.text_input("Mean Concave Points", "0.05")
            mean_symmetry = st.text_input("Mean Symmetry", "0.2")
            mean_fractal_dimension = st.text_input("Mean Fractal Dimension", "0.05")

        with col2:
            radius_error = st.text_input("Radius Error", "0.5")
            texture_error = st.text_input("Texture Error", "1.0")
            perimeter_error = st.text_input("Perimeter Error", "2.0")
            area_error = st.text_input("Area Error", "10.0")
            smoothness_error = st.text_input("Smoothness Error", "0.01")
            compactness_error = st.text_input("Compactness Error", "0.01")
            concavity_error = st.text_input("Concavity Error", "0.02")
            concave_points_error = st.text_input("Concave Points Error", "0.01")
            symmetry_error = st.text_input("Symmetry Error", "0.02")
            fractal_dimension_error = st.text_input("Fractal Dimension Error", "0.01")

        with col3:
            worst_radius = st.text_input("Worst Radius", "25.0")
            worst_texture = st.text_input("Worst Texture", "30.0")
            worst_perimeter = st.text_input("Worst Perimeter", "150.0")
            worst_area = st.text_input("Worst Area", "1000.0")
            worst_smoothness = st.text_input("Worst Smoothness", "0.2")
            worst_compactness = st.text_input("Worst Compactness", "0.1")
            worst_concavity = st.text_input("Worst Concavity", "0.2")
            worst_concave_points = st.text_input("Worst Concave Points", "0.1")
            worst_symmetry = st.text_input("Worst Symmetry", "0.3")
            worst_fractal_dimension = st.text_input("Worst Fractal Dimension", "0.1")

        diagnosis = ''

        if st.button("Predict Cancer Type"):
            try:
                # Collect and convert inputs into a list of floats
                input_data = [
                    float(mean_radius),
                    float(mean_texture),
                    float(mean_perimeter),
                    float(mean_area),
                    float(mean_smoothness),
                    float(mean_compactness),
                    float(mean_concavity),
                    float(mean_concave_points),
                    float(mean_symmetry),
                    float(mean_fractal_dimension),
                    float(radius_error),
                    float(texture_error),
                    float(perimeter_error),
                    float(area_error),
                    float(smoothness_error),
                    float(compactness_error),
                    float(concavity_error),
                    float(concave_points_error),
                    float(symmetry_error),
                    float(fractal_dimension_error),
                    float(worst_radius),
                    float(worst_texture),
                    float(worst_perimeter),
                    float(worst_area),
                    float(worst_smoothness),
                    float(worst_compactness),
                    float(worst_concavity),
                    float(worst_concave_points),
                    float(worst_symmetry),
                    float(worst_fractal_dimension)
                ]
                
                diagnosis = prediction(input_data)
                st.success(f"Prediction: {diagnosis}")

            except ValueError:
                st.error("Please enter valid input values.")


    if __name__ == '__main__':
        main()

    
if(selected == 'Calories Burnt'):

    def prediction(input_data):
        input_as_numpy_array = np.asarray(input_data, dtype=float)  # Convert input to numpy array
        input_standard = calories_burnt_loaded_std.transform(input_as_numpy_array.reshape(1, -1))  # Standardize the data
        pred = calories_burnt_loaded_model.predict(input_standard)  # Predict
        return pred[0]  # Return the prediction

    def main():
        st.title("Calories Burnt Prediction")

        col1, col2 = st.columns(2)

        # Collect inputs from the user
        with col1:
            Gender = st.text_input("Gender (0: Male, 1: Female)", "0")
            Age = st.text_input("Age", "25")
            Height = st.text_input("Height (in cm)", "170")
            Weight = st.text_input("Weight (in kg)", "70")
            
        with col2:
            Duration = st.text_input("Duration (in mins)", "30")
            Heart_Rate = st.text_input("Heart Rate", "120")
            Body_Temp = st.text_input("Body Temperature (in °C)", "36.6")

        diagnosis = ''

        # Predict calories burnt when the button is pressed
        if st.button("Calories Burnt Prediction"):
            try:
                input_data = [
                    int(Gender),
                    float(Age),
                    float(Height),
                    float(Weight),
                    float(Duration),
                    float(Heart_Rate),
                    float(Body_Temp)
                ]
                diagnosis = prediction(input_data)
                st.success(f"Calories Burnt: {diagnosis:.2f}")
            except ValueError:
                st.error("Please enter valid input values.")

    if __name__ == '__main__':
        main()

 
    
if(selected == 'Loan Approval'):
    def prediction(input):
        input_as_numpy_array = np.asarray(input)
        input_reshaped = input_as_numpy_array.reshape(1,-1)
        prediction = loan_approv_loaded_model.predict(input_reshaped)
        if (prediction[0] == 0):
            return 'The loan is not approved'
        else:
            return 'The loan is approved'
        
    def main():
        st.title("Loan Prediction Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            Gender = st.text_input("Gender (0: Male, 1: Female)", "0")
            Married = st.text_input("Married (0: No, 1: Yes)", "1")
            Dependents = st.text_input("Dependents (0, 1, 2, or 3+)", "1")
            Education = st.text_input("Education (0: Graduate, 1: Not Graduate)", "1")
            Self_Employed = st.text_input("Self_Employed (0: No, 1: Yes)", "0")
            ApplicantIncome = st.text_input("ApplicantIncome", "4583")
        with col2 : 
            CoapplicantIncome = st.text_input("CoapplicantIncome", "1500.0")
            LoanAmount = st.text_input("LoanAmount", "128.0")
            Loan_Amount_Term = st.text_input("Loan_Amount_Term", "360.0")
            Credit_History = st.text_input("Credit_History (0 or 1)", "1.0")
            Property_Area = st.text_input("Property_Area (Rural : 0,Semiurban : 1,Urban : 2)", "0")

        if st.button("Predict"):
            try:
                input_data = [
                    float(Gender),
                    float(Married),
                    float(Dependents),
                    float(Education),
                    float(Self_Employed),
                    float(ApplicantIncome),
                    float(CoapplicantIncome),
                    float(LoanAmount),
                    float(Loan_Amount_Term),
                    float(Credit_History),
                    float(Property_Area)
                ]
                result = prediction(input_data)
                st.success(result)
            
            except ValueError:
                st.error("Please enter valid numerical values for all inputs.")

    if __name__ == '__main__':
        main()