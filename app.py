from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Route to the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve user inputs from the form
        item_weight = float(request.form['item_weight'])
        item_fat_content = request.form['item_fat_content']
        item_visibility = float(request.form['item_visibility'])
        item_type = request.form['item_type']
        item_mrp = float(request.form['item_mrp'])
        outlet_identifier = request.form['outlet_identifier']
        outlet_establishment_year = int(request.form['outlet_establishment_year'])
        outlet_size = request.form['outlet_size']
        outlet_location_type = request.form['outlet_location_type']
        outlet_type = request.form['outlet_type']
        
        # Create a DataFrame with the input values
        input_data = pd.DataFrame([[item_weight, item_fat_content, item_visibility, item_type,
                                    item_mrp, outlet_identifier, outlet_establishment_year, outlet_size,
                                    outlet_location_type, outlet_type]],
                                  columns=['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
                                           'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year', 
                                           'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])

        # Preprocess the input data (if necessary, e.g., encoding categorical variables)
        # Example: encoding categorical variables
        # input_data = preprocess(input_data)
        
        # Make the prediction
        prediction = model.predict(input_data)

        # Render the result page with prediction
        return render_template('index.html', prediction_text=f'Predicted Sales: {prediction[0]:.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction_text="Error in prediction. Please check your input.")

if __name__ == "__main__":
    app.run(debug=True)
