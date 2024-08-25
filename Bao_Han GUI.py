import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

class LaptopPricePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Laptop Price Prediction')
        self.data = pd.read_csv('Laptop_price.csv')

        # Drop 'Brand' and 'Weight' columns
        self.data = self.data.drop(columns=['Brand', 'Weight'])

        # Convert any remaining categorical columns to numeric using LabelEncoder
        self.label_encoders = {}
        for column in self.data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])
            self.label_encoders[column] = le

        self.X = self.data.drop('Price', axis=1).values
        self.y = self.data['Price'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.model = XGBRegressor()
        self.model.fit(self.X_train, self.y_train)

        self.create_widgets()

    def create_widgets(self):
        self.comboboxes = []
        for i, column in enumerate(self.data.columns[:-1]):
            label = tk.Label(self.master, text=column + ': ')
            label.grid(row=i, column=0)

            # Create a Combobox for each feature
            combobox = ttk.Combobox(self.master)
            combobox['values'] = sorted(map(str, self.data[column].unique()))  # Convert to string for display
            combobox.grid(row=i, column=1)

            self.comboboxes.append(combobox)

        predict_button = tk.Button(self.master, text='Predict Price', command=self.predict_price)
        predict_button.grid(row=len(self.data.columns[:-1]), columnspan=2)

    def predict_price(self):
        inputs = []
        for combobox, column in zip(self.comboboxes, self.data.columns[:-1]):
            value = combobox.get()
            # Handle numeric inputs
            try:
                inputs.append(float(value))
            except ValueError:
                # Handle categorical inputs using label encoders
                le = self.label_encoders.get(column)
                if le:
                    inputs.append(le.transform([value])[0])

        price = self.model.predict([inputs])
        messagebox.showinfo('Predicted Price', f'The predicted laptop price is ${price[0]:.2f}')

if __name__ == '__main__':
    root = tk.Tk()
    app = LaptopPricePredictionApp(root)
    root.mainloop()
