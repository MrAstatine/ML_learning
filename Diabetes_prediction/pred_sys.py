import numpy as np
import pickle

loaded_model = pickle.load(open("Diabetes_prediction/trained_model.sav", "rb"))

input_data = (10, 122, 78, 31, 0, 27.6, 0.512, 45)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshaping the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
# standardizing the input data
# std_data=scaler.transform(input_data_reshaped)
# print(std_data)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
