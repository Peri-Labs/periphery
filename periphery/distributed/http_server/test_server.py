import requests
import numpy as np

# Function to create a sample .npz file
def create_sample_npz_file(filename):
    # Create some sample NumPy arrays
    data = {
        "array1": np.random.rand(3, 3),
        "array2": np.arange(10)
    }
    
    # Save the arrays to an .npz file
    np.savez(filename, **data)

# Define the server URL and infer_id
server_url = "http://127.0.0.1:8000/submit_input/5"
npz_file = "sample_data.npz"

# Create a sample .npz file
create_sample_npz_file(npz_file)

# Test the API endpoint by uploading the .npz file
with open(npz_file, "rb") as file:
    response = requests.post(server_url, files={"file": file})

# Print the response from the server
print("Status Code:", response.status_code)
print("Response JSON:", response.json())

