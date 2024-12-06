import requests
import time

import numpy as np

# Function to create a sample .npz file
def create_npz_file(filename, input_dict):
    input_dict = {k: np.array(v) for k, v in input_dict.items()}
    np.savez(filename, **input_dict)

def send_for_inference(domain, port, infer_id, input_dict):
    npz_filename = f"np_data/input_{infer_id}.npz"
    url = f"http://{domain}:{port}/submit_input/{infer_id}"

    create_npz_file(npz_filename, input_dict)

    with open(npz_filename, "rb") as file:
        response = requests.post(url, files={"file": file})

    return response

def get_inference_result(domain, port, infer_id):
    while True:
        url = f"http://{domain}:{port}/output/{infer_id}"

        response = requests.get(url)

        if response.status_code == 200:
            npz_file_path = f"np_data/output_{infer_id}.npz"

            with open(npz_file_path, "wb") as file:
                file.write(response.content)

            return np.load(npz_file_path)
        elif response.status_code != 202:
            return None
        
        time.sleep(1)
