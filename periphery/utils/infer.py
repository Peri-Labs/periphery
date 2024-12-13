import requests
import time
import io

import numpy as np

# Function to create a sample .npz file
def create_npz_buffer(input_dict):
    buffer = io.BytesIO()
    input_dict = {k: np.array(v) for k, v in input_dict.items()}
    np.savez(buffer, **input_dict)
    buffer.seek(0)
    
    return buffer

def send_for_inference(domain, port, infer_id, input_dict):
    npz_filename = f"input_{infer_id}.npz"
    url = f"http://{domain}:{port}/submit_input/{infer_id}"

    buffer = create_npz_buffer(input_dict)

    response = requests.post(url, files={"file": (npz_filename, buffer, "application/octet-stream")})

    return response

def get_inference_result(domain, port, infer_id):
    while True:
        url = f"http://{domain}:{port}/final_output/{infer_id}"

        response = requests.get(url)

        if response.status_code == 200:
            npz_file_path = f"np_data/output_{infer_id}.npz"

            flo = io.BytesIO(response.content)

            return np.load(flo)
        elif response.status_code != 202:
            return None
        
        time.sleep(1)
