from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Optional, List
import uvicorn
import numpy as np
import threading
import requests
import pickle
import io
import os
import signal
import psutil
import time

class Server:
    def __init__(self, node, protocol="https"):
        # Initialize the FastAPI app and the queue
        self.app = FastAPI()
        self.node = node
        self.task_manager = self.node.task_manager
        self.protocol = protocol

        self.registered_nodes = []
        self.registration_condition = threading.Condition()

        self.master_url = None

        # Register routes
        self.add_routes()

        self.server_thread = None

    def world_size(self):
        n_nodes = len(self.registered_nodes) + 1

        return n_nodes

    def wait_for_nodes(self, target_world_size):
        print(f"Waiting on nodes... 1/{target_world_size}")
        while self.world_size() < target_world_size:
            with self.registration_condition:
                self.registration_condition.wait()
            print(f"{self.world_size()}/{target_world_size}")
    
    def wait_for_master(self, master_url):
        print("Waiting on master node...")
        self.master_url = master_url

        while True:
            response = requests.get(f"{master_url}/")
            if response.status_code == 200:
                return True

            time.sleep(1)

    def assign_shards(self, submodels, shard_graph):
        if len(submodels) > self.world_size():
            raise Exception("Too many submodels for world size.")
        if len(submodels) == 0:
            raise Exception("No submodels included.")
        
        self.master_url = self.node.get_url(self.protocol)
        
        model_stack = shard_graph.get_parent_nodes()
        model_set = set(model_stack)
        node_stack = [x for x in self.registered_nodes]

        self.task_manager.clear_model()
        self.task_manager.clear_children()

        own_model_id = None
        assigned_models = {}
        assigned_nodes = {}

        # assign each submodel to each node, including the master node
        while len(model_stack) > 0:
            model_id = model_stack.pop()
            model_set.remove(model_id)
            new_models = [x.index for x in shard_graph.nodes[model_id].connection_set if x not in model_set]
            model_stack += new_models
            model_set = model_set.union(set(new_models))
            
            if own_model_id is None:
                own_model_id = model_id
                self.task_manager.model = submodels[model_id]
            else:
                next_node = node_stack.pop()
                assigned_models[next_node] = model_id
                assigned_nodes[model_id] = next_node
                
                url = f"{next_node}/model_assign"
                with open(submodels[model_id].path, "rb") as file:
                    print(f"sending {submodels[model_id].path}")
                    requests.post(url, files={"file": (submodels[model_id].path, file, "application/octet-stream")})
        
        # update each node with their children, including the proper inputs
        for connection in shard_graph.nodes[own_model_id].connection_set:
            outputs = shard_graph.nodes[own_model_id].connection_labels[connection]

            self.task_manager.children.append(assigned_nodes[connection.index])
            self.task_manager.child_output_mappings[assigned_nodes[connection.index]] += outputs
            
        for node_ip, model_id in assigned_models.items():
            for connection in shard_graph.nodes[model_id].connection_set:
                outputs = shard_graph.nodes[model_id].connection_labels[connection]
                url = f"{next_node}/child_assign"
                requests.post(url, {"outputs": outputs, "host_ip": node_ip})

    def register_self(self, master_url):
        url = f"{master_url}/register_node"
        self.master_url = master_url
        requests.post(url, json={"ip": self.node.get_url(self.protocol)})

    def add_routes(self):
        @self.app.get("/")
        async def root():
            return {"message": "Hello Peri!"}

        @self.app.post("/model_assign")
        async def assign_model(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
            if not file.filename.endswith(".onnx"):
                print(f"filename: {file.filename}")
                raise HTTPException(status_code=400, detail="Only .onnx files are allowed.")

            try:
                with open(self.task_manager.model.path, "wb") as f:
                    f.write(await file.read())
                return {"message": f"ONNX model saved as {file.filename}"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Request failed (Internal Server Error)")

        @self.app.post("/child_assign")
        async def assign_child(request: Request):
            data = await request.json()

            outputs = data.get("outputs")
            host_ip = data.get("host_ip")

            self.task_manager.children.append(host_ip)
            self.task_manager.child_output_mappings[host_ip] += outputs

        @self.app.post("/register_node")
        async def register_node(request: Request):
            data = await request.json()
            
            node_address = data.get("ip")

            if not node_address:
                raise HTTPException(status_code=400, detail=f"Field 'ip' missing")

            self.registered_nodes.append(node_address)
            with self.registration_condition:
                self.registration_condition.notify()

            return {"message": "Node registered successfully"}

        @self.app.post("/submit_input/{infer_id}")
        async def submit_input(background_tasks: BackgroundTasks, infer_id: int, file: UploadFile = File(...)):
            try:
                contents = await file.read()

                flo = io.BytesIO(contents)

                with np.load(flo) as np_contents:
                    data = {k: np_contents[k] for k in np_contents}

                self.task_manager.submit_input(data, infer_id)
                
                background_tasks.add_task(self.task_manager.check_for_completion)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Request failed (Internal Server Error)")

        @self.app.get("/inputs")
        async def get_inputs():
            return [{
                        "name": x.name,
                        "shape": x.shape,
                        "type": x.type
                    } for x in self.task_manager.model.get_inputs()]

        @self.app.get("/output/{infer_id}")
        async def output(infer_id: int):
            if infer_id not in self.task_manager.outputs:
                if infer_id not in self.task_manager.input_requests:
                    raise HTTPException(status_code=404, detail=f"Inference id {infer_id} not found")
                else:
                    raise HTTPException(status_code=202, detail=f"Inference id {infer_id} is still processing...")

            try:
                buffer, output_filename = self.task_manager.get_buffer(infer_id)
                return StreamingResponse(
                        buffer,
                        media_type="application/octet-stream",
                        headers={
                            "Content-Disposition": f"attachment; filename={output_filename}"
                            }
                        )
            except Exception as e:
                print(f"found exception... {str(e)}")
                raise HTTPException(status_code=500, detail=f"Request failed (Internal Server Error)")

    def run(self, host, port):
        # Start the FastAPI server
        def into_server():
            uvicorn.run(self.app, host=host, port=port)
        self.server_thread = threading.Thread(target = into_server, daemon=True)
        self.server_thread.start()

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, sig=None, frame=None):
        if self.server_thread and self.server_thread.is_alive():
            parent_pid = os.getpid()
            parent = psutil.Process(parent_pid)
            for child in parent.children(recursive=True):
                child.kill()

    def pause_signal(self):
        signal.pause()
