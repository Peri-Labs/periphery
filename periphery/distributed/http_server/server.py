from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from queue import Queue
from typing import Optional
import uvicorn
import numpy as np

import pickle
import io

class QueueItem(BaseModel):
    key: str
    value: str

class Server:
    def __init__(self, node):
        # Initialize the FastAPI app and the queue
        self.app = FastAPI()
        self.node = node
        self.task_manager = self.node.task_manager

        # Register routes
        self.add_routes()

    def add_routes(self):
        @self.app.get("/")
        async def root():
            return {"message": "Hello Peri!"}

        @self.app.post("/enqueue")
        async def enqueue_item(item: QueueItem, background_tasks: BackgroundTasks):
            # Add an item to the queue
            self.queue.put(item.dict())
            background_tasks.add_task(self.process_queue)
            return {"status": "Item added to queue", "item": item.dict()}

        @self.app.get("/queue_size")
        async def queue_size():
            # Return the current size of the queue
            return {"queue_size": self.queue.qsize()}

        @self.app.post("/model_assign")
        async def assign_model(background_tasks: BackgroundTasks, model_file: UploadFile = File(...)):
            if not file.filename.endswith(".onnx"):
                raise HTTPException(status_code=400, detail="Only .onnx files are allowed.")

            try:
                with open(self.task_manager.model.path, "wb") as f:
                    f.write(await model_file.read())
                return {"message": f"ONNX model saved as {file.filename}"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Request failed (Internal Server Error)")



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
                    }for x in self.task_manager.model.get_outputs()]

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

    def process_queue(self):
        print("processing...")

    def run(self, host: str = "127.0.0.1", port: int = 8000):
        # Start the FastAPI server
        uvicorn.run(self.app, host=host, port=port)

# Example usage
if __name__ == "__main__":
    server = Server(task_manager.TaskManager(None))
    server.run()
