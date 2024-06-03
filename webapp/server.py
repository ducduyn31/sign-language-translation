import contextlib
from functools import lru_cache
import logging
from multiprocessing import Manager, Process, Queue
import pickle
import platform
from queue import Empty
import subprocess
import time
from fastapi import FastAPI
from contextlib import asynccontextmanager
import onnxruntime as ort

class Status:
    OK = "OK"
    ERROR = "ERROR"
    PENDING = "PENDING"
    FINISH_STREAMING = "FINISH_STREAMING"

def lifespan(app: FastAPI):
    app.request_queue = Queue()
    manager = Manager()
    app.request_buffer = manager.dict()

    process_list = []
    for worker_id, device in enumerate(app.devices * app.workers_per_device):
        if len(device) == 1:
            device = device[0]
        process = Process(
            target=inference_worker,
            kwargs={
                "api_server": app,
                "device": device,
                "stream": app.stream,
                "request_queue": app.request_queue,
                "request_buffer": app.request_buffer,
                "max_batch_size": app.max_batch_size,
            },
            daemon=True,
        )
        process.start()
        process_list.append((process, worker_id))
        
    yield
    logging.info("Shutting down workers")
    for process, worker_id in process_list:
        logging.info(f"Terminating worker {worker_id}")
        process.terminate()
        logging.info(f"Worker {worker_id} terminated")
        
def run_streaming_loop(api_server: 'SignTranslationServer', request_queue: Queue, request_buffer):
    while True:
        try:
            uid = request_queue.get(timeout=1.0)
            try:
                x_enc, pipe_s = request_buffer.pop(uid)
            except KeyError:
                continue
        except (Empty, ValueError):
            continue
        
        try:
            x = api_server.decode_request(x_enc)
            y_gen = api_server.predict(x)
            y_enc_gen = api_server.encode_response(y_gen)
            
            for y_enc in y_enc_gen:
                with contextlib.suppress(BrokenPipeError):
                    pipe_s.send((y_enc, Status.OK))
            pipe_s.send((None, Status.FINISH_STREAMING))
        except Exception as e:
            logging.exception(e)
            pipe_s.send((pickle.dumps(e), Status.ERROR))

def collate_requests(api_server: 'SignTranslationServer', request_queue: Queue, request_buffer):
    batch_uids = []
    entered_at = time.time()
    timeout = 1.0
    batch_size = 64
    
    # Collect requests until the batch size is reached or the timeout is exceeded
    while (timeout - (time.time() - entered_at) > 0) and len(batch_uids) < batch_size:
        try:
            uid = request_queue.get(timeout=0.001)
            batch_uids.append(uid)
        except (Empty, ValueError):
            continue
    
    # Group requests
    batch = []
    for uid in batch_uids:
        try:
            x_enc, pipe_s = request_buffer.pop(uid)
        except KeyError:
            continue
        x = api_server.decode_request(x_enc)
        batch.append((uid, x, pipe_s))
    return batch

def run_batched_loop(api_server: 'SignTranslationServer', request_queue: Queue, request_buffer):
    while True:
        batches = collate_requests(api_server, request_queue, request_buffer)
        if not batches:
            continue
        
        inputs, pipes = zip(*batches)
        
        try:
            x = api_server.batch(inputs)
            y = api_server.predict(x)
            outputs = api_server.unbatch(y)
            
            for y, pipe_s in zip(outputs, pipes):
                y_enc = api_server.encode_request(y)
                
                with contextlib.suppress(BrokenPipeError):
                    pipe_s.send((y_enc, Status.OK))
        except Exception as e:
            logging.exception(e)
            err_pkl = pickle.dumps(e)
            
            for pipe_s in pipes:
                pipe_s.send((err_pkl, Status.ERROR))

def run_sing_loop(api_server: 'SignTranslationServer', request_queue: Queue, request_buffer):
    while True:
        try:
            uid = request_queue.get(timeout=1.0)
            try:
                x_enc, pipe_s = request_buffer.pop(uid)
            except KeyError:
                continue
        except (Empty, ValueError):
            continue
        
        try:
            x = api_server.decode_request(x_enc)
            y = api_server.predict(x)
            y_enc = api_server.encode_request(y)
            with contextlib.suppress(BrokenPipeError):
                pipe_s.send((y_enc, Status.OK))
        except Exception as e:
            logging.exception(e)
            pipe_s.send((pickle.dumps(e), Status.ERROR))
        
def inference_worker(api_server: 'SignTranslationServer', device: str, stream: bool, request_queue: Queue, request_buffer, max_batch_size=64):
    api_server.setup_server(device)
    
    if stream:
        run_streaming_loop(api_server, request_queue, request_buffer)
        return
    
    if max_batch_size == 1:
        run_sing_loop(api_server, request_queue, request_buffer)
        return
    
    run_batched_loop(api_server, request_queue, request_buffer)
    

class SignTranslationServer(FastAPI):
    def __init__(self, devices=1, *args, **kwargs):
        super().__init__(lifespan=lifespan, *args, **kwargs)
        self.max_batch_size = 1
        self.workers_per_device = 1
        self.stream = False
        self.devices = self._select_devices(devices)
        
    @lru_cache(maxsize=1)
    def _select_devices(self, devices):
        import torch
        
        device_list = devices if isinstance(devices, list) else range(devices)
        
        def check_cuda_with_nvidia_smi():
            try:
                return b"GPU" in subprocess.check_output(["nvidia-smi", "-L"])
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
    
        if check_cuda_with_nvidia_smi():
            return [f"cuda:{i}" for i in device_list]
        if torch.backends.mps.is_available() and platform.processor() in ("arm", "arm64"):
            return [f"mps:{i}" for i in device_list]
        return ["cpu"]
        
        
    def setup_server(self, device: str):
        self.yolo = ort.InferenceSession("./yolov9c.onnx")
        self.rtmpose = ort.InferenceSession("./rtmpose_x.onnx")
        self.sign_detector = ort.InferenceSession("./sign_recognition_model.onnx")
        
    def predict(self, video):
        return {"output": "Hello World!"}
    
    def decode_request(self, request):
        return request
    
    def encode_request(self, request):
        return request
    
    def batch(self, requests):
        return requests
    
    def unbatch(self, responses):
        return responses
    
    def run(self, port=8000, log_level="info", **kwargs):
        import uvicorn
        uvicorn.run(host="0.0.0.0", port=port, app=self, log_level=log_level, **kwargs)