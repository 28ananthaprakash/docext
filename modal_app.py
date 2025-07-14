from __future__ import annotations

import gradio as gr
import modal
from fastapi import FastAPI
from gradio.routes import mount_gradio_app

from docext.app.app import gradio_app
from docext.core.vllm import VLLMServer  # <-- Import VLLMServer

app = modal.App("docext-app")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .workdir("/root/docext")
    .add_local_python_source("docext")
)


@app.function(image=image, gpu="t4", max_containers=1)
@modal.asgi_app()
def gradio_web():
    # Set your desired Gradio parameters here
    model_name = "hosted_vllm/Qwen/Qwen2.5-VL-3B-Instruct"  # Use the default model from client.py
    gradio_port = 7860
    max_img_size = 1024
    concurrency_limit = 1
    share = False
    vllm_server_host = "localhost"
    vllm_server_port = 8000
    max_gen_tokens = 512
    vllm_start_timeout = 300
    dtype = "float16"  # Updated to match your local run

    # Match local CLI args
    max_model_len = 16000
    gpu_memory_utilization = 0.9
    max_num_imgs = 1

    # Start vLLM server in background
    vllm_server = VLLMServer(
        model_name=model_name,
        host=vllm_server_host,
        port=vllm_server_port,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_imgs=max_num_imgs,
        vllm_start_timeout=vllm_start_timeout,
        dtype=dtype,
    )
    vllm_server.run_in_background()

    # Create the Gradio Blocks app (do NOT call .launch())
    demo = gradio_app(
        model_name,
        gradio_port,
        max_img_size,
        concurrency_limit,
        share,
        vllm_server_host,
        vllm_server_port,
        max_gen_tokens,
        launch=False,  # Ensure .launch() is not called
    )
    web_app = FastAPI()
    return mount_gradio_app(app=web_app, blocks=demo, path="/")


@app.local_entrypoint()
def main():
    print(
        "To access the Gradio app, deploy with `modal deploy modal_app.py` and use the public URL provided by Modal."
    )
