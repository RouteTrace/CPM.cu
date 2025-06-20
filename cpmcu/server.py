#!/usr/bin/env python3
import os
import sys
import argparse
import asyncio
import json
import torch
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download

from .llm import LLM
from .llm_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
from .speculative import LLM_with_eagle
from .speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
from .api_models import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatMessage,
    ErrorResponse,
    HealthResponse
)

def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Convert dtype string to torch dtype
    if 'dtype' in config:
        if config['dtype'] == 'float16':
            config['dtype'] = torch.float16
        elif config['dtype'] == 'bfloat16':
            config['dtype'] = torch.bfloat16
    
    return config

def get_default_config():
    """Get default configuration from file"""
    default_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.json')
    
    if not os.path.exists(default_config_path):
        print(f"Error: Default config file not found at {default_config_path}")
        print("Please ensure the config/default_config.json file exists.")
        # Fallback to hardcoded config if file doesn't exist
        return {
            "test_minicpm4": True,
            "use_stream": True,
            "apply_eagle": True,
            "apply_quant": True,
            "apply_sparse": True,
            "apply_eagle_quant": True,
            "minicpm4_yarn": True,
            "frspec_vocab_size": 32768,
            "eagle_window_size": 1024,
            "eagle_num_iter": 2,
            "eagle_topk_per_iter": 10,
            "eagle_tree_size": 12,
            "apply_compress_lse": True,
            "sink_window_size": 1,
            "block_window_size": 8,
            "sparse_topk_k": 64,
            "sparse_switch": 1,
            "num_generate": 256,
            "chunk_length": 2048,
            "memory_limit": 0.9,
            "cuda_graph": True,
            "dtype": torch.float16,
            "use_terminators": True,
            "temperature": 0.0,
            "random_seed": None,
            "use_enter": False,
            "use_decode_enter": False
        }
    
    try:
        return load_config_from_file(default_config_path)
    except json.JSONDecodeError as e:
        print(f"Error parsing default config file {default_config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading default config file {default_config_path}: {e}")
        sys.exit(1)

def check_or_download_model(path):
    """Check if model exists locally, otherwise download from HuggingFace"""
    if os.path.exists(path):
        return path
    else:
        cache_dir = snapshot_download(path)
        return cache_dir

def get_model_paths(path_prefix, config):
    """Get model paths based on configuration"""
    if config['test_minicpm4']:
        if config['apply_eagle_quant']:
            eagle_repo_id = f"{path_prefix}/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu"
        else:
            eagle_repo_id = f"{path_prefix}/MiniCPM4-8B-Eagle-FRSpec"
    else:
        eagle_repo_id = f"{path_prefix}/EAGLE-LLaMA3-Instruct-8B"
    
    if not config['apply_quant']:
        if config['test_minicpm4']:
            base_repo_id = f"{path_prefix}/MiniCPM4-8B"
        else:
            base_repo_id = f"{path_prefix}/Meta-Llama-3-8B-Instruct"
    else:
        base_repo_id = f"{path_prefix}/MiniCPM4-8B-marlin-cpmcu"

    eagle_path = check_or_download_model(eagle_repo_id)
    base_path = check_or_download_model(base_repo_id)
    
    return eagle_path, base_path, eagle_repo_id, base_repo_id

def apply_minicpm4_yarn_config(llm, config):
    """Apply MiniCPM4 YARN configuration to model config"""
    yarn_factors = [
        0.9977997200264581, 1.014658295992452, 1.0349680404997148, 1.059429246056193,
        1.0888815016813513, 1.1243301355211495, 1.166977103606075, 1.2182568066927284,
        1.2798772354275727, 1.3538666751582975, 1.4426259039919596, 1.5489853358570191,
        1.6762658237220625, 1.8283407612492941, 2.0096956085876183, 2.225478927469756,
        2.481536379650452, 2.784415934557119, 3.1413289096347365, 3.560047844772632,
        4.048719380066383, 4.752651957515948, 5.590913044973868, 6.584005926629993,
        7.7532214876576155, 9.119754865903639, 10.704443927019176, 12.524994176518703,
        14.59739595363613, 16.93214476166354, 19.53823297353041, 22.417131025031697,
        25.568260840911098, 28.991144156566317, 32.68408069090375, 36.65174474170465,
        40.90396065611201, 45.4664008671033, 50.37147343433591, 55.6804490772103,
        61.470816952306556, 67.8622707390618, 75.00516023410414, 83.11898235973767,
        92.50044360202462, 103.57086856690864, 116.9492274587385, 118.16074567836519,
        119.18497548708795, 120.04810876261652, 120.77352815196981, 121.38182790207875,
        121.89094985353891, 122.31638758099915, 122.6714244963338, 122.9673822552567,
        123.21386397019609, 123.41898278254268, 123.58957065488238, 123.73136519024158,
        123.84917421274221, 123.94701903496814, 124.02825801299717, 124.09569231686116
    ]
    
    # Create or modify rope_scaling configuration
    if not hasattr(llm.config, 'rope_scaling') or llm.config.rope_scaling is None:
        llm.config.rope_scaling = {}
        
    llm.config.rope_scaling['rope_type'] = 'longrope'
    llm.config.rope_scaling['long_factor'] = yarn_factors
    llm.config.rope_scaling['short_factor'] = yarn_factors
    print("Forcing MiniCPM4 YARN rope_scaling parameters")

def create_model(eagle_path, base_path, config):
    """Create model instance based on configuration"""
    common_kwargs = {
        'dtype': config['dtype'],
        'chunk_length': config['chunk_length'],
        'cuda_graph': config['cuda_graph'],
        'apply_sparse': config['apply_sparse'],
        'sink_window_size': config['sink_window_size'],
        'block_window_size': config['block_window_size'],
        'sparse_topk_k': config['sparse_topk_k'],
        'sparse_switch': config['sparse_switch'],
        'apply_compress_lse': config['apply_compress_lse'],
        'memory_limit': config['memory_limit'],
        'use_enter': config['use_enter'],
        'use_decode_enter': config['use_decode_enter'],
        'temperature': config['temperature'],
        'random_seed': config['random_seed'],
    }
    
    eagle_kwargs = {
        'num_iter': config['eagle_num_iter'],
        'topk_per_iter': config['eagle_topk_per_iter'],
        'tree_size': config['eagle_tree_size'],
        'eagle_window_size': config['eagle_window_size'],
        'frspec_vocab_size': config['frspec_vocab_size'],
        'apply_eagle_quant': config['apply_eagle_quant'],
        'use_rope': config['test_minicpm4'],
        'use_input_norm': config['test_minicpm4'],
        'use_attn_norm': config['test_minicpm4']
    }
    
    if config['apply_quant']:
        if config['apply_eagle']:
            return W4A16GPTQMarlinLLM_with_eagle(eagle_path, base_path, **common_kwargs, **eagle_kwargs)
        else:
            return W4A16GPTQMarlinLLM(base_path, **common_kwargs)
    else:
        if config['apply_eagle']:
            return LLM_with_eagle(eagle_path, base_path, **common_kwargs, **eagle_kwargs)
        else:
            return LLM(base_path, **common_kwargs)

# Global model instance
model_instance: Optional[LLM] = None
model_config: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model on startup"""
    global model_instance, model_config
    
    print(f"Loading model with configuration:")
    print(f"Model path prefix: {model_config['model_path_prefix']}")
    print(f"Configuration: {model_config['config']}")
    
    try:
        config = model_config['config']
        path_prefix = model_config['model_path_prefix']
        
        # Disable sparse attention for non-MiniCPM4 models
        if not config['test_minicpm4']:
            print(f"test_minicpm4 is False, set apply_sparse to False")
            config['apply_sparse'] = False
        
        # Get model paths and create model
        eagle_path, base_path, eagle_repo_id, base_repo_id = get_model_paths(path_prefix, config)
        model_instance = create_model(eagle_path, base_path, config)
        
        # Initialize model storage
        model_instance.init_storage()
        
        # Apply MiniCPM4 YARN configuration if enabled
        if config['test_minicpm4'] and config['minicpm4_yarn']:
            apply_minicpm4_yarn_config(model_instance, config)
        
        # Load frequency speculative vocabulary if enabled
        if config['apply_eagle'] and config['frspec_vocab_size'] > 0:
            fr_path = f'{eagle_path}/freq_{config["frspec_vocab_size"]}.pt'
            if not os.path.exists(fr_path):
                cache_dir = snapshot_download(
                    eagle_repo_id,
                    ignore_patterns=["*.bin", "*.safetensors"],
                )
                fr_path = os.path.join(cache_dir, f'freq_{config["frspec_vocab_size"]}.pt')
            
            with open(fr_path, 'rb') as f:
                token_id_remap = torch.tensor(torch.load(f, weights_only=True), dtype=torch.int32, device="cpu")
            model_instance._load("token_id_remap", token_id_remap, cls="eagle")
        
        # Load model weights
        model_instance.load_from_hf()
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Cleanup
    print("Shutting down...")
    model_instance = None

# Create FastAPI app
app = FastAPI(
    title="CPM.cu OpenAI API Server",
    description="OpenAI API compatible server for CPM.cu models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_messages_to_prompt(messages: list, tokenizer) -> str:
    """Convert OpenAI messages format to a single prompt string using chat template"""
    
    # Convert OpenAI messages format to the format expected by chat template
    chat_messages = []
    
    for message in messages:
        chat_messages.append({
            "role": message.role,
            "content": message.content
        })
    
    # Use tokenizer's chat template to format the prompt
    try:
        prompt = tokenizer.apply_chat_template(
            chat_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt
    except Exception as e:
        # Fallback to simple formatting if chat template fails
        print(f"Warning: Chat template failed ({e}), falling back to simple formatting")
        prompt_parts = []
        
        for message in messages:
            role = message.role
            content = message.content
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)

def get_stop_tokens(stop_param: Optional[Union[str, list]] = None) -> list:
    """Convert stop parameter to token IDs"""
    if not stop_param:
        return []
    
    if isinstance(stop_param, str):
        stop_list = [stop_param]
    else:
        stop_list = stop_param
    
    # Convert to token IDs using tokenizer
    stop_token_ids = []
    for stop_str in stop_list:
        try:
            tokens = model_instance.tokenizer.encode(stop_str, add_special_tokens=False)
            stop_token_ids.extend(tokens)
        except:
            pass
    
    return stop_token_ids

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        model_loaded=model_instance is not None,
        memory_usage=f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else None
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI Chat Completions API endpoint"""
    global model_instance
    
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get tokenizer from model instance first
        tokenizer = getattr(model_instance, 'tokenizer', None)
        if hasattr(model_instance, 'base_model') and hasattr(model_instance.base_model, 'tokenizer'):
            tokenizer = model_instance.base_model.tokenizer
        
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="Tokenizer not available")
        
        # Format messages to prompt using chat template
        prompt = format_messages_to_prompt(request.messages, tokenizer)
        
        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(torch.int32).cuda()
        
        # Get stop token IDs and add EOS token if use_terminators is enabled
        stop_tokens = get_stop_tokens(request.stop)
        config = model_config['config']
        if config.get('use_terminators', True):
            if tokenizer.eos_token_id not in stop_tokens:
                stop_tokens.append(tokenizer.eos_token_id)
        
        # Set temperature for this request
        original_temp = getattr(model_instance, 'temperature', 0.0)
        model_instance.temperature = request.temperature or 0.0
        
        try:
            if request.stream:
                return StreamingResponse(
                    stream_chat_completion(
                        input_ids, 
                        request.max_tokens or 100,
                        stop_tokens,
                        request.model,
                        request,
                        tokenizer
                    ),
                    media_type="text/plain",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # Disable nginx buffering
                    }
                )
            else:
                return await generate_chat_completion(
                    input_ids,
                    request.max_tokens or 100, 
                    stop_tokens,
                    request.model,
                    request,
                    tokenizer
                )
        finally:
            # Restore original temperature
            model_instance.temperature = original_temp
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

async def generate_chat_completion(
    input_ids: torch.Tensor, 
    max_tokens: int, 
    stop_tokens: list,
    model_name: str,
    request: ChatCompletionRequest,
    tokenizer
) -> ChatCompletionResponse:
    """Generate non-streaming chat completion"""
    
    # Generate tokens using the same interface as test_generate.py
    gen_result = model_instance.generate(
        input_ids.view(-1), 
        generation_length=max_tokens,
        teminators=stop_tokens,
        use_stream=False
    )
    
    # Handle different return formats based on model type
    config = model_config['config']
    if config.get('apply_eagle', False):
        # Eagle models return: (tokens, accept_lengths, decode_time, prefill_time)
        tokens, accept_lengths, decode_time, prefill_time = gen_result
    else:
        # Base models return: (tokens, decode_time, prefill_time)
        tokens, decode_time, prefill_time = gen_result
    
    # Decode response
    generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    
    # Extract only the assistant's response (after the last "Assistant:")
    if "Assistant:" in generated_text:
        response_text = generated_text.split("Assistant:")[-1].strip()
    else:
        response_text = generated_text.strip()
    
    # Determine finish reason
    finish_reason = "stop"
    if len(tokens) >= max_tokens:
        finish_reason = "length"
    elif any(token in tokens for token in stop_tokens):
        finish_reason = "stop"
    
    return ChatCompletionResponse(
        model=model_name,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason=finish_reason
            )
        ],
        usage={
            "prompt_tokens": input_ids.numel(),
            "completion_tokens": len(tokens),
            "total_tokens": input_ids.numel() + len(tokens)
        }
    )

async def stream_chat_completion(
    input_ids: torch.Tensor,
    max_tokens: int,
    stop_tokens: list, 
    model_name: str,
    request: ChatCompletionRequest,
    tokenizer
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion"""
    
    import time
    import uuid
    import asyncio
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    # Start generation stream
    stream_gen = model_instance.generate(
        input_ids.view(-1),
        generation_length=max_tokens,
        teminators=stop_tokens,
        use_stream=True
    )
    
    accumulated_text = ""
    
    try:
        for chunk in stream_gen:
            token = chunk['token']
            text = chunk['text']
            is_finished = chunk['is_finished']
            
            # Only send the new text part
            accumulated_text += text
            
            # Extract only assistant's response
            if "Assistant:" in accumulated_text:
                display_text = accumulated_text.split("Assistant:")[-1].strip()
            else:
                display_text = accumulated_text.strip()
            
            # Create streaming response
            if not is_finished:
                response = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": text},
                            finish_reason=None
                        )
                    ]
                )
            else:
                # Final chunk
                finish_reason = "stop"
                if token in stop_tokens:
                    finish_reason = "stop"
                elif len(accumulated_text) >= max_tokens:
                    finish_reason = "length"
                
                response = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={},
                            finish_reason=finish_reason
                        )
                    ]
                )
            
            # Yield with immediate flush to ensure real-time streaming
            yield f"data: {response.model_dump_json()}\n\n"
            
            # Add a small async yield to allow the event loop to process
            # This ensures immediate delivery of each chunk
            await asyncio.sleep(0)
            
            if is_finished:
                break
                
    except Exception as e:
        error_response = {
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "generation_failed"
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        await asyncio.sleep(0)
    
    yield "data: [DONE]\n\n"
    await asyncio.sleep(0)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error={
                "message": str(exc),
                "type": "internal_error", 
                "code": "server_error"
            }
        ).model_dump()
    )

def print_config(config):
    """Print all configuration parameters"""
    print("=" * 50)
    print("Server Configuration Parameters:")
    print("=" * 50)
    print(f"Features: eagle={config['apply_eagle']}, quant={config['apply_quant']}, sparse={config['apply_sparse']}")
    print(f"Generation: num_generate={config['num_generate']}, chunk_length={config['chunk_length']}, use_terminators={config['use_terminators']}")
    print(f"Sampling: temperature={config['temperature']}, random_seed={config['random_seed']}")
    print(f"Demo: use_enter={config['use_enter']}, use_decode_enter={config['use_decode_enter']}")
    print(f"Others: dtype={config['dtype']}, minicpm4_yarn={config['minicpm4_yarn']}, cuda_graph={config['cuda_graph']}, memory_limit={config['memory_limit']}")
    if config['apply_sparse']:
        print(f"Sparse Attention: sink_window={config['sink_window_size']}, block_window={config['block_window_size']}, sparse_topk_k={config['sparse_topk_k']}, sparse_switch={config['sparse_switch']}, compress_lse={config['apply_compress_lse']}")
    if config['apply_eagle']:
        print(f"Eagle: eagle_num_iter={config['eagle_num_iter']}, eagle_topk_per_iter={config['eagle_topk_per_iter']}, eagle_tree_size={config['eagle_tree_size']}, apply_eagle_quant={config['apply_eagle_quant']}, window_size={config['eagle_window_size']}, frspec_vocab_size={config['frspec_vocab_size']}")
    print("=" * 50)
    print()

def main():
    """Main entry point for the server"""
    parser = argparse.ArgumentParser(description="CPM.cu OpenAI API Server")
    
    # Basic arguments
    parser.add_argument('--path-prefix', '--path_prefix', '-p', type=str, default='openbmb', 
                        help='Path prefix for model directories, you can use openbmb to download models, or your own path (default: openbmb)')
    parser.add_argument("--config-file", "--config_file", type=str, default=None, 
                        help="Path to configuration file (JSON format, default: use default_config.json)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    # Model configuration boolean arguments
    parser.add_argument('--test-minicpm4', '--test_minicpm4', action='store_true',
                        help='Use MiniCPM4 model')
    parser.add_argument('--no-test-minicpm4', '--no_test_minicpm4', action='store_false', dest='test_minicpm4',
                        help='Do not use MiniCPM4 model')
    parser.add_argument('--apply-eagle', '--apply_eagle', action='store_true',
                        help='Use Eagle speculative decoding')
    parser.add_argument('--no-apply-eagle', '--no_apply_eagle', action='store_false', dest='apply_eagle',
                        help='Do not use Eagle speculative decoding')
    parser.add_argument('--apply-quant', '--apply_quant', action='store_true',
                        help='Use quantized model')
    parser.add_argument('--no-apply-quant', '--no_apply_quant', action='store_false', dest='apply_quant',
                        help='Do not use quantized model')
    parser.add_argument('--apply-sparse', '--apply_sparse', action='store_true',
                        help='Use sparse attention')
    parser.add_argument('--no-apply-sparse', '--no_apply_sparse', action='store_false', dest='apply_sparse',
                        help='Do not use sparse attention')
    parser.add_argument('--apply-eagle-quant', '--apply_eagle_quant', action='store_true',
                        help='Use quantized Eagle model')
    parser.add_argument('--no-apply-eagle-quant', '--no_apply_eagle_quant', action='store_false', dest='apply_eagle_quant',
                        help='Do not use quantized Eagle model')
    parser.add_argument('--apply-compress-lse', '--apply_compress_lse', action='store_true',
                        help='Apply LSE compression, only support on sparse attention')
    parser.add_argument('--no-apply-compress-lse', '--no_apply_compress_lse', action='store_false', dest='apply_compress_lse',
                        help='Do not apply LSE compression')
    parser.add_argument('--cuda-graph', '--cuda_graph', action='store_true',
                        help='Use CUDA graph optimization')
    parser.add_argument('--no-cuda-graph', '--no_cuda_graph', action='store_false', dest='cuda_graph',
                        help='Do not use CUDA graph optimization')
    parser.add_argument('--minicpm4-yarn', '--minicpm4_yarn', action='store_true',
                        help='Use MiniCPM4 YARN for very long context')
    parser.add_argument('--no-minicpm4-yarn', '--no_minicpm4_yarn', action='store_false', dest='minicpm4_yarn',
                        help='Do not use MiniCPM4 YARN')
    parser.add_argument('--use-terminators', '--use_terminators', action='store_true',
                        help='Use terminators for generation')
    parser.add_argument('--no-use-terminators', '--no_use_terminators', action='store_false', dest='use_terminators',
                        help='Do not use terminators')
    parser.add_argument('--use-enter', '--use_enter', action='store_true',
                        help='Use enter to generate')
    parser.add_argument('--no-use-enter', '--no_use_enter', action='store_false', dest='use_enter',
                        help='Do not use enter to generate')
    parser.add_argument('--use-decode-enter', '--use_decode_enter', action='store_true',
                        help='Use enter before decode phase')
    parser.add_argument('--no-use-decode-enter', '--no_use_decode_enter', action='store_false', dest='use_decode_enter',
                        help='Do not use enter before decode phase')

    # Model configuration numeric arguments  
    parser.add_argument('--frspec-vocab-size', '--frspec_vocab_size', type=int, default=None,
                        help='Frequent speculation vocab size')
    parser.add_argument('--eagle-window-size', '--eagle_window_size', type=int, default=None,
                        help='Eagle window size')
    parser.add_argument('--eagle-num-iter', '--eagle_num_iter', type=int, default=None,
                        help='Eagle number of iterations')
    parser.add_argument('--eagle-topk-per-iter', '--eagle_topk_per_iter', type=int, default=None,
                        help='Eagle top-k per iteration')
    parser.add_argument('--eagle-tree-size', '--eagle_tree_size', type=int, default=None,
                        help='Eagle tree size')
    parser.add_argument('--sink-window-size', '--sink_window_size', type=int, default=None,
                        help='Sink window size of sparse attention')
    parser.add_argument('--block-window-size', '--block_window_size', type=int, default=None,
                        help='Block window size of sparse attention')
    parser.add_argument('--sparse-topk-k', '--sparse_topk_k', type=int, default=None,
                        help='Sparse attention top-k')
    parser.add_argument('--sparse-switch', '--sparse_switch', type=int, default=None,
                        help='Context length of dense and sparse attention switch')
    parser.add_argument('--chunk-length', '--chunk_length', type=int, default=None,
                        help='Chunk length for prefilling')
    parser.add_argument('--memory-limit', '--memory_limit', type=float, default=None,
                        help='Memory limit for use')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Default temperature for processing')
    parser.add_argument('--dtype', type=str, default=None, choices=['float16', 'bfloat16'],
                        help='Model dtype')
    parser.add_argument('--random-seed', '--random_seed', type=int, default=None,
                        help='Random seed for processing')
    
    args = parser.parse_args()
    
    # Load configuration from file if specified
    if args.config_file:
        try:
            config = load_config_from_file(args.config_file)
            print(f"Loaded server configuration from: {args.config_file}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing config file {args.config_file}: {e}")
            sys.exit(1)
    else:
        config = get_default_config()
        print("Using default server configuration")
    
    # Set default values to None for boolean arguments that weren't specified
    bool_args = [key for key, value in config.items() if isinstance(value, bool)]
    for arg in bool_args:
        # Convert underscores to hyphens for command line argument names
        arg_hyphen = arg.replace('_', '-')
        # Check for both formats (hyphen and underscore)
        arg_specified = (f'--{arg_hyphen}' in sys.argv or f'--no-{arg_hyphen}' in sys.argv or
                        f'--{arg}' in sys.argv or f'--no-{arg}' in sys.argv)
        if not arg_specified:
            setattr(args, arg, None)

    # Define parameter mappings for automatic override (exclude dtype which needs special handling)
    auto_override_params = [key for key in config.keys() if key != 'dtype']

    # Override config values if arguments are provided
    for param in auto_override_params:
        arg_value = getattr(args, param, None)
        if arg_value is not None:
            config[param] = arg_value

    # Handle dtype separately due to type conversion
    if args.dtype is not None:
        config['dtype'] = torch.float16 if args.dtype == 'float16' else torch.bfloat16
    
    # Set global model config
    global model_config
    model_config = {
        "model_path_prefix": args.path_prefix,
        "config": config
    }
    
    print_config(config)
    
    print(f"Starting CPM.cu OpenAI API Server on {args.host}:{args.port}")
    print(f"Model path prefix: {args.path_prefix}")
    print(f"Configuration: {config}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )

if __name__ == "__main__":
    main() 