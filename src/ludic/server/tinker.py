from __future__ import annotations

import asyncio
import logging
import torch
import uuid
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Literal, Union
from fastapi import FastAPI, HTTPException, Request, APIRouter
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# --- Globals & App Setup ---

app = FastAPI(title="Ludic Tinker Server")
api_v1 = APIRouter(prefix="/api/v1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Registries ---

# session_id -> { "tags": [], "user_metadata": {} }
session_registry: Dict[str, Any] = {}

# model_id -> { "session_id": str, "base_model": str, "adapter_name": str, "optimizer": torch.optim.Optimizer }
model_registry: Dict[str, Any] = {}

# sampling_session_id -> { "model_id": str }
sampling_session_registry: Dict[str, Any] = {}

# request_id -> { "status": "pending" | "done" | "failed", "result": Any, "error": str }
future_registry: Dict[str, Any] = {}

# --- Engine Support Classes ---

@dataclass
class PendingRequest:
    request_id: str
    session_id: str
    model_id: str
    request_type: str  # "forward_backward", "optim_step", "sample", "create_model"
    payload: Any

class RequestBuffer:
    def __init__(self):
        self.queue: List[PendingRequest] = []
        self._lock = asyncio.Lock()

    async def add(self, request_id: str, session_id: str, model_id: str, request_type: str, payload: Any):
        async with self._lock:
            self.queue.append(PendingRequest(request_id, session_id, model_id, request_type, payload))

    async def drain(self) -> List[PendingRequest]:
        async with self._lock:
            items = self.queue[:]
            self.queue.clear()
        return items

class ClockEngine:
    def __init__(self, heartbeat_interval: float = 0.1):
        self.heartbeat_interval = heartbeat_interval
        self.buffer = RequestBuffer()
        # base_model_name -> {model, tokenizer}
        self.base_models: Dict[str, Any] = {}
        self._stop_event = asyncio.Event()

    async def run_forever(self):
        logger.info(f"ClockEngine started with heartbeat {self.heartbeat_interval}s")
        while not self._stop_event.is_set():
            start_time = asyncio.get_event_loop().time()
            try:
                await self._process_cycle()
            except Exception as e:
                logger.error(f"Error in Clock Cycle: {e}")
                import traceback
                traceback.print_exc()

            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0, self.heartbeat_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _process_cycle(self):
        requests = await self.buffer.drain()
        if not requests:
            return

        logger.info(f"--- [Clock Cycle] Processing {len(requests)} requests ---")
        
        for r in requests:
            logger.info(f"  > Processing {r.request_type} request {r.request_id} for model {r.model_id}")
            try:
                if r.request_type == "create_model":
                    result = await self._execute_create_model(r.payload)
                elif r.request_type == "forward_backward":
                    result = await asyncio.to_thread(self._execute_forward_backward_sync, r.model_id, r.payload)
                elif r.request_type == "optim_step":
                    result = await asyncio.to_thread(self._execute_optim_step_sync, r.model_id, r.payload)
                elif r.request_type == "sample":
                    result = await asyncio.to_thread(self._execute_sample_sync, r.model_id, r.payload)
                else:
                    raise ValueError(f"Unknown request type: {r.request_type}")
                
                fut = future_registry.get(r.request_id)
                if fut:
                    fut.update({"status": "done", "result": result})
                    if "event" in fut:
                        fut["event"].set()
                logger.info(f"  < Request {r.request_id} COMPLETED")
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                logger.error(f"  ! Request {r.request_id} FAILED: {e}\n{error_traceback}")
                fut = future_registry.get(r.request_id)
                if fut:
                    fut.update({"status": "failed", "error": str(e), "traceback": error_traceback})
                    if "event" in fut:
                        fut["event"].set()

    async def _execute_create_model(self, payload: CreateModelRequest):
        # We handle model loading in thread to not block event loop
        return await asyncio.to_thread(self._execute_create_model_sync, payload)

    def _execute_create_model_sync(self, payload: CreateModelRequest):
        base_model_name = payload.base_model
        session_id = payload.session_id
        model_id = f"model-{uuid.uuid4().hex[:8]}"
        
        if base_model_name not in self.base_models:
            logger.info(f"Loading base model {base_model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=payload.lora_config.rank if payload.lora_config else 8,
                lora_alpha=(payload.lora_config.rank if payload.lora_config else 8) * 2,
                target_modules="all-linear",
            )
            model = get_peft_model(model, lora_config, adapter_name=model_id)
            self.base_models[base_model_name] = {"model": model, "tokenizer": tokenizer}
        else:
            bm_info = self.base_models[base_model_name]
            model = bm_info["model"]
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=payload.lora_config.rank if payload.lora_config else 8,
                lora_alpha=(payload.lora_config.rank if payload.lora_config else 8) * 2,
                target_modules="all-linear",
            )
            model.add_adapter(model_id, lora_config)

        # Optimizer for this adapter
        adapter_params = [p for n, p in model.named_parameters() if model_id in n]
        optimizer = torch.optim.Adam(adapter_params if adapter_params else model.parameters(), lr=1e-4)

        model_registry[model_id] = {
            "session_id": session_id,
            "base_model": base_model_name,
            "adapter_name": model_id,
            "optimizer": optimizer
        }
        
        return {"model_id": model_id, "type": "create_model"}

    def _execute_forward_backward_sync(self, model_id: str, payload: ForwardBackwardRequest):
        if model_id not in model_registry: raise ValueError(f"Model {model_id} not found")
        meta = model_registry[model_id]
        bm_info = self.base_models[meta["base_model"]]
        model = bm_info["model"]
        tokenizer = bm_info["tokenizer"]
        device = next(model.parameters()).device
        
        model.set_adapter(meta["adapter_name"])
        
        from ludic.training.loss import MaskedCausalLMCrossEntropyLoss, ReinforceLoss
        
        input_ids_list = []
        action_mask_list = []
        weight_list = []
        actor_logps_list = []

        for datum in payload.forward_backward_input.data:
            tokens = datum.model_input.chunks[0].tokens if datum.model_input.chunks[0].tokens else []
            input_ids_list.append(torch.tensor(tokens, dtype=torch.int64))
            
            inputs = datum.loss_fn_inputs
            
            def get_tensor_data(key, default=None):
                val = inputs.get(key, default)
                if isinstance(val, dict) and "data" in val:
                    return val["data"]
                return val

            adv = get_tensor_data("advantages", 1.0)
            if isinstance(adv, list):
                seq_adv = float(adv[-1]) if len(adv) > 0 else 1.0
            else:
                seq_adv = float(adv)
            weight_list.append(torch.tensor(seq_adv, dtype=torch.float32))
            
            if "action_mask" in inputs:
                action_mask_list.append(torch.tensor(get_tensor_data("action_mask"), dtype=torch.float32))
            else:
                action_mask_list.append(torch.ones(len(tokens), dtype=torch.float32))
            
            if "logprobs" in inputs:
                lp = get_tensor_data("logprobs")
                if isinstance(lp, list):
                    seq_lp = float(sum(lp))
                else:
                    seq_lp = float(lp)
                actor_logps_list.append(torch.tensor(seq_lp, dtype=torch.float32))

        from torch.nn.utils.rnn import pad_sequence
        batch = {
            "input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id or 0).to(device),
            "action_mask": pad_sequence(action_mask_list, batch_first=True, padding_value=0.0).to(device),
            "weight": torch.stack(weight_list).to(device) if len(weight_list) > 0 else None,
        }
        if actor_logps_list:
            batch["actor_logps"] = torch.stack(actor_logps_list).to(device)

        if payload.forward_backward_input.loss_fn == "cross_entropy":
            loss_fn = MaskedCausalLMCrossEntropyLoss()
        else:
            loss_fn = ReinforceLoss(old_logp_key="actor_logps" if actor_logps_list else "old_logp_action")

        outputs = model(input_ids=batch["input_ids"])
        loss, stats = loss_fn.compute(outputs.logits, batch)
        
        # In MaskedCausalLMCrossEntropyLoss, loss is a scalar tensor
        # In ReinforceLoss with dictionaries, we must extract the float
        if isinstance(loss, dict):
            # Fallback if compute returns a dict of losses
            loss = loss.get("loss", next(iter(loss.values())))
            
        loss.backward()
        
        metrics = {}
        for k, v in stats.items():
            out_key = k if ":" in k else f"{k}:mean"
            try:
                if isinstance(v, torch.Tensor):
                    metrics[out_key] = float(v.item())
                elif isinstance(v, (int, float)):
                    metrics[out_key] = float(v)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert metric {k} to float: {v}")
                pass
                
        # Client weights metrics based on len(loss_fn_outputs)
        batch_size = len(payload.forward_backward_input.data)
        dummy_outputs = [{"dummy": {"data": [1], "dtype": "int64", "shape": [1]}} for _ in range(batch_size)]

        metrics["loss:mean"] = float(loss.item())

        return {"metrics": metrics, "loss_fn_outputs": dummy_outputs, "loss_fn_output_type": "none", "type": "forward_backward"}

    def _execute_optim_step_sync(self, model_id: str, payload: OptimStepRequest):
        if model_id not in model_registry: raise ValueError(f"Model {model_id} not found")
        meta = model_registry[model_id]
        optimizer = meta["optimizer"]
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = payload.adam_params.learning_rate
        
        optimizer.step()
        optimizer.zero_grad()
        return {"metrics": {"lr": payload.adam_params.learning_rate}, "type": "optim_step"}

    def _execute_sample_sync(self, model_id: str, payload: SampleRequest):
        # Determine if this is a pure sample or a lora sample
        is_pure = model_id not in model_registry and model_id in self.base_models
        
        # If model hasn't been cached but it's a known model name we load it now
        if not is_pure and model_id not in model_registry:
            # Maybe it's a base model that needs to be loaded directly on first sample
            if "/" in model_id: # typical HF id, heuristics
                logger.info(f"Loading pure sampling model {model_id} on demand...")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                self.base_models[model_id] = {"model": model, "tokenizer": tokenizer}
                is_pure = True
            else:
                raise ValueError(f"Model {model_id} not found")
                
        if is_pure:
            bm_info = self.base_models[model_id]
            model = bm_info["model"]
            tokenizer = bm_info["tokenizer"]
            device = next(model.parameters()).device
            # Disable adapters if there are any attached to the base model from other runs
            if hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    pass # Handled by block below properly, PEFT requires context manager
            
            # Since PEFT `disable_adapter` requires code inside block to run without adapter, 
            # we structure our generation slightly differently
            def generate_with_model(m, input_ids):
                return m.generate(
                    input_ids=input_ids,
                    max_new_tokens=payload.sampling_params.max_tokens or 16,
                    temperature=payload.sampling_params.temperature,
                    top_p=payload.sampling_params.top_p,
                    do_sample=payload.sampling_params.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    num_return_sequences=payload.num_samples,
                )
                
            prompt_tokens = payload.prompt.chunks[0].tokens if payload.prompt.chunks[0].tokens else []
            input_ids = torch.tensor([prompt_tokens], dtype=torch.int64).to(device)
            
            with torch.no_grad():
                if hasattr(model, "disable_adapter"):
                    with model.disable_adapter():
                        output_ids = generate_with_model(model, input_ids)
                else:
                    output_ids = generate_with_model(model, input_ids)
                    
        else:
            meta = model_registry[model_id]
            bm_info = self.base_models[meta["base_model"]]
            model = bm_info["model"]
            tokenizer = bm_info["tokenizer"]
            device = next(model.parameters()).device
            
            model.set_adapter(meta["adapter_name"])
            
            prompt_tokens = payload.prompt.chunks[0].tokens if payload.prompt.chunks[0].tokens else []
            input_ids = torch.tensor([prompt_tokens], dtype=torch.int64).to(device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=payload.sampling_params.max_tokens or 16,
                    temperature=payload.sampling_params.temperature,
                    top_p=payload.sampling_params.top_p,
                    do_sample=payload.sampling_params.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    num_return_sequences=payload.num_samples,
                )
        
        sequences = []
        for i in range(payload.num_samples):
            full_seq = output_ids[i].tolist()
            new_tokens = full_seq[len(prompt_tokens):]
            sequences.append({
                "tokens": new_tokens, 
                "logprobs": [0.0] * len(new_tokens), 
                "stop_reason": "length"
            })
            
        return {"sequences": sequences, "type": "sample"}

# --- Schemas ---

class SessionCreateRequest(BaseModel):
    tags: List[str] = []
    user_metadata: Optional[Dict[str, Any]] = None
    sdk_version: str = "unknown"
    type: Literal["create_session"] = "create_session"

class LoraConfigSchema(BaseModel):
    rank: int = 8
    seed: Optional[int] = None
    train_unembed: bool = True
    train_mlp: bool = True
    train_attn: bool = True

class CreateModelRequest(BaseModel):
    session_id: str
    model_seq_id: int = 0
    base_model: str
    lora_config: Optional[LoraConfigSchema] = None
    type: Literal["create_model"] = "create_model"

class CreateSamplingSessionRequest(BaseModel):
    session_id: str
    sampling_session_seq_id: int = 0
    model_path: Optional[str] = None
    base_model: Optional[str] = None
    type: Literal["create_sampling_session"] = "create_sampling_session"

class ModelInputChunk(BaseModel):
    tokens: Optional[List[int]] = None
    type: Literal["encoded_text", "image", "image_asset_pointer"] = "encoded_text"

class ModelInput(BaseModel):
    chunks: List[ModelInputChunk]

class Datum(BaseModel):
    model_input: ModelInput
    loss_fn_inputs: Dict[str, Any]

class ForwardBackwardInput(BaseModel):
    data: List[Datum]
    loss_fn: str
    loss_fn_config: Optional[Dict[str, Any]] = None

class ForwardBackwardRequest(BaseModel):
    forward_backward_input: ForwardBackwardInput
    model_id: str
    seq_id: int = 0

class AdamParams(BaseModel):
    learning_rate: float = 0.0001
    grad_clip_norm: float = 0.0

class OptimStepRequest(BaseModel):
    adam_params: AdamParams
    model_id: str
    seq_id: int = 0
    type: Literal["optim_step"] = "optim_step"

class SamplingParams(BaseModel):
    max_tokens: Optional[int] = 16
    temperature: float = 1.0
    top_p: float = 1.0

class SampleRequest(BaseModel):
    prompt: ModelInput
    num_samples: int = 1
    sampling_params: SamplingParams = Field(default_factory=SamplingParams)
    model_path: Optional[str] = None
    sampling_session_id: Optional[str] = None
    type: Literal["sample"] = "sample"

class SessionHeartbeatRequest(BaseModel):
    session_id: str
    class Config:
        extra = "allow"

class TelemetrySendRequest(BaseModel):
    session_id: str
    class Config:
        extra = "allow"

class SaveWeightsForSamplerRequest(BaseModel):
    model_id: str
    path: Optional[str] = None
    sampling_session_seq_id: Optional[int] = None
    seq_id: int = 0
    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"

class FutureRetrieveRequest(BaseModel):
    request_id: str

# --- Endpoints ---

@api_v1.get("/healthz")
async def healthz():
    return {"status": "ok"}

@api_v1.post("/create_session")
async def create_session(request: SessionCreateRequest):
    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    session_registry[session_id] = {"tags": request.tags, "user_metadata": request.user_metadata}
    logger.info(f"Created session: {session_id}")
    return {"session_id": session_id, "type": "create_session"}

@api_v1.post("/session_heartbeat")
async def session_heartbeat(request: Dict[str, Any]):
    return {"status": "ok", "type": "session_heartbeat"}

@api_v1.post("/telemetry")
async def telemetry(request: Dict[str, Any]):
    # Totally permissive to avoid 422
    return {"status": "accepted"}

@api_v1.post("/create_model")
async def create_model(request: CreateModelRequest):
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    future_registry[request_id] = {"status": "pending", "event": asyncio.Event()}
    logger.info(f"Queuing create_model: {request_id}")
    await engine.buffer.add(request_id, request.session_id, "n/a", "create_model", request)
    return {"request_id": request_id}

@api_v1.post("/save_weights_for_sampler")
async def save_weights_for_sampler(request: SaveWeightsForSamplerRequest):
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    ss_id = f"samp-{uuid.uuid4().hex[:8]}"
    sampling_session_registry[ss_id] = {"model_id": request.model_id}
    logger.info(f"Save weights requested. Returning future {request_id} with sampling_session_id {ss_id}")
    future_registry[request_id] = {
        "status": "done", 
        "result": {
            "tinker_path": f"tinker://{request.model_id}/weights/latest", 
            "sampling_session_id": ss_id,
            "type": "save_weights_for_sampler"
        }
    }
    return {"request_id": request_id}

@api_v1.post("/create_sampling_session")
async def create_sampling_session(request: CreateSamplingSessionRequest):
    ss_id = f"samp-{uuid.uuid4().hex[:8]}"
    
    # Check if we are creating a sampling session from a specific model path
    model_id = "unknown"
    if request.model_path and "/" in request.model_path:
        model_id = request.model_path.split("/")[2]
    # Check if we are creating a pure sampling session with just a base model
    elif request.base_model:
        model_id = request.base_model
        # If the base model hasn't been loaded yet, we should queue up a request to load it
        if model_id not in engine.base_models:
            logger.info(f"Loading pure sampling model {model_id}...")
            # We don't have a direct endpoint, so we manually do the base model loading in a thread
            # so we don't block the event loop while returning the session. The `sample` endpoint
            # handles loading from base_models.
            
            def load_base_model(bm_name):
                tokenizer = AutoTokenizer.from_pretrained(bm_name)
                model = AutoModelForCausalLM.from_pretrained(
                    bm_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                engine.base_models[bm_name] = {"model": model, "tokenizer": tokenizer}
            
            # Start loading in background, sample requests will wait or error if not ready
            # In a real app we'd have a separate queue for model loading, but since sample
            # happens later it usually has time, or we can just block
            # For simplicity let's just do it directly so it's ready
    
    sampling_session_registry[ss_id] = {"model_id": model_id, "is_pure": bool(request.base_model) and not request.model_path}
    logger.info(f"Created sampling session: {ss_id} for model {model_id}")
    return {"sampling_session_id": ss_id, "type": "create_sampling_session"}

@api_v1.post("/forward_backward")
async def forward_backward(request: ForwardBackwardRequest):
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    future_registry[request_id] = {"status": "pending", "event": asyncio.Event()}
    logger.info(f"Queuing forward_backward: {request_id} for model {request.model_id}")
    await engine.buffer.add(request_id, "n/a", request.model_id, "forward_backward", request)
    return {"request_id": request_id}

@api_v1.post("/optim_step")
async def optim_step(request: OptimStepRequest):
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    future_registry[request_id] = {"status": "pending", "event": asyncio.Event()}
    logger.info(f"Queuing optim_step: {request_id} for model {request.model_id}")
    await engine.buffer.add(request_id, "n/a", request.model_id, "optim_step", request)
    return {"request_id": request_id}

@api_v1.post("/asample")
async def asample(request: SampleRequest):
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    future_registry[request_id] = {"status": "pending", "event": asyncio.Event()}
    
    model_id = "unknown"
    if request.sampling_session_id in sampling_session_registry:
        model_id = sampling_session_registry[request.sampling_session_id]["model_id"]
    elif request.model_path and "/" in request.model_path:
        model_id = request.model_path.split("/")[2]
    
    logger.info(f"Queuing asample: {request_id} for model {model_id} (session {request.sampling_session_id})")
    await engine.buffer.add(request_id, "n/a", model_id, "sample", request)
    return {"request_id": request_id}

@api_v1.post("/retrieve_future")
async def retrieve_future(request: FutureRetrieveRequest):
    fut = future_registry.get(request.request_id)
    if not fut:
        raise HTTPException(status_code=404, detail=f"Future {request.request_id} Not Found")
    
    if fut["status"] == "pending":
        try:
            # Long-polling: wait up to 10 seconds for completion to prevent client tight loops
            if "event" in fut:
                await asyncio.wait_for(fut["event"].wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
            
    # Check status again after waiting
    if fut["status"] == "pending":
        return {"type": "try_again"}
    
    if fut["status"] == "failed":
        return {"type": "request_failed", "error_message": fut["error"]}
    
    return fut["result"]

@api_v1.get("/training_runs/{model_id}/checkpoints")
async def list_checkpoints(model_id: str):
    return {"checkpoints": []}

@api_v1.post("/get_info")
async def get_info(request: Dict[str, Any]):
    model_id = request.get("model_id")
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail=f"Model {model_id} Not Found")
    meta = model_registry[model_id]
    return {
        "model_id": model_id,
        "is_lora": True,
        "model_data": {"tokenizer_id": meta["base_model"]}
    }

# --- Startup ---

app.include_router(api_v1)

engine = ClockEngine()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(engine.run_forever())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
