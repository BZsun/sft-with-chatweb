import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer
# 部分版本的 transformers 未把 BaseStreamer 导出到顶层，走具体模块路径最稳
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

# ---------------- 日志 ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------- 配置（默认值来自项目根 .env，已 export 的环境变量优先） ----------------
# 模型解析链：合并后微调模型 → LoRA 适配器 + 基座 → 纯基座，启动时打印实际加载来源
MERGED_MODEL_PATH = os.getenv("MERGED_MODEL_PATH", "./finetune/Qwen3-8B-finetuned")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "./outputs/lora_adapter")
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "/ethan/pretrained_model/Qwen3-8B")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "Qwen3-8B")
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8118"))

# 生成参数服务端默认值（客户端请求显式传参时覆盖）
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.8"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "20"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))
DEFAULT_ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").strip().lower() in ("1", "true", "yes", "on")

# GPU 推理由 Semaphore(1) 串行化：同一时刻只跑一条序列，显存峰值确定
# （权重+单请求 KV cache）。请求直接进入异步端点等待信号量；
# 等待中的请求数超过该上限立即返回 429 让客户端快速失败重试，
# 避免无界堆积最终全部超时
GEN_WAITING_LIMIT = 32

# 每条请求推理结束后，闲置缓存超过该值就 empty_cache 还给驱动；
# GPU0 与 embedServer 共驻，回收可缓解共卡显存压力
RECLAIM_THRESHOLD_BYTES = 2 * 1024**3

THINK_END_TOKEN_ID = 151668  # Qwen3 </think>


def _is_model_dir(path: str) -> bool:
    return (Path(path) / "config.json").is_file()


def _is_lora_dir(path: str) -> bool:
    return (Path(path) / "adapter_config.json").is_file()


def resolve_load_source() -> Dict[str, str]:
    """微调模型解析链：合并模型 → LoRA 适配器 + 基座 → 纯基座。"""
    if _is_model_dir(MERGED_MODEL_PATH):
        return {"mode": "merged", "model_path": MERGED_MODEL_PATH}
    if _is_lora_dir(LORA_ADAPTER_PATH) and _is_model_dir(BASE_MODEL_PATH):
        return {"mode": "lora", "model_path": BASE_MODEL_PATH, "lora_path": LORA_ADAPTER_PATH}
    return {"mode": "base", "model_path": BASE_MODEL_PATH}


tokenizer = None
model = None
gen_semaphore: Optional[asyncio.Semaphore] = None
_waiting = 0  # 等待信号量的请求数（事件循环单线程，检查+自增间无 await 即原子）


class ServerOverloaded(Exception):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, gen_semaphore
    src = resolve_load_source()
    logging.info(f"[startup] load source: mode={src['mode']}, model={src['model_path']}"
                 + (f", lora={src['lora_path']}" if src["mode"] == "lora" else ""))
    logging.info(f"[startup] loading tokenizer from {src['model_path']} ...")
    tokenizer = AutoTokenizer.from_pretrained(src["model_path"])
    # 单请求无 padding，pad token 仅用于 generate 的 pad_token_id，缺失时用 eos 兜底
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"[startup] loading model from {src['model_path']} ...")
    model = AutoModelForCausalLM.from_pretrained(
        src["model_path"],
        torch_dtype="auto",
        device_map=0,
    )
    if src["mode"] == "lora":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, src["lora_path"])
        logging.info(f"[startup] LoRA adapter attached: {src['lora_path']}")
    model.eval()
    # 清空 generation_config 内置采样默认值：采样参数只来自客户端请求，
    # 贪心路径也不会再触发 HF 的 do_sample=False 告警
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    gen_semaphore = asyncio.Semaphore(1)
    logging.info(
        f"[startup] model ready (serial single-request serving), "
        f"OpenAI API: http://{HOST}:{PORT}/v1, Web chat: http://{HOST}:{PORT}/"
    )
    yield
    # in-flight 的 to_thread generate 无法中断，uvicorn 关闭时等线程自然结束即可
    torch.cuda.empty_cache()


app = FastAPI(title="Qwen3 OpenAI-compatible async server (serial GPU serving, client sampling + streaming)", lifespan=lifespan)


# ---------------- OpenAI 请求/响应模型 ----------------
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    # openai SDK 的 extra_body 字段会平铺到请求体顶层：
    # - enable_thinking 直接作为顶层字段接收（DashScope/Qwen 官方风格），
    #   优先级高于 chat_template_kwargs 内的同名参数
    # - chat_template_kwargs 兼容 vLLM 风格 {"enable_thinking": false}
    # - repetition_penalty 同理平铺接收（非 OpenAI 标准字段）
    model_config = ConfigDict(extra="ignore")
    model: str = SERVED_MODEL_NAME
    messages: List[ChatMessage]
    # 采样参数缺省时使用 .env 中的服务端默认值；
    # temperature 显式传 0（OpenAI 语义=贪心）或默认值 <=0 时走贪心解码
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: bool = True  # 默认流式输出
    stream_options: Optional[StreamOptions] = None
    enable_thinking: Optional[bool] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: str
    reasoning_content: Optional[str] = None


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


def openai_error(status_code: int, message: str, err_type: str = "invalid_request_error", code: Optional[str] = None):
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type, "code": code}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return openai_error(400, f"Invalid request: {exc.errors()}")


def effective_max_new_tokens(req: ChatCompletionRequest) -> int:
    return req.max_tokens or req.max_completion_tokens or DEFAULT_MAX_NEW_TOKENS


def maybe_reclaim_cache():
    # 单请求串行，此处无并发推理，回收安全
    if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > RECLAIM_THRESHOLD_BYTES:
        torch.cuda.empty_cache()
        logging.info(
            f"[cache] reclaimed idle cache, mem_reserved={torch.cuda.memory_reserved() / 1024**3:.2f}G"
        )


async def acquire_slot():
    global _waiting
    if _waiting >= GEN_WAITING_LIMIT:
        raise ServerOverloaded
    _waiting += 1
    try:
        await gen_semaphore.acquire()
    finally:
        _waiting -= 1


# ---------------- 请求 -> 推理入参 ----------------
def prepare_inputs(req: ChatCompletionRequest) -> Dict[str, Any]:
    # 思考模式：顶层 enable_thinking 优先，其次 chat_template_kwargs（vLLM 风格），
    # 都没传则用 .env 的 ENABLE_THINKING 默认值
    enable_thinking = DEFAULT_ENABLE_THINKING
    if req.chat_template_kwargs:
        enable_thinking = bool(req.chat_template_kwargs.get("enable_thinking", enable_thinking))
    if req.enable_thinking is not None:
        enable_thinking = req.enable_thinking
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 采样参数：客户端显式传参 ?? .env 服务端默认值。
    # 只在采样分支传采样键：temperature <= 0（OpenAI 语义 0=贪心）走纯贪心，
    # 不传采样键；HF 的 TemperatureLogitsWarper 要求 temperature>0，传 0.0 会直接抛错
    temperature = req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": effective_max_new_tokens(req),
    }
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = req.top_p if req.top_p is not None else DEFAULT_TOP_P
        gen_kwargs["top_k"] = req.top_k if req.top_k is not None else DEFAULT_TOP_K
    repetition_penalty = req.repetition_penalty if req.repetition_penalty is not None else DEFAULT_REPETITION_PENALTY
    if repetition_penalty is not None and repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    return {
        "model_inputs": model_inputs,
        "prompt_len": int(model_inputs.input_ids.shape[1]),
        "gen_kwargs": gen_kwargs,
        "max_new": effective_max_new_tokens(req),
        "enable_thinking": enable_thinking,
    }


def finalize_output_ids(out_ids: List[int], max_new: int) -> Tuple[List[int], str]:
    # 截掉 EOS 及之后的内容，判定 finish_reason
    eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    if not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]
    for j, t in enumerate(out_ids):
        if t in eos_ids:
            out_ids = out_ids[:j]
            if len(out_ids) <= max_new:
                return out_ids, "stop"
            break
    # 无 EOS 跑满，或 EOS 在自身上限之后才出现：截到自身上限
    return out_ids[:max_new], "length"


# ---------------- 非流式推理（在工作线程中执行） ----------------
def run_generation(prep: Dict[str, Any]) -> Dict[str, Any]:
    gen_kwargs = prep["gen_kwargs"]
    prompt_len = prep["prompt_len"]
    max_new = prep["max_new"]

    start_t = time.perf_counter()
    logging.info(
        f"[generate] start prompt_len={prompt_len} "
        f"max_new_tokens={max_new} "
        f"temperature={prep['gen_kwargs'].get('temperature')} "
        f"mem_allocated={torch.cuda.memory_allocated() / 1024**3:.2f}G "
        f"mem_reserved={torch.cuda.memory_reserved() / 1024**3:.2f}G"
    )
    with torch.inference_mode():
        generated_ids = model.generate(**prep["model_inputs"], **gen_kwargs)
    logging.info(
        f"[generate] done in {time.perf_counter() - start_t:.1f}s "
        f"completion_tokens={int(generated_ids.shape[1]) - prompt_len} "
        f"mem_allocated={torch.cuda.memory_allocated() / 1024**3:.2f}G "
        f"mem_reserved={torch.cuda.memory_reserved() / 1024**3:.2f}G"
    )

    out_ids, finish_reason = finalize_output_ids(generated_ids[0][prompt_len:].tolist(), max_new)

    # 解析思考内容：定位最后一个 </think>（与 modelCall.py 相同逻辑）
    try:
        index = len(out_ids) - out_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        index = 0

    reasoning_content = tokenizer.decode(out_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(out_ids[index:], skip_special_tokens=True).strip("\n")

    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "prompt_tokens": int(prompt_len),
        "completion_tokens": len(out_ids),
        "finish_reason": finish_reason,
    }


# ---------------- 流式推理 ----------------
class TokenIdStreamer(BaseStreamer):
    # 用 token id 而非解码文本做流：需要按 id 151668 精确判定思考/正文边界，
    # 而 skip_special_tokens 解码会把 </think> 标记本身剥掉
    def __init__(self):
        self.queue: Queue = Queue()

    def put(self, value):
        # 初始调用传入完整 prompt（2-D 且 seq_len>1）：跳过，只流式生成增量，
        # 否则 prompt 会混进 out_ids 污染 token 计数与思考边界判定
        if value.dim() > 1 and value.shape[1] > 1:
            return
        # 兼容不同 transformers 版本的逐步解码形状：旧版 (batch, 1) 2-D、新版 (batch,) 1-D，
        # 统一展平成 list[int]，避免 value[0] 取出 0 维标量导致 extend 报错
        self.queue.put(value.reshape(-1).tolist())

    def end(self):
        self.queue.put(None)


class StopOnDisconnect(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.event.is_set()


async def stream_chat_chunks(req: ChatCompletionRequest, prep: Dict[str, Any]):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_name = req.model or SERVED_MODEL_NAME
    enable_thinking = prep["enable_thinking"]
    include_usage = bool(req.stream_options and req.stream_options.include_usage)

    prompt_tokens = prep["prompt_len"]
    completion_tokens = 0
    start_t = time.perf_counter()

    stop_event = threading.Event()
    streamer = TokenIdStreamer()

    def _generate_worker():
        try:
            with torch.inference_mode():
                model.generate(
                    **prep["model_inputs"],
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([StopOnDisconnect(stop_event)]),
                    **prep["gen_kwargs"],
                )
        except Exception as exc:
            streamer.queue.put(exc)
        finally:
            streamer.end()

    thread = threading.Thread(target=_generate_worker, daemon=True)
    thread.start()
    logging.info(
        f"[generate-stream] start prompt_len={prompt_tokens} "
        f"max_new_tokens={prep['max_new']} enable_thinking={enable_thinking} "
        f"mem_allocated={torch.cuda.memory_allocated() / 1024**3:.2f}G "
        f"mem_reserved={torch.cuda.memory_reserved() / 1024**3:.2f}G"
    )

    def sse_frame(obj: Dict[str, Any]) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def base_chunk(delta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }

    out_ids: List[int] = []
    seen_think_end = False
    prev_text = ""
    error: Optional[Exception] = None
    loop = asyncio.get_running_loop()

    try:
        yield sse_frame(base_chunk({"role": "assistant", "content": ""}))
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: streamer.queue.get(timeout=5))
            except Empty:
                if not thread.is_alive():
                    logging.error("[generate-stream] generation thread died without sentinel")
                    break
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                error = item
                break
            out_ids.extend(item)

            # </think> 本身解码为空文本，本步 delta 仍是边界前的思考内容，
            # 之后的 delta 才是正文。边界判定与 enable_thinking 解耦：
            # 即使关思考，模型异常输出的思考段也会被正确分栏进 reasoning_content
            boundary_before = seen_think_end
            if not seen_think_end and THINK_END_TOKEN_ID in item:
                seen_think_end = True

            # 累积解码取增量：多字节中文可能跨 token，delta 含替换符就等下一步
            text = tokenizer.decode(out_ids, skip_special_tokens=True)
            delta_text = text[len(prev_text):]
            if "\ufffd" in delta_text:
                continue
            prev_text = text
            if not delta_text:
                continue
            if enable_thinking and not boundary_before:
                delta = {"reasoning_content": delta_text}
            else:
                delta = {"content": delta_text}
            yield sse_frame(base_chunk(delta))

        if error is not None:
            logging.error(f"[generate-stream] generation failed: {error}", exc_info=error)
            yield sse_frame({"error": {"message": f"Inference failed: {error}", "type": "server_error", "code": None}})
            yield "data: [DONE]\n\n"
            return

        final_ids, finish_reason = finalize_output_ids(out_ids, prep["max_new"])
        completion_tokens = len(final_ids)
        yield sse_frame({**base_chunk({}), "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
        if include_usage:
            yield sse_frame({
                **base_chunk({}),
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })
        yield "data: [DONE]\n\n"
        logging.info(
            f"[generate-stream] done in {time.perf_counter() - start_t:.1f}s "
            f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
            f"finish_reason={finish_reason}"
        )
    finally:
        # 正常结束此处是 no-op；客户端断开时通知 generate 在一个 token 内停止，
        # 不让已断开请求继续占用 GPU
        stop_event.set()
        if thread.is_alive():
            # 断开场景下生成器处于取消态，原地 await 会立即被打断、跳过 release
            # 造成信号量泄漏（后续请求全部挂起）——交给独立任务在线程退出后放锁
            asyncio.get_running_loop().create_task(_finalize_after_generate(thread))
        else:
            gen_semaphore.release()
            maybe_reclaim_cache()
        if error is not None or thread.is_alive():
            logging.info(
                f"[generate-stream] ended early in {time.perf_counter() - start_t:.1f}s, "
                f"completion_tokens~{len(out_ids)}"
            )


async def _finalize_after_generate(thread: threading.Thread):
    """断开场景的兜底收尾：等生成线程退出（≤10s）后释放信号量并回收显存缓存。

    必须放在独立任务里 await：流式生成器的 finally 处于 anyio 取消作用域内，
    任何原地 await 都会被立即打断，导致 gen_semaphore.release() 永远不执行、
    信号量泄漏、后续所有请求挂起。
    """
    try:
        await asyncio.wait_for(asyncio.to_thread(thread.join, 10), timeout=12)
    except Exception:
        pass
    gen_semaphore.release()
    maybe_reclaim_cache()
    logging.info("[generate-stream] finalized after client disconnect, semaphore released")


# ---------------- 端点 ----------------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        return openai_error(400, "messages must contain at least one message")
    if model is None or tokenizer is None or gen_semaphore is None:
        return openai_error(503, "Model is still loading, please retry later.", err_type="server_error")

    try:
        await acquire_slot()
    except ServerOverloaded:
        logging.warning(f"[slot] waiting limit reached ({GEN_WAITING_LIMIT}), rejecting with 429")
        return openai_error(
            429,
            "The server is overloaded (too many waiting requests), please retry later.",
            err_type="rate_limit_error",
            code="rate_limit_exceeded",
        )

    try:
        prep = prepare_inputs(request)

        if request.stream:
            # 流式：锁的释放在 SSE 生成器的 finally 中
            return StreamingResponse(stream_chat_chunks(request, prep), media_type="text/event-stream")

        task = asyncio.create_task(asyncio.to_thread(run_generation, prep))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            # 客户端断开：后台线程无法中断，等它自然跑完再放锁，
            # 避免僵尸 generate 与下一请求在 GPU 上重叠
            try:
                await task
            except Exception:
                pass
            raise
        except Exception as exc:
            logging.error(f"[generate] generation failed: {exc}", exc_info=exc)
            return openai_error(500, f"Inference failed: {exc}", err_type="server_error")
        finally:
            gen_semaphore.release()
            maybe_reclaim_cache()

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model or SERVED_MODEL_NAME,
            choices=[
                Choice(
                    message=AssistantMessage(
                        content=result["content"],
                        reasoning_content=result["reasoning_content"] or None,
                    ),
                    finish_reason=result["finish_reason"],
                )
            ],
            usage=Usage(
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                total_tokens=result["prompt_tokens"] + result["completion_tokens"],
            ),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return openai_error(500, f"Request failed: {exc}", err_type="server_error")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "modelscope",
            }
        ],
    }


# ---------------- Web 聊天界面（与 API 同端口托管） ----------------

STATIC_DIR = Path(__file__).resolve().parent / "serve" / "static"


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    # log_config=None：跳过 uvicorn 自带日志配置，让其日志传播到 root，
    # 统一走 basicConfig 的格式（带日期时间）
    uvicorn.run(app, host=HOST, port=PORT, log_config=None)
