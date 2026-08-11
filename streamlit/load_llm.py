"""GGUF model loader — POC level, LLM-agnostic via llama-cpp-python.

Automatically detects and loads any GGUF file in models/ (e.g., qwen2.5-3b-instruct-q4_k_m.gguf).
Tested with Qwen2.5, Phi-3, Mistral, TinyLlama.

Usage
-----
    from load_llm import get_llm
    llm = get_llm()                                              # auto-detects model
    llm = get_llm("models/qwen2.5-3b-instruct-q4_k_m.gguf")      # explicit path
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env on first import (no-op if already loaded or file missing)
load_dotenv()


def _get_default_model_path() -> str:
    """Return configured model path, or auto-download Qwen GGUF if missing."""
    env_path = os.environ.get("LLM_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if models_dir.exists():
        gguf_files = sorted(models_dir.glob("*.gguf"))
        if gguf_files:
            return str(gguf_files[0])

    # Auto-download Qwen2.5 GGUF model from Hugging Face if not present
    try:
        from huggingface_hub import hf_hub_download
        print("  [LLM] GGUF model missing locally — downloading Qwen2.5 from Hugging Face...")
        downloaded = hf_hub_download(
            repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
            local_dir=str(models_dir),
        )
        return downloaded
    except Exception:
        return str(models_dir / "qwen2.5-1.5b-instruct-q4_k_m.gguf")


DEFAULT_MODEL_PATH: str = _get_default_model_path()

DEFAULT_N_CTX: int = int(os.environ.get("LLM_N_CTX", "2048"))
DEFAULT_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "512"))
DEFAULT_N_THREADS: int | None = (
    int(os.environ["LLM_N_THREADS"]) if "LLM_N_THREADS" in os.environ else None
)
DEFAULT_VERBOSE: bool = os.environ.get("LLM_VERBOSE", "0") == "1"

# ── Module-level cache so we only load model into RAM once ────────────────────
_loaded: dict[str, Any] = {}


def get_llm(
    model_path: str | Path | None = None,
    *,
    n_ctx: int | None = None,
    n_threads: int | None = None,
    verbose: bool | None = None,
) -> Any:
    """Return a cached llama_cpp.Llama instance for *model_path*.

    Parameters
    ----------
    model_path:
        Path to the .gguf file. Defaults to DEFAULT_MODEL_PATH / env var.
    n_ctx:
        Context window size in tokens.
    n_threads:
        CPU threads. None = auto-detect (llama.cpp default).
    verbose:
        Show llama.cpp loading output.
    """
    if os.name == "nt":
        # Register site-packages llama_cpp/lib directory with Windows DLL loader
        try:
            for path_item in sys.path:
                candidate = Path(path_item) / "llama_cpp" / "lib"
                if candidate.exists():
                    os.add_dll_directory(str(candidate.resolve()))
        except Exception:
            pass

    try:
        from llama_cpp import Llama
    except (ImportError, RuntimeError, OSError) as exc:
        err_msg = str(exc)
        if "llama.dll" in err_msg or "Could not find module" in err_msg:
            raise RuntimeError(
                f"Failed to load shared library llama.dll ({err_msg}).\n\n"
                "CAUSE: The installed wheel ('llama_cpp_python-0.3.20+cuda13.0...') was compiled for CUDA 13.0 / Blackwell GPUs,\n"
                "so Windows cannot find the required CUDA 13 runtime DLLs on your system.\n\n"
                "FIX:\n"
                "  Run: uv add llama-cpp-python\n"
                "  (or install Visual C++ Redistributable / CUDA drivers if using CUDA wheel).\n"
            ) from exc
        raise RuntimeError("llama-cpp-python is not installed.\nRun: uv add llama-cpp-python") from exc

    path = str(model_path or DEFAULT_MODEL_PATH)

    if not Path(path).exists():
        raise FileNotFoundError(
            f"GGUF model file not found at '{path}'.\n"
            "Please place your .gguf model in models/ or set LLM_MODEL_PATH in .env."
        )

    if path not in _loaded:
        resolved_ctx = n_ctx if n_ctx is not None else DEFAULT_N_CTX
        resolved_verbose = verbose if verbose is not None else DEFAULT_VERBOSE
        resolved_threads = n_threads if n_threads is not None else DEFAULT_N_THREADS
        kwargs: dict[str, Any] = {
            "model_path": path,
            "n_ctx": resolved_ctx,
            "verbose": resolved_verbose,
        }
        if resolved_threads is not None:
            kwargs["n_threads"] = resolved_threads
        _loaded[path] = Llama(**kwargs)

    return _loaded[path]


def unload_llm(model_path: str | Path | None = None) -> None:
    """Remove a loaded model from the cache (frees RAM)."""
    key = str(model_path or DEFAULT_MODEL_PATH)
    _loaded.pop(key, None)
