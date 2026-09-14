"""HuggingFace check: is the saved token valid, what can it do, and is the gated model reachable?

Never prints the token. Reports the token's name, its role, and per model: ok / gate not accepted /
token lacks the gated-repos permission.
"""

from __future__ import annotations

from tmg import paths
from tmg.jobs import protocol

MODELS = ["stabilityai/stable-audio-open-1.0", "facebook/musicgen-stereo-small"]


def run(params: dict) -> dict:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    if not paths.HF_TOKEN_FILE.exists():
        return {"token": "not set", "models": {}}
    api = HfApi()
    protocol.progress(0.2, "checking the token")
    try:
        me = api.whoami()
    except Exception as exc:  # noqa: BLE001
        return {"token": f"INVALID ({type(exc).__name__})", "models": {}}
    tok = (me.get("auth") or {}).get("accessToken") or {}
    fg = tok.get("fineGrained") or {}
    perms = sorted({p for e in (fg.get("scoped") or []) for p in (e.get("permissions") or [])} | set(fg.get("global") or []))
    result = {"token": "valid", "token_name": tok.get("displayName"), "role": tok.get("role"), "permissions": perms, "models": {}}
    for i, repo in enumerate(MODELS):
        protocol.progress(0.4 + 0.5 * i / len(MODELS), repo)
        try:
            api.auth_check(repo)
            result["models"][repo] = "ok"
        except GatedRepoError:
            result["models"][repo] = ("gated: accept access on the model page, and make sure the token (if fine-grained) has "
                                      "'Read contents of public gated repos you can access' - or use a classic Read token")
        except HfHubHTTPError as exc:
            result["models"][repo] = f"http {exc.response.status_code if exc.response is not None else '?'}"
        except Exception as exc:  # noqa: BLE001
            result["models"][repo] = f"{type(exc).__name__}"
    protocol.progress(1.0, "done")
    return result
