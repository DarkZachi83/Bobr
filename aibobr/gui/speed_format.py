"""Teksty wyników benchmarku prędkości 0.9 (bez tkintera – testowalne wszędzie).

Zasady autora (9.11.2026): werdykt główny tylko z pakietu 15 zadań; przy nim zawsze podstawa
(„60,0 tok/s z 578 tokenów w 9,6 s”); główna prędkość promptu z pomiaru 1k (brak / błąd →
„Pomiar 1k niedostępny”, bez podstawiania krótkich promptów); przy 1k/8k rzeczywisty TTFT
(`ttft_wall_ms`, nie wyliczenie) z oceną informacyjną; brak pomiaru → „Brak danych”.
"""
from __future__ import annotations

import json
from typing import Optional

from aibobr.bench import speed
from aibobr.i18n import fmt_context, t

DOTS = {"fast": "🟢", "useful": "🔵", "slow": "🟡", "too_slow": "🔴"}


def num(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "–"
    text = f"{x:,.{digits}f}".replace(",", " ")
    return text.replace(".", t("bench.decimal"))


def ms(value: Optional[float]) -> str:
    """56 ms; od sekundy – „4,35 s”."""
    if value is None:
        return "–"
    return f"{num(value, 0)} ms" if value < 1000 else f"{num(value / 1000, 2)} s"


def level(name: Optional[str]) -> str:
    return f"{DOTS[name]} {t(f'speed.level.{name}')}" if name in DOTS else t("speed.no_data")


def loads(value, default=None):
    if value is None or not isinstance(value, str):
        return value if value is not None else default
    try:
        return json.loads(value)
    except ValueError:
        return default


def verdict(run: dict) -> str:
    if run.get("status") == "unreliable":
        return "⚠ " + t("speed.unreliable")
    return level(run.get("verdict"))


def generation(run: dict) -> str:
    b = speed.basis(run)
    if b is None:
        return t("speed.no_data") if run.get("gen_tps") is None else f"{num(run['gen_tps'])} tok/s"
    return t("speed.gen_basis", tps=num(b["tps"]), tokens=num(b["tokens"], 0), seconds=num(b["seconds"]))


def ttft(run: dict) -> str:
    if run.get("ttft_wall_median_ms") is None:
        return t("speed.no_data")
    return t("speed.ttft_pair", median=ms(run["ttft_wall_median_ms"]), worst=ms(run.get("ttft_wall_max_ms")))


def long_for(run: dict, ctx: int) -> Optional[dict]:
    return next((lc for lc in run.get("long_context") or () if lc.get("num_ctx") == ctx), None)


def long_valid(lc: Optional[dict]) -> bool:
    return bool(lc) and lc.get("valid") == 1


def long_cell(run: dict, ctx: int, th: dict) -> str:
    """„55,9 tok/s · TTFT 0,41 s 🟢”; nie mierzony – „—”; nieważny – „Brak danych (powód)”."""
    lc = long_for(run, ctx)
    if lc is None:
        return "—"
    if not long_valid(lc):
        return f"{t('speed.no_data')} ({reason(lc.get('invalid_reason'))})"
    parts = [f"{num(lc.get('tg_tps'))} tok/s"]
    if lc.get("ttft_wall_ms") is not None:
        lvl = speed.ttft_level(lc["ttft_wall_ms"] / 1000, th)
        parts.append(f"TTFT {ms(lc['ttft_wall_ms'])} {DOTS.get(lvl, '')}".rstrip())
    else:
        parts.append(f"TTFT {t('speed.no_data')}")
    return " · ".join(parts)


def prompt_main(run: dict) -> str:
    """Główna prędkość promptu – z pomiaru 1k (decyzja B)."""
    lc = long_for(run, 1024)
    if not long_valid(lc) or lc.get("pp_tps") is None:
        return t("speed.pp_1k_missing")
    return t("speed.pp_1k", tps=num(lc["pp_tps"], 0), tokens=num(lc.get("prompt_tokens"), 0))


def reason(code: Optional[str]) -> str:
    return t(f"speed.reason.{code}") if code else "?"


def warnings(run: dict) -> str:
    codes = loads(run.get("warnings_json"), run.get("warnings_json")) or []
    return " · ".join(t(f"speed.warn.{c}") for c in codes) if codes else ""


def flags(task) -> str:
    items = task.flags if hasattr(task, "flags") else (loads(task.get("flags_json"), []) or [])
    return ", ".join(t(f"speed.flag.{f}") for f in items)


def _get(task, key):
    return getattr(task, key) if hasattr(task, key) else task.get(key)


def task_tps(task) -> Optional[float]:
    tokens, eval_ms = _get(task, "gen_tokens"), _get(task, "eval_ms")
    return tokens / (eval_ms / 1000) if tokens and eval_ms and not _get(task, "error") else None


def details(run: dict, th: dict) -> str:
    """Tekst okna szczegółów przebiegu (dwuklik w tabeli Wydajność)."""
    lines = [f"{run['model']} · {t('speed.run_n', n=run.get('run_no', 1))}",
             f"{t('speed.col.verdict')}: {verdict(run)}"]
    if run.get("verdict_measure"):
        lines.append(t(f"speed.measure.{run['verdict_measure']}"))
    lines += [f"{t('speed.col.gen')}: {generation(run)}",
              f"{t('speed.col.ttft')}: {ttft(run)}",
              f"{t('speed.col.pp')}: {prompt_main(run)}",
              t("speed.pp_pack", tps=num(run.get("pp_tps"), 0)),
              t("speed.load", ms=ms(run.get("load_wall_ms")), engine=ms(run.get("load_ms"))),
              t("speed.overhead", pct=num((run.get("overhead_ratio") or 0) * 100)),
              t("speed.counted", ok=run.get("tasks_counted", 0), errors=run.get("tasks_error", 0)),
              t("speed.stabilization", status=t(f"speed.stab.{run.get('stabilization') or 'none'}"),
                seconds=num(run.get("cooling_seconds"), 0))]
    if warnings(run):
        lines.append("⚠ " + warnings(run))
    base = loads(run.get("baseline_json"), run.get("baseline_json")) or {}
    mean = base.get("mean") or {}
    if mean:
        lines.append(t("speed.baseline", cpu=num(mean.get("cpu_pct")), cpu_t=num(mean.get("cpu_temp_c"), 0),
                       gpu=num(mean.get("gpu_util_pct")), gpu_t=num(mean.get("gpu_temp_c"), 0),
                       vram=num(mean.get("vram_used_mb"), 0), ram=num(mean.get("ram_available_mb"), 0)))
    used = loads(run.get("thresholds_json"), run.get("thresholds_json")) or th
    lines.append(t("speed.thresholds", gen=" / ".join(num(x, 0) for x in used["gen_tps"]),
                   ttft=" / ".join(num(x, 0) for x in used["ttft_s"])))
    lines.append("")
    lines.append(t("speed.tasks_head"))
    for task in run.get("tasks") or ():
        err = _get(task, "error")
        state = t(f"speed.error.{err.split(':')[0]}") if err else flags(task)
        lines.append(f"{_get(task, 'task_no'):>2}. {_get(task, 'task_id'):<26} {num(task_tps(task)):>7} tok/s "
                     f"{_get(task, 'gen_tokens') or 0:>5} tok  TTFT {ms(_get(task, 'ttft_wall_ms')):>8}  {state}")
    longs = run.get("long_context") or ()
    if longs:
        lines.append("")
        lines.append(t("speed.long_head"))
        for lc in longs:
            head = f"{fmt_context(lc['num_ctx'])}: "
            if long_valid(lc):
                head += t("speed.long_line", tokens=num(lc.get("prompt_tokens"), 0), ctx=num(lc["num_ctx"], 0),
                          gen=num(lc.get("gen_tokens"), 0), pp=num(lc.get("pp_tps"), 0), tg=num(lc.get("tg_tps")),
                          ttft=ms(lc.get("ttft_wall_ms")))
                lvl = speed.ttft_level(lc["ttft_wall_ms"] / 1000, used) if lc.get("ttft_wall_ms") else None
                if lvl:
                    head += f" · TTFT {level(lvl)} ({t('speed.info_only')})"
            else:
                head += f"{t('speed.no_data')} ({reason(lc.get('invalid_reason'))})"
            lines.append(head)
    lines.append("")
    lines.append(t("speed.disclaimer"))
    return "\n".join(lines)
