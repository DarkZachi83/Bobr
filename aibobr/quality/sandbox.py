"""Piaskownica dla kodu Pythona napisanego przez model (decyzja autora 9.11.2026, pytanie 5a).

Kod modelu jest niezaufany. Uruchamiamy go WYŁĄCZNIE w bubblewrap na Linuksie, w jednej stałej
konfiguracji: bez sieci, bez katalogu domowego, system tylko do odczytu, prywatny pusty /tmp,
minimalne /dev (bez karty graficznej), osobne procesy, koniec razem z testem. Bubblewrap sam nie
ogranicza zasobów, więc dokładamy limity jądra (rlimit: pamięć, czas procesora, rozmiar plików,
liczba otwartych plików) i limit czasu z zewnątrz; cgroups (systemd-run --user --scope) tylko jako
dodatek, gdy działają.

Przed pierwszym użyciem test kontrolny: w piaskownicy próbujemy połączyć się z siecią, przeczytać
katalog domowy, zapisać plik poza /tmp, przekroczyć limit pamięci i zobaczyć kartę graficzną.
Automatyczna ocena kodu włącza się tylko, gdy WSZYSTKIE próby się nie udały. Inaczej – „izolacja
niedostępna” z powodem, bez cichego przełączania na słabszy mechanizm (Windows, Linux bez bwrap).

Nie analizujemy treści kodu (to nie antywirus) – zamykamy go w pudełku, cokolwiek zrobi.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MEMORY_MB = 512
CPU_SECONDS = 10
WALL_SECONDS = 15
FILE_MB = 8
TMP_MB = 64              # cały prywatny /tmp (w pamięci)
EXTRA_PROCESSES = 48     # ponad procesy, które użytkownik już ma (RLIMIT_NPROC liczy wszystkie jego procesy)
MARK = "BOBR_RESULT:"

HARNESS = r'''
import json, sys
sys.__stdout__.write("BOBR_START\n"); sys.__stdout__.flush()   # od tej chwili działa kod modelu
MARK = "BOBR_RESULT:"
def out(obj):
    sys.stdout.flush()
    sys.__stdout__.write("\n" + MARK + json.dumps(obj, default=repr) + "\n")
spec = json.load(open(sys.argv[2], encoding="utf-8"))
ns = {"__name__": "bobr_task"}
try:
    exec(compile(open(sys.argv[1], encoding="utf-8").read(), "<model>", "exec"), ns)
except BaseException as e:
    out({"error": type(e).__name__ + ": " + str(e)[:300]}); sys.exit(0)
fn = ns.get(spec["function"])
if not callable(fn):
    out({"error": "brak funkcji " + spec["function"]}); sys.exit(0)
results = []
for t in spec["tests"]:
    try:
        results.append({"value": fn(*t["args"])})
    except BaseException as e:
        results.append({"error": type(e).__name__ + ": " + str(e)[:300]})
out({"results": results})
'''

PROBE = r'''
import json, os, socket, sys
home = sys.argv[1]
broken = []
try:
    s = socket.create_connection(("1.1.1.1", 53), timeout=2); s.close(); broken.append("network")
except OSError:
    pass
try:
    os.listdir(home); broken.append("home")
except OSError:
    pass
try:
    open("/usr/bobr_probe", "w").close(); broken.append("write")
except OSError:
    pass
try:
    block = bytearray(%d * 1024 * 1024); broken.append("memory")
except MemoryError:
    pass
if any(n.startswith(("nvidia", "dri", "kfd")) for n in os.listdir("/dev")):
    broken.append("gpu")
try:
    for i in range(%d):
        with open("/tmp/fill%%d" %% i, "wb") as f:
            f.write(b"x" * (4 * 1024 * 1024))
    broken.append("tmp")
except OSError:
    pass
children = []
try:
    for i in range(%d):
        pid = os.fork()
        if pid == 0:
            import time; time.sleep(5); os._exit(0)
        children.append(pid)
    broken.append("processes")
except OSError:
    pass
for pid in children:
    try:
        os.kill(pid, 9)
    except OSError:
        pass
print("BOBR_RESULT:" + json.dumps({"broken": broken}))
''' % (MEMORY_MB * 2, TMP_MB // 4 + 4, EXTRA_PROCESSES * 4)


@dataclass
class Status:
    ready: bool
    reason: str          # ok / unsupported / missing / failed:<co się udało> / error:<treść>


def user_processes(uid: Optional[int] = None) -> int:
    """Ile procesów ma już użytkownik – RLIMIT_NPROC liczy je wszystkie, nie tylko piaskownicę."""
    uid = os.getuid() if uid is None else uid
    n = 0
    for entry in os.scandir("/proc"):
        if entry.name.isdigit():
            try:
                with open(f"/proc/{entry.name}/status", encoding="ascii", errors="replace") as f:
                    for line in f:
                        if line.startswith("Uid:"):
                            n += int(line.split()[1]) == uid
                            break
            except OSError:
                continue
    return n


def limiter(nproc: int):
    """W procesie potomnym przed exec: limity jądra (nie wymagają uprawnień administratora).
    Uwaga: root (CAP_SYS_RESOURCE) omija limit procesów – wtedy test kontrolny to wykryje."""
    def apply() -> None:
        import resource
        mem = MEMORY_MB * 1024 * 1024
        for res, value in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, CPU_SECONDS),
                           (resource.RLIMIT_FSIZE, FILE_MB * 1024 * 1024), (resource.RLIMIT_NOFILE, 64),
                           (resource.RLIMIT_CORE, 0), (resource.RLIMIT_NPROC, nproc)):
            try:
                resource.setrlimit(res, (value, value))
            except (ValueError, OSError):
                pass
        os.setsid()
    return apply


class Sandbox:
    def __init__(self, bwrap: Optional[str] = None, python: Optional[str] = None,
                 platform: str = sys.platform, use_cgroups: bool = True):
        self.platform = platform
        self.bwrap = bwrap if bwrap is not None else shutil.which("bwrap")
        self.python = python or os.path.realpath(sys.executable)
        self.use_cgroups = use_cgroups
        self._status: Optional[Status] = None
        self._scope: Optional[list[str]] = None
        self._sized: Optional[bool] = None

    # -------------------------------------------------------------- polecenie
    def _binds(self) -> list[str]:
        args = ["--ro-bind", "/usr", "/usr"]
        for link in ("/bin", "/lib", "/lib64", "/sbin"):
            if os.path.islink(link):
                args += ["--symlink", os.readlink(link), link]
            elif os.path.isdir(link):
                args += ["--ro-bind", link, link]
        for extra in ("/etc/ld.so.cache", "/etc/localtime", "/etc/alternatives"):
            args += ["--ro-bind-try", extra, extra]
        prefix = os.path.realpath(sys.base_prefix)       # interpreter spoza /usr (np. /opt, pyenv)
        exe_dir = os.path.dirname(self.python)
        for path in {prefix, exe_dir}:
            if path and not path.startswith("/usr"):
                args += ["--ro-bind", path, path]
        return args

    def command(self, work: str, *argv: str) -> list[str]:
        """Jedna stała konfiguracja (bez opcji do wyboru) – patrz opis modułu."""
        return [self.bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
                "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp",
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "PYTHONIOENCODING", "utf-8",
                *self._binds(), "--proc", "/proc", "--dev", "/dev", *self._tmp(),
                "--ro-bind", work, "/work", "--chdir", "/tmp",
                self.python, "-I", "-S", *argv]

    def _tmp(self) -> list[str]:
        """Prywatny /tmp w pamięci – z limitem rozmiaru (bwrap ≥ 0.9: --size), inaczej test
        kontrolny wykryje brak limitu i piaskownica nie zostanie włączona."""
        if self._sized is None:
            try:
                help_text = subprocess.run([self.bwrap, "--help"], capture_output=True, text=True, timeout=10).stdout
            except (OSError, subprocess.SubprocessError):
                help_text = ""
            self._sized = "--size" in help_text
        return (["--size", str(TMP_MB * 1024 * 1024)] if self._sized else []) + ["--tmpfs", "/tmp"]

    def _cgroup_prefix(self) -> list[str]:
        if self._scope is None:
            self._scope = []
            tool = shutil.which("systemd-run") if self.use_cgroups else None
            if tool:
                probe = [tool, "--user", "--scope", "--quiet", "-p", f"MemoryMax={MEMORY_MB}M", "-p", "TasksMax=32",
                         "true"]
                try:
                    if subprocess.run(probe, capture_output=True, timeout=10).returncode == 0:
                        self._scope = probe[:-1]
                except (OSError, subprocess.SubprocessError):
                    pass
        return self._scope

    def _run(self, files: dict[str, str], *argv: str) -> tuple[Optional[dict], str]:
        """(wynik spod MARK, komunikat). Pliki trafiają do /work tylko do odczytu."""
        with tempfile.TemporaryDirectory(prefix="bobr-sandbox-") as work:
            for name, text in files.items():
                Path(work, name).write_text(text, encoding="utf-8")
            os.chmod(work, 0o755)
            out_path = Path(work).with_suffix(".out")
            cmd = self._cgroup_prefix() + self.command(work, *argv)
            try:
                with open(out_path, "w+b") as out:
                    nproc = user_processes() + EXTRA_PROCESSES
                    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                            preexec_fn=limiter(nproc))
                    try:
                        proc.wait(timeout=WALL_SECONDS)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except OSError:
                            proc.kill()
                        proc.wait()
                        return None, "timeout"
                    out.seek(0)
                    text = out.read(FILE_MB * 1024 * 1024).decode("utf-8", "replace")
            except OSError as e:
                return None, f"error:{e}"
            finally:
                try:
                    out_path.unlink()
                except OSError:
                    pass
        lines = [line for line in text.splitlines() if line.startswith(MARK)]
        if not lines:
            if "BOBR_START" in text.splitlines():               # kod modelu ruszył i został przerwany
                return None, f"killed:{proc.returncode}"
            return None, f"error:{text.strip()[-300:] or proc.returncode}"
        return json.loads(lines[-1][len(MARK):]), "ok"

    # -------------------------------------------------------------- sprawdzenie izolacji
    def status(self) -> Status:
        if self._status is None:
            self._status = self._verify()
        return self._status

    def _verify(self) -> Status:
        if not self.platform.startswith("linux"):
            return Status(False, "unsupported")
        if not self.bwrap:
            return Status(False, "missing")
        home = os.path.expanduser("~")
        result, msg = self._run({"probe.py": PROBE}, "/work/probe.py", home)
        if result is None:
            return Status(False, msg if ":" in msg else f"error:{msg}")
        broken = result.get("broken", ["?"])
        return Status(not broken, "ok" if not broken else "failed:" + ",".join(broken))

    # -------------------------------------------------------------- sprawdzenie kodu
    def run_tests(self, code: str, function: str, tests: list) -> tuple[Optional[bool], str]:
        """Dla checks.check_python: (True/False, opis) albo (None, powód), gdy piaskownica niedostępna."""
        st = self.status()
        if not st.ready:
            return None, st.reason
        spec = json.dumps({"function": function, "tests": tests}, ensure_ascii=False)
        result, msg = self._run({"code.py": code, "spec.json": spec, "harness.py": HARNESS},
                                "/work/harness.py", "/work/code.py", "/work/spec.json")
        if result is None:
            if msg == "timeout" or msg.startswith("killed:"):     # limit czasu/pamięci/procesów – wina kodu
                return False, msg
            return None, msg
        if "error" in result:
            return False, result["error"]
        from aibobr.quality.checks import same
        for test, got in zip(tests, result["results"]):
            call = f"{function}({', '.join(map(repr, test['args']))})"
            if "error" in got:
                return False, f"{call}: {got['error']}"
            value = json.loads(json.dumps(got["value"], default=repr))   # krotka = lista, jak w JSON
            if not same(value, test["expected"]):
                return False, f"{call} → {value!r}, oczekiwano {test['expected']!r}"
        return True, "ok"
