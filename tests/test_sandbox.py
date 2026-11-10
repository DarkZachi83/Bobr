"""Piaskownica dla kodu modelu: stała konfiguracja, uczciwy brak izolacji, prawdziwa izolacja (gdy jest)."""
import os
import shutil
import sys
import unittest

from aibobr.quality import checks, suites
from aibobr.quality.sandbox import Sandbox, user_processes

REFERENCE = {
    "fibonacci": "def fibonacci(n):\n    out, a, b = [], 0, 1\n    for _ in range(n):\n        out.append(a)\n"
                 "        a, b = b, a + b\n    return out\n",
    "is_palindrome": "def is_palindrome(s):\n    t = s.replace(' ', '').lower()\n    return t == t[::-1]\n",
    "roman_to_int": "def roman_to_int(s):\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
                    "    total = 0\n    for a, b in zip(s, s[1:] + ' '):\n"
                    "        total += -v[a] if b != ' ' and v[a] < v[b] else v[a]\n    return total\n",
}


class ConfigTest(unittest.TestCase):
    def test_no_isolation_is_reported_not_faked(self):
        self.assertEqual(Sandbox(platform="win32").status().reason, "unsupported")
        self.assertEqual(Sandbox(bwrap="", platform="linux").status().reason, "missing")
        none = Sandbox(bwrap="", platform="linux")
        self.assertEqual(none.run_tests("def f(): return 1", "f", []), (None, "missing"))
        r = checks.check("def f(x):\n    return x", {"type": "python", "function": "f", "tests": []}, run_python=none.run_tests)
        self.assertEqual((r.correct, r.reason), (None, "not_run"))           # nieocenione, nie błąd modelu

    def test_one_fixed_command(self):
        sb = Sandbox(bwrap="/usr/bin/bwrap", python="/usr/bin/python3", platform="linux")
        sb._sized = True
        cmd = sb.command("/tmp/w", "/work/x.py")
        for flag in ("--unshare-all", "--die-with-parent", "--new-session", "--clearenv"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--tmpfs") - 2:cmd.index("--tmpfs") + 2],
                         ["--size", str(64 * 1024 * 1024), "--tmpfs", "/tmp"])
        self.assertEqual(cmd[cmd.index("/tmp/w") - 1:cmd.index("/tmp/w") + 2], ["--ro-bind", "/tmp/w", "/work"])
        home = os.path.expanduser("~")
        binds = [cmd[i + 1] for i, a in enumerate(cmd) if a in ("--bind", "--ro-bind", "--ro-bind-try")]
        self.assertFalse(any(b == home or b.startswith(home + os.sep) for b in binds if home not in ("/", "/root")))
        self.assertNotIn("--bind", cmd)                                       # nic z zapisem do systemu
        self.assertEqual(cmd[-3:], ["-I", "-S", "/work/x.py"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux")
    def test_counts_users_processes(self):
        self.assertGreaterEqual(user_processes(), 1)


def real_sandbox():
    if not sys.platform.startswith("linux") or not shutil.which("bwrap"):
        return None, "brak bwrap"
    sb = Sandbox(use_cgroups=False)
    st = sb.status()
    return (sb, "") if st.ready else (None, st.reason)


SB, WHY = real_sandbox() if sys.platform.startswith("linux") else (None, "nie Linux")


@unittest.skipIf(SB is None, f"piaskownica niedostępna: {WHY}")
class RealSandboxTest(unittest.TestCase):
    """Na prawdziwym bubblewrap (Linux, zwykły użytkownik; root omija limit procesów – wtedy pominięte)."""

    def test_reference_solutions_pass_standard_tasks(self):
        science = suites.load("scisly_v1")
        for task in science["tasks"]:
            spec = task["check"]
            if spec["type"] != "python":
                continue
            r = checks.check(f"```python\n{REFERENCE[spec['function']]}```", spec, run_python=SB.run_tests)
            self.assertEqual((r.correct, r.format_compliant), (True, False), (task["id"], r))

    def test_wrong_and_hostile_code_fails_inside(self):
        spec = {"type": "python", "function": "f", "tests": [{"args": [1], "expected": 2}]}
        cases = {
            "def f(x):\n    return x + 2\n": "f(1) → 3",
            "import socket\ndef f(x):\n    socket.create_connection(('1.1.1.1', 53), timeout=2)\n    return 2\n": "OSError",
            f"import os\ndef f(x):\n    os.listdir({os.path.expanduser('~')!r})\n    return 2\n": "FileNotFoundError",
            "def f(x):\n    open('/usr/bobr', 'w')\n    return 2\n": "OSError",
            "def f(x):\n    bytearray(2 * 1024 ** 3)\n    return 2\n": "MemoryError",
        }
        for code, expected in cases.items():
            r = checks.check(code, spec, run_python=SB.run_tests)
            self.assertEqual(r.correct, False, code)
            self.assertIn(expected, r.value, code)

    def test_endless_loop_is_stopped(self):
        passed, detail = SB.run_tests("def f(x):\n    while True:\n        pass\n", "f", [{"args": [1], "expected": 2}])
        self.assertIs(passed, False)
        self.assertTrue(detail.startswith(("killed", "timeout")), detail)


if __name__ == "__main__":
    unittest.main()
