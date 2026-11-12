import math
import unittest

from aibobr.bench.ollama import Generation, OllamaBench
from aibobr.diagnostics import fragility as fr
from aibobr.diagnostics.fragility import FragilityProbe, Thresholds
from aibobr.quality import suites


def lp(token, p, *others):
    """Wpis logprobs jak z Ollamy: wybrany token i kandydaci (token, p)."""
    cands = [(token, p)] + list(others)
    return {"token": token, "logprob": math.log(p),
            "top_logprobs": [{"token": t, "logprob": math.log(q)} for t, q in cands]}


STABLE = [lp("Paryż", 0.97, ("Lyon", 0.01)), lp(".", 0.95, ("!", 0.02))]
FRAGILE = [lp("To", 0.9, ("W", 0.05)), lp(" wtorek", 0.52, (" piątek", 0.45))]    # odstęp 0,07
TIE = [lp("Za", 0.6, ("W", 0.2)), lp("żółć", 0.41, ("zółć", 0.40)), lp(" gęślą", 0.99)]


class Classify(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(fr.classify("S1", STABLE).label, "stable")
        f = fr.classify("S9", FRAGILE)
        self.assertEqual((f.label, f.fragile_steps, f.tie_steps, f.first_pos), ("fragile", 1, 0, 1))
        self.assertEqual(f.first_pair, ((" wtorek", 0.52), (" piątek", 0.45)))
        self.assertEqual(f.first_context, "To")
        self.assertAlmostEqual(f.min_margin, 0.07, places=6)
        t = fr.classify("H1", TIE)
        self.assertEqual((t.label, t.min_pos, t.first_context), ("tie", 1, "Za"))
        self.assertEqual(t.min_pair[0][0], "żółć")

    def test_single_candidate_and_order(self):
        """Bez drugiego kandydata odstęp = p1; kandydaci posortowani niezależnie od kolejności Ollamy."""
        s = fr.steps([{"token": "x", "logprob": math.log(0.99), "top_logprobs": []}])
        self.assertAlmostEqual(s[0].margin, 0.99)
        e = {"token": "b", "logprob": math.log(0.5),
             "top_logprobs": [{"token": "a", "logprob": math.log(0.3)}, {"token": "b", "logprob": math.log(0.5)}]}
        self.assertEqual(fr.steps([e])[0].first[0], "b")

    def test_thresholds_are_users(self):
        strict = Thresholds(fragile=0.05, tie=0.01)
        self.assertEqual(fr.classify("S9", FRAGILE, thresholds=strict).label, "stable")
        self.assertEqual(fr.classify("S9", FRAGILE, thresholds=Thresholds(0.5, 0.1)).label, "tie")
        with self.assertRaises(ValueError):
            Thresholds(fragile=0.01, tie=0.05)

    def test_missing_logprobs(self):
        self.assertEqual(fr.classify("x", None).label, "unsupported")
        self.assertEqual(fr.classify("x", []).label, "not_run")


class FakeOllama:
    def __init__(self, answers, support=True, release_ok=True):
        self.answers, self.support, self.release_ok = answers, support, release_ok
        self.loaded, self.calls = False, []

    def unload(self, model):
        if self.release_ok:
            self.loaded = False

    def is_loaded(self, model):
        return self.loaded

    def generate(self, model, prompt, num_ctx, num_predict, top_logprobs=None, **kw):
        self.calls.append((prompt, num_ctx, num_predict, top_logprobs, self.loaded))
        cached = self.loaded
        self.loaded = True
        steps = self.answers[prompt]
        text = "".join(s["token"] for s in steps)
        return Generation(10, 0.1, len(steps), 0.1, 0.5, text, prompt_cached=5 if cached else 0,
                          logprobs=steps if self.support else None)


def probe(fake):
    t = [0.0]
    return FragilityProbe(fake, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s), unload_timeout=1)


class Run(unittest.TestCase):
    TASKS = [("S1", "Stolica Francji?", 64), ("S9", "Jaki dzień?", 64), ("H1", "Popraw znaki", 64)]
    ANSWERS = {"Stolica Francji?": STABLE, "Jaki dzień?": FRAGILE, "Popraw znaki": TIE}

    def test_every_task_from_zero_with_logprobs(self):
        fake = FakeOllama(self.ANSWERS)
        events = []
        run = probe(fake).run("m", self.TASKS, report=events.append)
        self.assertEqual([t.label for t in run.tasks], ["stable", "fragile", "tie"])
        self.assertEqual(run.counts()["fragile"], 1)
        self.assertTrue(all(c[3] == 2 and c[4] is False for c in fake.calls))     # top 2, model zwolniony
        self.assertEqual({c[1] for c in fake.calls}, {2048})                       # jak test jakości
        self.assertFalse(fake.loaded)                                              # zwolniony na koniec
        self.assertEqual([e["event"] for e in events].count("f_task_done"), 3)
        self.assertEqual(run.tasks[1].answer, "To wtorek")

    def test_old_ollama_without_logprobs_stops(self):
        run = probe(FakeOllama(self.ANSWERS, support=False)).run("m", self.TASKS)
        self.assertEqual([t.label for t in run.tasks], ["unsupported"])

    def test_not_released(self):
        fake = FakeOllama(self.ANSWERS, release_ok=False)
        fake.loaded = True
        run = probe(fake).run("m", self.TASKS[:1])
        self.assertEqual((run.tasks[0].label, run.tasks[0].reason), ("not_run", "not_released"))
        self.assertEqual(fake.calls, [])

    def test_stop(self):
        run = probe(FakeOllama(self.ANSWERS)).run("m", self.TASKS, stop=lambda: True)
        self.assertEqual(run.tasks, [])

    def test_suite_tasks(self):
        tasks = fr.suite_tasks(suites.load("scisly_v1"), "en")
        self.assertEqual(len(tasks), 15)
        self.assertTrue(all(isinstance(t[1], str) and t[2] >= 256 for t in tasks))


class Client(unittest.TestCase):
    def test_logprobs_request_is_not_streamed(self):
        class Http:
            def __init__(self):
                self.sent = []

            def post_lines(self, url, payload, timeout=None):
                raise AssertionError("logprobs only without stream")

            def post_json(self, url, payload, timeout=None):
                self.sent.append(payload)
                return {"response": "x", "logprobs": [{"token": "x", "logprob": -0.1}]}
        http = Http()
        g = OllamaBench("http://h", http=http).generate("m", "p", 2048, 8, on_chunk=lambda t: None, top_logprobs=2)
        self.assertEqual((http.sent[0]["logprobs"], http.sent[0]["top_logprobs"], http.sent[0]["stream"]),
                         (True, 2, False))
        self.assertEqual(g.logprobs, [{"token": "x", "logprob": -0.1}])


if __name__ == "__main__":
    unittest.main()
