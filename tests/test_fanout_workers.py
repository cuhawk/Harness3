"""Fan-out concurrency must be tunable independently of this box's core count.

The default pool size is ``cpu_count - 2``, a CPU-bound heuristic applied to an I/O-bound
workload: every sub-agent thread sits blocked on an HTTP call to the router. With a
multi-key router the real ceiling is provider rate limits, not cores, so the pool has to be
raisable past ``cpu_count`` - and pinnable low while measuring where 429s start. These tests
hold that contract, including the cases where a bad value must fall back rather than throw.
"""
import os
import unittest
from unittest import mock

from harness_agent.subagent import _fanout_workers


CPU_DEFAULT = max(1, (os.cpu_count() or 1) - 2)


class FanoutWorkerTests(unittest.TestCase):
    def _workers(self, task_count, env=None):
        patched = {"HARNESS_MAP_WORKERS": env} if env is not None else {}
        with mock.patch.dict(os.environ, patched, clear=False):
            if env is None:
                os.environ.pop("HARNESS_MAP_WORKERS", None)
            return _fanout_workers(task_count)

    def test_default_is_cpu_derived(self):
        self.assertEqual(self._workers(1000), CPU_DEFAULT)

    def test_never_exceeds_task_count(self):
        # Two tasks must not open a 40-thread pool just because the override allows it.
        self.assertEqual(self._workers(2, env="40"), 2)
        self.assertEqual(self._workers(2), min(2, CPU_DEFAULT))

    def test_override_raises_pool_above_cpu_count(self):
        # The whole point: an I/O-bound fan-out should be allowed past cpu_count - 2.
        self.assertEqual(self._workers(1000, env=str(CPU_DEFAULT + 30)), CPU_DEFAULT + 30)

    def test_override_can_pin_low_for_rate_limit_measurement(self):
        self.assertEqual(self._workers(1000, env="1"), 1)

    def test_bad_values_fall_back_to_default(self):
        # A typo in the env must not crash a sweep or silently serialize it.
        for bad in ("", "   ", "abc", "0", "-4", "3.5"):
            with self.subTest(bad=bad):
                self.assertEqual(self._workers(1000, env=bad), CPU_DEFAULT)

    def test_never_returns_zero(self):
        # ThreadPoolExecutor(max_workers=0) raises; an empty batch must still be legal.
        self.assertEqual(self._workers(0), 1)
        self.assertEqual(self._workers(0, env="8"), 1)


if __name__ == "__main__":
    unittest.main()
