import unittest

from sandbox_exec import run_sandboxed


class RunSandboxedTests(unittest.TestCase):
    def test_stdout_is_returned(self):
        result = run_sandboxed("print('hello')")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello", result.stdout)
        self.assertFalse(result.timed_out)

    def test_network_access_is_blocked(self):
        code = (
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "s.settimeout(2)\n"
            "try:\n"
            "    s.connect(('8.8.8.8', 53))\n"
            "    print('CONNECTED')\n"
            "except OSError as e:\n"
            "    print('BLOCKED')\n"
        )
        result = run_sandboxed(code, timeout=5)
        self.assertIn("BLOCKED", result.stdout)
        self.assertNotIn("CONNECTED", result.stdout)

    def test_infinite_loop_hits_timeout(self):
        result = run_sandboxed("while True: pass", timeout=1)
        self.assertTrue(result.timed_out)

    def test_memory_limit_is_enforced(self):
        code = (
            "try:\n"
            "    bytearray(1024 * 1024 * 1024)\n"
            "    print('ALLOCATED')\n"
            "except MemoryError:\n"
            "    print('BLOCKED')\n"
        )
        result = run_sandboxed(code, timeout=5)
        self.assertIn("BLOCKED", result.stdout)

    def test_filesystem_outside_tmp_is_read_only(self):
        code = (
            "try:\n"
            "    open('/etc/should-not-be-writable', 'w').close()\n"
            "    print('WROTE')\n"
            "except OSError:\n"
            "    print('BLOCKED')\n"
        )
        result = run_sandboxed(code, timeout=5)
        self.assertIn("BLOCKED", result.stdout)


if __name__ == "__main__":
    unittest.main()
