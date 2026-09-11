"""Sandboxed execution of untrusted (model-generated) code via bubblewrap.

Used by verifiers/code_reward.py to check GRPO completions against test cases without
trusting the model's output. Isolation: no network namespace, no filesystem access outside
a fresh tmpfs, capped memory/CPU, killed on timeout.
"""

import dataclasses
import resource
import subprocess

_MAX_MEMORY_BYTES = 512 * 1024 * 1024
_MAX_CPU_SECONDS = 10

_BWRAP_ARGS = [
    "bwrap",
    "--unshare-all",
    "--die-with-parent",
    "--new-session",
    "--ro-bind", "/usr", "/usr",
    "--ro-bind-try", "/bin", "/bin",
    "--ro-bind-try", "/lib", "/lib",
    "--ro-bind-try", "/lib64", "/lib64",
    "--ro-bind-try", "/sbin", "/sbin",
    "--proc", "/proc",
    "--dev", "/dev",
    "--tmpfs", "/tmp",
    "--chdir", "/tmp",
    "--setenv", "PATH", "/usr/bin:/bin",
]


@dataclasses.dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def _limit_resources():
    resource.setrlimit(resource.RLIMIT_AS, (_MAX_MEMORY_BYTES, _MAX_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (_MAX_CPU_SECONDS, _MAX_CPU_SECONDS))


def run_sandboxed(code: str, stdin: str = "", timeout: float = 10.0) -> ExecResult:
    """Runs `code` as a Python script inside a bwrap sandbox and returns the outcome.

    Never raises on the sandboxed code's own errors, non-zero exit, or timeout -- those are
    all reported via the returned ExecResult, not exceptions.
    """
    cmd = [*_BWRAP_ARGS, "python3", "-c", code]
    try:
        proc = subprocess.run(
            cmd,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_limit_resources,
        )
        return ExecResult(proc.stdout, proc.stderr, proc.returncode, timed_out=False)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return ExecResult(stdout, stderr, -1, timed_out=True)
