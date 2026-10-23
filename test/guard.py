"""Call a function of the participant's main.py in a forked child with a hard timeout.

The parent process never imports main.py, so an infinite loop, a hang at import time or a crash only
costs one test (bounded by `timeout`), which keeps the whole judge inside Quera's time limit.
Copied verbatim into the test folder of each Python judge as `guard.py`.
"""
import multiprocessing
import os
import sys
import time

_CTX = multiprocessing.get_context("fork")


class Timeout(Exception):
    pass


class CallError(Exception):
    pass


def _child(conn, root, func_name, args):
    try:
        sys.path.insert(0, root)
        import main  # noqa: F401  (imported here, in the child only)
        started = time.perf_counter()
        result = getattr(main, func_name)(*args)
        took = time.perf_counter() - started
        conn.send(("ok", result, took))
    except BaseException as exc:  # noqa: BLE001 - report anything the participant raises
        try:
            conn.send(("error", "%s: %s" % (type(exc).__name__, exc), 0.0))
        except Exception:
            pass
    finally:
        conn.close()


def call(root, func_name, args, timeout):
    """Return (result, seconds spent inside the call). Raises Timeout or CallError."""
    receiver, sender = _CTX.Pipe(duplex=False)
    proc = _CTX.Process(target=_child, args=(sender, root, func_name, args))
    proc.start()
    sender.close()
    try:
        if not receiver.poll(timeout):
            raise Timeout()
        try:
            kind, value, took = receiver.recv()
        except EOFError:
            raise CallError("برنامه وسط اجرا از کار افتاد (کد خروج %s)." % proc.exitcode)
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join()
        receiver.close()
    if kind == "error":
        raise CallError(value)
    return value, took
