"""Process deadline for native StructureMatcher lattice searches."""

import multiprocessing
import os
import threading

class _MatcherTimedOut(TimeoutError):
    pass


_matcher_worker = None
_matcher_lock = threading.Lock()


def _matcher_process(connection):
    connection.send("ready")
    while True:
        matcher, reference, candidate = connection.recv()
        try:
            connection.send((True, matcher.get_rms_dist(reference, candidate)))
        except Exception as exc:
            connection.send((False, str(exc)))


def _matcher_rmsd(matcher, reference, candidate, seconds):
    """Bound native lattice searches too; Python alarms cannot interrupt them."""
    if seconds <= 0:
        return matcher.get_rms_dist(reference, candidate)
    global _matcher_worker
    # ponytail: one worker per scorer process; add workers only if matching dominates runtime.
    with _matcher_lock:
        if _matcher_worker is None or _matcher_worker[0] != os.getpid():
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            process = context.Process(target=_matcher_process, args=(child,), daemon=True)
            process.start()
            child.close()
            if not parent.poll(60):
                process.kill()
                process.join()
                parent.close()
                raise RuntimeError("matcher worker failed to start")
            assert parent.recv() == "ready"
            _matcher_worker = (os.getpid(), process, parent)
        _, process, connection = _matcher_worker
        try:
            connection.send((matcher, reference, candidate))
            if not connection.poll(seconds):
                raise _MatcherTimedOut
            ok, value = connection.recv()
            if not ok:
                raise ValueError(value)
            return value
        except BaseException:
            if process.is_alive():
                process.kill()
            process.join()
            connection.close()
            _matcher_worker = None
            raise


