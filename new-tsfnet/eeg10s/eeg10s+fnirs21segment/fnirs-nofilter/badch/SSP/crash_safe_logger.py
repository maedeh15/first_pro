# ============================================================================
# Crash-safe output logging -- works the same way whether you're running
# tsfnet_model_notebook.py cell-by-cell (or via %run) in Jupyter, OR running
# run_all_subjects.py as a plain command-line script (python run_all_subjects.py
# --seed 42). Duplicates EVERY print() (and any error traceback) to BOTH the
# normal console/notebook output AND a plain-text file on disk, flushing to
# disk after every completed line. If the kernel crashes, the terminal is
# killed, or an exception kills the run partway through, everything printed
# UP TO that point is already safely on disk.
#
# COMPATIBILITY NOTES FOR THIS PROJECT'S TWO ENTRY POINTS
# --------------------------------------------------------------------------
# 1. run_all_subjects.py wraps its ENTIRE 26-subject run in one
#    `with CrashSafeLogger(...) as log:` block. Inside that block it calls
#    run_subject_cross_session(...) once per subject (defined in
#    tsfnet_model_notebook.py / the tsfnet_model package).
# 2. tsfnet_model_notebook.py's training loops print per-epoch status lines,
#    and its diagnostics routines print "[n/9] ..." progress lines -- some
#    of these (and any tqdm bars, if you add them later) use "\r" to update
#    a single line in place rather than starting a new line every time.
#
# Two things matter for that combination to work well and not need any
# changes on either side:
#   - NESTING-SAFE: if code under tsfnet_model_notebook.py's control ever
#     opens its own CrashSafeLogger (e.g. for a single ad-hoc subject run in
#     a notebook cell) while run_all_subjects.py's outer CrashSafeLogger is
#     already active, this version detects that sys.stdout/sys.stderr are
#     already wrapped and reuses the existing tee instead of double-wrapping
#     it -- so output is never duplicated N times for N nested loggers, and
#     the inner block's __exit__ correctly leaves the outer logger's wrapping
#     in place afterward instead of restoring the real console streams too
#     early.
#   - PROGRESS-BAR/CARRIAGE-RETURN SAFE: a "\r"-updated status line (tqdm,
#     or any print(..., end="\r")) still updates live on the console/notebook
#     exactly as before, but only the FINAL state of that line is written to
#     the log FILE (once it's superseded by a newline or a different line)
#     instead of every intermediate tick. Without this, a long training run
#     with per-batch progress bars can turn a log file into hundreds of MB
#     of redundant lines and slow every batch down with disk flushes.
#
# UPDATED FOR THE BASELINE / ABLATION MULTI-EXPERIMENT ARCHITECTURE:
# This project has multiple parallel experiment folders that must never mix
# logs (e.g. TSFNET_PAPER_EXACT_nback for the 11-segment baseline vs.
# TSFNET_PAPER_EXACT_nback_20SEG_ABLATION for the 20-segment ablation):
#   - CrashSafeLogger creates any missing parent directories for log_path
#     automatically (pointing it at a not-yet-created experiment subfolder
#     no longer crashes on open()).
#   - The optional `experiment_tag` parameter labels WHICH experiment a log
#     belongs to directly in the start/stop banners.
#
# USAGE (identical for a notebook cell or a standalone script):
#
#     with CrashSafeLogger("VP001_training_log.txt") as log:
#         result = run_subject_cross_session(...)
#
#     with CrashSafeLogger(
#         os.path.join(DATA_ROOT, f"baseline_run_log_{RUN_TAG}.txt"),
#         experiment_tag="11SEG_BASELINE",
#     ) as log:
#         ...
#
# Everything printed inside the `with` block appears on the console/notebook
# cell AS NORMAL, and simultaneously accumulates in the log file in real
# time. If an exception occurs inside the block, its full traceback is also
# written to the log file before being re-raised (so a plain script still
# prints its usual traceback to the real console afterward, and a notebook
# cell still shows its usual red error display -- the traceback is ALSO
# safely on disk either way).
# ============================================================================
import os
import sys
import traceback
import datetime


class CrashSafeLogger:
    """
    log_path: exact filename to use. If append=False (default), and this
    exact file already exists, a numeric suffix is added automatically
    (e.g. "VP001_training_log_2.txt") so a NEW run never silently mixes
    into an OLD run's log -- each attempt gets its own clean file by
    default. Pass append=True to explicitly opt into appending onto an
    existing file instead (e.g. if you WANT one combined history file
    across multiple attempts).

    experiment_tag: optional short label (e.g. "11SEG_BASELINE" or
    "20SEG_ABLATION") identifying which experiment this run belongs to.
    Purely cosmetic -- included in the start/stop banners -- but very useful
    when you have multiple experiment variants (baseline vs. ablation) and
    want the log file itself to make unambiguous which one produced it,
    even if you later move or rename the file.

    Safe to use from a Jupyter notebook cell (tsfnet_model_notebook.py) or
    from a plain `python run_all_subjects.py` process -- both just need
    `sys.stdout`/`sys.stderr` to be writable text streams, which is true in
    both contexts. Also safe to nest (e.g. an outer logger around the whole
    26-subject run in run_all_subjects.py, with an inner logger opened for
    just one subject somewhere inside tsfnet_model_notebook.py) -- the inner
    logger detects the outer one and simply adds its own file to the
    existing tee instead of wrapping sys.stdout a second time.
    """
    def __init__(self, log_path, also_print_banner=True, append=False, experiment_tag=None):
        self.experiment_tag = experiment_tag
        self.log_path = self._resolve_path(log_path, append)
        self.also_print_banner = also_print_banner
        self.append = append
        self._log_file = None
        self._stdout_orig = None
        self._stderr_orig = None
        # Whether __enter__ decided to reuse an already-active outer tee
        # instead of installing a fresh one (nested-logger case).
        self._nested = False

    @staticmethod
    def _resolve_path(log_path, append):
        # Create any missing parent directories (e.g. a fresh
        # TSFNET_PAPER_EXACT_nback_20SEG_ABLATION folder that doesn't exist
        # yet on first run) -- previously this would crash on open() below.
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if append or not os.path.exists(log_path):
            return log_path
        base, ext = os.path.splitext(log_path)
        i = 2
        while os.path.exists(f"{base}_{i}{ext}"):
            i += 1
        new_path = f"{base}_{i}{ext}"
        print(f"NOTE: '{log_path}' already exists; writing this run to '{new_path}' instead "
              f"(pass append=True to CrashSafeLogger to append onto the existing file instead).")
        return new_path

    class _Tee:
        """
        Writes to multiple underlying file-like objects, flushing after every
        completed line so nothing sits in an in-memory buffer that a crash
        could lose. One of the streams (the real console/notebook stream) is
        treated as "live" and receives every write immediately, unchanged.
        The rest (log files) are treated as "logged" streams that receive
        full lines only: a "\r"-updated status line (tqdm-style progress
        bars, or any print(..., end="\r")) still updates live on the console,
        but only its final state -- once superseded by "\n" or a different
        line -- is written to the log file, instead of every intermediate
        tick. This keeps long training runs with per-batch progress output
        from bloating the log file or slowing every batch down with disk
        flushes.
        """
        def __init__(self, live_stream, logged_streams):
            self.live_stream = live_stream
            self.logged_streams = list(logged_streams)
            self._pending_line = ""

        def add_logged_stream(self, stream):
            self.logged_streams.append(stream)

        def write(self, data):
            # Live stream (console / notebook cell) always gets the raw data
            # immediately, exactly as if no logger were installed.
            self.live_stream.write(data)
            self.live_stream.flush()

            # Logged streams (files on disk) get complete lines only.
            self._pending_line += data
            while True:
                nl = self._pending_line.find("\n")
                cr = self._pending_line.find("\r")
                if nl == -1 and cr == -1:
                    break
                if nl != -1 and (cr == -1 or nl < cr):
                    line, self._pending_line = self._pending_line[:nl + 1], self._pending_line[nl + 1:]
                else:
                    # A "\r" without an earlier "\n" means an in-place progress
                    # update -- only the final tick before a real newline (or
                    # end of run, via flush()) actually gets written to disk.
                    line, self._pending_line = self._pending_line[:cr] + "\n", self._pending_line[cr + 1:]
                for s in self.logged_streams:
                    s.write(line)
                    s.flush()

        def flush(self):
            # Called at logger exit (and by anything that explicitly flushes)
            # so a final in-progress line (no trailing newline yet) still
            # makes it to disk instead of being silently dropped.
            if self._pending_line:
                for s in self.logged_streams:
                    s.write(self._pending_line)
                    s.flush()
                self._pending_line = ""
            self.live_stream.flush()
            for s in self.logged_streams:
                s.flush()

        # Needed so libraries that check for these (tqdm, some loggers) don't break
        def isatty(self):
            return False

    def __enter__(self):
        self._log_file = open(self.log_path, "a", encoding="utf-8", buffering=1)  # line-buffered

        self._stdout_orig = sys.stdout
        self._stderr_orig = sys.stderr

        # Nesting-safe: if sys.stdout is already one of our _Tee objects (an
        # outer CrashSafeLogger is active), don't wrap it again -- just add
        # this logger's file to the existing tee. This avoids double/triple
        # duplicated output when, e.g., run_all_subjects.py's outer logger is
        # already active and code called from inside tsfnet_model_notebook.py
        # opens its own CrashSafeLogger too.
        if isinstance(sys.stdout, self._Tee):
            self._nested = True
            sys.stdout.add_logged_stream(self._log_file)
            if isinstance(sys.stderr, self._Tee):
                sys.stderr.add_logged_stream(self._log_file)
        else:
            sys.stdout = self._Tee(self._stdout_orig, [self._log_file])
            sys.stderr = self._Tee(self._stderr_orig, [self._log_file])

        if self.also_print_banner:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tag_suffix = f" [{self.experiment_tag}]" if self.experiment_tag else ""
            nested_note = " (nested inside an already-active CrashSafeLogger)" if self._nested else ""
            print(f"===== CrashSafeLogger started{tag_suffix} {stamp} -> {self.log_path}{nested_note} =====")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Write the full traceback to the log file explicitly, in addition
            # to whatever the console/notebook's own exception display does --
            # this guarantees it's captured even if IPython's rich traceback
            # formatting, or a plain script's default excepthook, doesn't go
            # through plain sys.stdout/sys.stderr the same way.
            print("\n===== EXCEPTION -- run did not complete =====")
            traceback.print_exception(exc_type, exc_val, exc_tb, file=sys.stdout)

        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.also_print_banner:
            tag_suffix = f" [{self.experiment_tag}]" if self.experiment_tag else ""
            print(f"===== CrashSafeLogger stopped{tag_suffix} {stamp} =====")

        # Make sure any still-buffered partial line (e.g. a progress line with
        # no trailing newline yet) is flushed to disk before this logger's
        # file handle closes.
        if isinstance(sys.stdout, self._Tee):
            sys.stdout.flush()

        if self._nested:
            # An outer CrashSafeLogger is still supposed to be active --
            # only remove OUR file from the shared tee and close it; do NOT
            # restore the real console streams, that's the outer logger's job.
            if isinstance(sys.stdout, self._Tee) and self._log_file in sys.stdout.logged_streams:
                sys.stdout.logged_streams.remove(self._log_file)
            if isinstance(sys.stderr, self._Tee) and self._log_file in sys.stderr.logged_streams:
                sys.stderr.logged_streams.remove(self._log_file)
        else:
            sys.stdout = self._stdout_orig
            sys.stderr = self._stderr_orig

        self._log_file.close()
        # Returning False re-raises the exception (if any) as normal, so a
        # notebook cell still shows its usual red traceback and a standalone
        # script still exits with its usual non-zero status + traceback on
        # the real console -- this context manager only adds logging, it
        # never swallows errors.
        return False
