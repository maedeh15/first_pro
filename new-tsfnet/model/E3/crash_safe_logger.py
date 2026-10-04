# ============================================================================
# Crash-safe output logging for Jupyter.
#
# Duplicates EVERY print() (and any error traceback) to BOTH the notebook
# cell's output AND a plain-text file on disk, flushing to disk after every
# single write. This means: if the kernel crashes, the cell is interrupted,
# or an exception kills the run partway through, everything printed UP TO
# that point is already safely on disk -- nothing is lost, because nothing
# is buffered and written only "at the end".
#
# UPDATED FOR THE BASELINE / ABLATION MULTI-EXPERIMENT ARCHITECTURE:
# This project now has multiple parallel experiment folders that must never
# mix logs (e.g. TSFNET_PAPER_EXACT_nback for the 11-segment baseline vs.
# TSFNET_PAPER_EXACT_nback_20SEG_ABLATION for the 20-segment ablation). Two
# changes support this cleanly:
#   1. CrashSafeLogger now creates any missing parent directories for
#      log_path automatically (previously, pointing it at a not-yet-created
#      experiment subfolder would crash on open()).
#   2. A new optional `experiment_tag` parameter lets you label WHICH
#      experiment a log belongs to directly in the banner/filename, so when
#      you're flipping between run_all_subjects.py (baseline) and
#      run_all_subjects_ABLATION_20seg.py in the same Jupyter session, it's
#      immediately obvious from the log itself -- and from its default
#      location -- which experiment it came from, without you having to
#      remember to change the filename by hand every time.
#
# USAGE (unchanged for simple cases):
#
#     with CrashSafeLogger("VP001_training_log.txt") as log:
#         result = run_subject_cross_session(...)
#
# USAGE (experiment-aware, recommended for this project):
#
#     with CrashSafeLogger(
#         os.path.join(DATA_ROOT, f"baseline_run_log_{RUN_TAG}.txt"),
#         experiment_tag="11SEG_BASELINE",
#     ) as log:
#         ...
#
#     with CrashSafeLogger(
#         os.path.join(DATA_ROOT, f"baseline_run_log_{RUN_TAG}.txt"),
#         experiment_tag="20SEG_ABLATION",
#     ) as log:
#         ...
#
# Everything printed inside the `with` block (training epochs, diagnostics
# progress, etc.) appears in the notebook cell AS NORMAL, and simultaneously
# accumulates in the log file in real time. If an exception occurs inside
# the block, its full traceback is also written to the log file before being
# re-raised (so you still see the normal Jupyter error display, but the
# traceback is ALSO safely on disk).
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
    """
    def __init__(self, log_path, also_print_banner=True, append=False, experiment_tag=None):
        self.experiment_tag = experiment_tag
        self.log_path = self._resolve_path(log_path, append)
        self.also_print_banner = also_print_banner
        self.append = append
        self._log_file = None
        self._stdout_orig = None
        self._stderr_orig = None

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
        """Writes to multiple streams; flushes to disk after every write so
        nothing sits in an in-memory buffer that a crash could lose."""
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

        # Needed so libraries that check for these (tqdm, some loggers) don't break
        def isatty(self):
            return False

    def __enter__(self):
        self._log_file = open(self.log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
        self._stdout_orig = sys.stdout
        self._stderr_orig = sys.stderr
        sys.stdout = self._Tee(self._stdout_orig, self._log_file)
        sys.stderr = self._Tee(self._stderr_orig, self._log_file)
        if self.also_print_banner:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tag_suffix = f" [{self.experiment_tag}]" if self.experiment_tag else ""
            print(f"===== CrashSafeLogger started{tag_suffix} {stamp} -> {self.log_path} =====")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Write the full traceback to the log file explicitly, in addition to
            # whatever Jupyter's own exception display does -- this guarantees
            # it's captured even if IPython's rich traceback formatting doesn't
            # go through plain sys.stderr the same way a normal script's would.
            print("\n===== EXCEPTION -- run did not complete =====")
            traceback.print_exception(exc_type, exc_val, exc_tb, file=sys.stdout)

        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.also_print_banner:
            tag_suffix = f" [{self.experiment_tag}]" if self.experiment_tag else ""
            print(f"===== CrashSafeLogger stopped{tag_suffix} {stamp} =====")

        sys.stdout = self._stdout_orig
        sys.stderr = self._stderr_orig
        self._log_file.close()
        # Returning False re-raises the exception (if any) as normal, so Jupyter
        # still shows its usual red traceback -- this context manager only adds
        # logging, it never swallows errors.
        return False
