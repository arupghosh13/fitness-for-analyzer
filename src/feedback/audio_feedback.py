"""Non-blocking audio feedback with debouncing so identical messages don't
spam every frame.

IMPORTANT: pyttsx3 (and its underlying OS TTS drivers, e.g. SAPI5 on
Windows) must be created AND used on the same thread -- creating the
engine on one thread and calling it from another causes it to silently
produce no audio on many systems, with no exception raised. This is why
the engine is created inside the worker thread's run loop (`_run`), not
in `__init__`. An `engine_factory` (not a ready-made engine instance) is
injected so tests can supply a fake without hitting this same constraint.

If no TTS engine is available (missing dependency, or no audio backend on
the host OS/container), audio is disabled gracefully rather than crashing
the whole app -- consistent with how PoseDetector handles its missing
model file.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional, Protocol

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TTSEngine(Protocol):
    def say(self, message: str) -> None: ...
    def runAndWait(self) -> None: ...


class AudioFeedback:
    def __init__(
        self,
        engine_factory: Optional[Callable[[], TTSEngine]] = None,
        debounce_seconds: float = 3.0,
        init_timeout_seconds: float = 3.0,
    ):
        self._engine_factory = engine_factory or self._default_engine_factory
        self._debounce_seconds = debounce_seconds
        self._last_spoken: dict[str, float] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self.enabled = False  # set by the worker thread once it knows

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

        # Block briefly so callers (and tests) can rely on `enabled` being
        # correct immediately after construction, without waiting on the
        # full engine init to happen asynchronously and unobserved.
        self._ready_event.wait(timeout=init_timeout_seconds)

    @staticmethod
    def _default_engine_factory() -> TTSEngine:
        import pyttsx3  # local import: only required if no factory is injected

        return pyttsx3.init()

    def announce(self, message: str) -> None:
        """Queue a message to be spoken, unless it was spoken too recently.

        No-op if audio is disabled (missing engine) -- callers don't need
        to check `enabled` themselves before calling this.
        """
        if not self.enabled:
            return

        now = time.monotonic()
        last_time = self._last_spoken.get(message)
        if last_time is not None and (now - last_time) < self._debounce_seconds:
            return  # debounced: skip duplicate
        self._last_spoken[message] = now
        self._queue.put(message)

    def _run(self) -> None:
        try:
            engine = self._engine_factory()
        except Exception as exc:  # missing package OR missing OS audio backend
            logger.warning(
                "Audio feedback disabled: TTS engine unavailable (%s). "
                "The app will continue with visual feedback only.", exc,
            )
            self.enabled = False
            self._ready_event.set()
            return

        self.enabled = True
        self._ready_event.set()

        while not self._stop_event.is_set():
            try:
                message = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                engine.say(message)
                engine.runAndWait()
            except Exception as exc:
                # A single failed utterance shouldn't kill the whole audio
                # feature -- log it and keep processing the queue.
                logger.warning("Text-to-speech playback failed: %s", exc)

    def close(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=2.0)
