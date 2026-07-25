import queue
import threading
import time
import unittest

from src.feedback.audio_feedback import AudioFeedback


class FakeTTSEngine:
    """Records spoken messages, and the thread it was used from, instead of
    real speech -- lets us assert the same-thread creation fix directly."""

    def __init__(self):
        self.spoken: "queue.Queue[str]" = queue.Queue()
        self.creation_thread_id = threading.get_ident()

    def say(self, message: str) -> None:
        self.spoken.put(message)

    def runAndWait(self) -> None:
        pass


def _drain(q: "queue.Queue[str]", expected_count: int, timeout: float = 2.0) -> list[str]:
    items = []
    deadline = time.monotonic() + timeout
    while len(items) < expected_count and time.monotonic() < deadline:
        try:
            items.append(q.get(timeout=0.1))
        except queue.Empty:
            continue
    return items


class TestAudioFeedback(unittest.TestCase):
    def setUp(self):
        # A factory (not a pre-built instance) -- the fix requires the
        # engine to be created inside the worker thread itself.
        self.created_engines: list[FakeTTSEngine] = []

        def factory() -> FakeTTSEngine:
            engine = FakeTTSEngine()
            self.created_engines.append(engine)
            return engine

        self.factory = factory
        self.audio = AudioFeedback(engine_factory=factory, debounce_seconds=0.3)

    def tearDown(self):
        self.audio.close()

    def test_engine_is_created_on_the_worker_thread_not_main_thread(self):
        # This is the direct regression test for the threading bug: the
        # engine must be created on a DIFFERENT thread than this test
        # (which runs on the main thread), proving construction happened
        # inside the worker, not in AudioFeedback.__init__.
        self.assertEqual(len(self.created_engines), 1)
        engine = self.created_engines[0]
        self.assertNotEqual(engine.creation_thread_id, threading.get_ident())

    def test_announce_reaches_engine(self):
        self.audio.announce("Keep your back more upright")
        spoken = _drain(self.created_engines[0].spoken, 1)
        self.assertEqual(spoken, ["Keep your back more upright"])

    def test_duplicate_message_within_debounce_window_is_suppressed(self):
        self.audio.announce("Keep your knee behind your toes")
        self.audio.announce("Keep your knee behind your toes")
        self.audio.announce("Keep your knee behind your toes")
        spoken = _drain(self.created_engines[0].spoken, 1, timeout=0.5)
        self.assertEqual(len(spoken), 1)

    def test_message_repeats_after_debounce_window_passes(self):
        self.audio.announce("Good form")
        _drain(self.created_engines[0].spoken, 1)
        time.sleep(0.4)
        self.audio.announce("Good form")
        spoken = _drain(self.created_engines[0].spoken, 1)
        self.assertEqual(len(spoken), 1)

    def test_different_messages_both_reach_engine(self):
        self.audio.announce("Keep your knee behind your toes")
        self.audio.announce("Keep your back more upright")
        spoken = _drain(self.created_engines[0].spoken, 2)
        self.assertEqual(
            set(spoken),
            {"Keep your knee behind your toes", "Keep your back more upright"},
        )

    def test_announce_does_not_block_caller(self):
        start = time.monotonic()
        self.audio.announce("Good form")
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 0.05)

    def test_missing_engine_disables_audio_without_crashing(self):
        def failing_factory():
            raise RuntimeError("no audio backend available")

        audio = AudioFeedback(engine_factory=failing_factory, debounce_seconds=0.1)
        self.assertFalse(audio.enabled)
        audio.announce("this should be a safe no-op")  # must not raise
        audio.close()

    def test_engine_that_fails_mid_speech_does_not_crash_worker(self):
        class FlakyEngine:
            def say(self, message):
                raise RuntimeError("simulated driver failure")

            def runAndWait(self):
                pass

        audio = AudioFeedback(engine_factory=FlakyEngine, debounce_seconds=0.1)
        audio.announce("this will fail internally")
        time.sleep(0.3)
        # Worker thread must still be alive (didn't crash) and still usable
        self.assertTrue(audio._worker.is_alive())
        audio.close()


if __name__ == "__main__":
    unittest.main()
