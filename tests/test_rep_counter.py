import unittest

from src.core.rep_counter import RepCounter


class TestRepCounter(unittest.TestCase):
    def setUp(self):
        # e.g. a squat: knee angle >160 = "up" (standing), <90 = "down" (squatting)
        self.counter = RepCounter(up_threshold=160, down_threshold=90)

    def test_one_clean_rep(self):
        sequence = [170, 150, 100, 80, 70, 100, 150, 170]
        events = [self.counter.update(a) for a in sequence]
        self.assertEqual(events[-1].count, 1)

    def test_starting_in_down_position_does_not_prematurely_count(self):
        # If you start already squatted, going up once should count exactly one rep
        sequence = [80, 90, 170]
        events = [self.counter.update(a) for a in sequence]
        self.assertEqual(events[-1].count, 1)

    def test_noisy_angles_near_threshold_do_not_overcount(self):
        # Angle oscillates right around the up_threshold without a real down phase
        sequence = [161, 159, 162, 158, 161, 160, 159]
        events = [self.counter.update(a) for a in sequence]
        self.assertEqual(events[-1].count, 0)

    def test_multiple_reps_count_correctly(self):
        one_rep = [170, 80, 170]
        sequence = one_rep * 3
        # Inject increasing timestamps 0.5s apart to simulate realistic
        # rep timing (this test isn't testing debounce, so time must
        # clearly exceed min_rep_interval_seconds between each update).
        events = [
            self.counter.update(a, now=i * 0.5) for i, a in enumerate(sequence)
        ]
        self.assertEqual(events[-1].count, 3)

    def test_partial_movement_without_reaching_down_threshold_does_not_count(self):
        sequence = [170, 130, 170]  # never actually gets below down_threshold
        events = [self.counter.update(a) for a in sequence]
        self.assertEqual(events[-1].count, 0)

    def test_invalid_thresholds_raise(self):
        with self.assertRaises(ValueError):
            RepCounter(up_threshold=90, down_threshold=160)

    def test_reset(self):
        self.counter.update(170)
        self.counter.update(80)
        self.counter.update(170)
        self.assertEqual(self.counter.update(170).count, 1)
        self.counter.reset()
        self.assertEqual(self.counter._count, 0)
        self.assertEqual(self.counter._state, "unknown")

    def test_rapid_reps_faster_than_min_interval_are_discarded(self):
        # Simulate jitter: down -> up -> down -> up happening within
        # milliseconds (impossible for a real human rep).
        t = 0.0
        self.counter.update(170, now=t)       # up
        self.counter.update(80, now=t + 0.01)  # down
        self.counter.update(170, now=t + 0.02) # up -> 1st legit rep, count=1
        self.counter.update(80, now=t + 0.03)  # down again, way too fast
        event = self.counter.update(170, now=t + 0.04)  # up again, too fast
        self.assertEqual(event.count, 1)  # second "rep" discarded, too fast

    def test_reps_spaced_beyond_min_interval_both_count(self):
        t = 0.0
        self.counter.update(170, now=t)
        self.counter.update(80, now=t + 0.1)
        self.counter.update(170, now=t + 0.2)  # rep 1
        self.counter.update(80, now=t + 0.5)
        event = self.counter.update(170, now=t + 0.8)  # rep 2, spaced enough
        self.assertEqual(event.count, 2)

    def test_current_event_does_not_change_state(self):
        self.counter.update(170)
        self.counter.update(80)
        before = self.counter.current_event()
        after = self.counter.current_event()
        self.assertEqual(before, after)
        self.assertEqual(before.count, 0)


if __name__ == "__main__":
    unittest.main()
