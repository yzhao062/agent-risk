import pytest

from agent_risk import Episode

from toydata import TOY_COST, TOY_SCORES


@pytest.fixture
def toy_episodes():
    """Two episodes stopped at step 2 and one that never exceeds 0.94.

    Episode 0 has harm at step 1 (before the stop, so it counts).
    Episode 1 has harm at step 3 (after the stop, so the replay removes it).
    Episode 2 agrees with the log.
    """
    return [
        Episode(TOY_SCORES, [False, True, False, False], TOY_COST, failed=False),
        Episode(TOY_SCORES, [False, False, False, True], TOY_COST, failed=False),
        Episode([0.10, 0.20], [False, False], [10.0, 20.0], failed=False),
    ]
