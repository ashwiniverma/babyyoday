import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.planner import Planner


def test_single_question():
    planner = Planner()
    tasks = planner.plan("Do you have vegan options?")
    assert len(tasks) == 1
    assert tasks[0].query == "Do you have vegan options?"


def test_compound_question():
    planner = Planner()
    tasks = planner.plan("Do you have vegan options? Are there nut-free cakes?")
    assert len(tasks) == 2


def test_conjunction_split():
    planner = Planner()
    tasks = planner.plan("Show me the menu and also what are the opening hours")
    assert len(tasks) == 2
