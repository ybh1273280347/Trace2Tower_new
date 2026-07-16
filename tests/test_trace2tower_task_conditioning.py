from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.methods.trace2tower.alfworld_task_adapter import AlfworldTaskAdapter
from trace2tower.methods.trace2tower.skills import HighSkillCard
from trace2tower.methods.trace2tower.task_conditioning import (
    TaskCompatibility,
    TaskCondition,
    TaskConditionProfile,
    TaskProperty,
    SkillTaskCondition,
    retrieve_task_conditioned_high,
)
from trace2tower.methods.trace2tower.webshop_task_adapter import WebshopTaskAdapter
from trace2tower.semantic_index import SkillEmbeddingIndex


def _state(observation: str = "") -> EnvironmentState:
    return EnvironmentState(observation, (), {}, False, 0.0, False, True)


class CalendarTaskAdapter:
    domain = "calendar"

    def extract_query(self, task_goal: str, state: EnvironmentState) -> TaskCondition:
        return TaskCondition(task_goal, task_goal, ("SCHEDULE",), ())

    def profile_condition(self, record: dict) -> TaskCondition:
        task_text = str(record["task_text"])
        return TaskCondition(task_text, task_text, ("SCHEDULE",), ())

    def compatibility(
        self,
        query: TaskCondition,
        candidate: TaskCondition,
    ) -> TaskCompatibility:
        return (
            TaskCompatibility.EXACT
            if query.values("calendar") == candidate.values("calendar")
            else TaskCompatibility.INCOMPATIBLE
        )

    def bind(
        self,
        source_card: HighSkillCard,
        query: TaskCondition,
        candidate: TaskCondition,
    ) -> HighSkillCard:
        return source_card


def test_core_accepts_an_unregistered_dataset_adapter() -> None:
    index = SkillEmbeddingIndex(
        ("personal", "team"),
        ((1.0, 0.0), (0.8, 0.6)),
    )
    query = TaskCondition(
        "schedule team review",
        "schedule team review",
        ("SCHEDULE",),
        (TaskProperty("calendar", ("team",)),),
    )
    conditions = {
        "personal": TaskCondition(
            "personal event",
            "personal event",
            ("SCHEDULE",),
            (TaskProperty("calendar", ("personal",)),),
        ),
        "team": TaskCondition(
            "team event",
            "team event",
            ("SCHEDULE",),
            (TaskProperty("calendar", ("team",)),),
        ),
    }
    selected = retrieve_task_conditioned_high(
        (1.0, 0.0),
        index,
        query,
        conditions,
        CalendarTaskAdapter(),
        minimum_compatibility=TaskCompatibility.EXACT,
    )
    assert selected is not None
    assert selected.semantic_match.skill_id == "team"


def test_alfworld_adapter_rejects_semantically_closer_wrong_event() -> None:
    adapter = AlfworldTaskAdapter()
    index = SkillEmbeddingIndex(
        ("cool_mug", "clean_mug"),
        ((1.0, 0.0), (0.8, 0.6)),
    )
    query = adapter.extract_query(
        "external title",
        _state("Your task is to: clean some mug and put it in coffeemachine.\n"),
    )
    conditions = {
        "cool_mug": adapter.profile_condition(
            {
                "canonical_goal": "cool some mug and put it in coffeemachine.",
                "target_object": "mug",
                "transformation": "COOL",
                "destination_receptacles": ("coffeemachine",),
                "transformation_devices": ("fridge",),
            }
        ),
        "clean_mug": adapter.profile_condition(
            {
                "canonical_goal": "clean some mug and put it in coffeemachine.",
                "target_object": "mug",
                "transformation": "CLEAN",
                "destination_receptacles": ("coffeemachine",),
                "transformation_devices": ("sinkbasin",),
            }
        ),
    }
    selected = retrieve_task_conditioned_high(
        (1.0, 0.0),
        index,
        query,
        conditions,
        adapter,
        minimum_compatibility=TaskCompatibility.EXACT,
    )
    assert selected is not None
    assert selected.semantic_match.skill_id == "clean_mug"


def test_webshop_adapter_binds_the_current_constraints() -> None:
    adapter = WebshopTaskAdapter()
    query = adapter.extract_query(
        "Buy a red leather handbag under $50",
        _state("WebShop search page."),
    )
    candidate = adapter.profile_condition(
        {"task_text": "Buy a blue leather handbag under $60"}
    )
    assert adapter.compatibility(query, candidate) is TaskCompatibility.PARTIAL

    card = adapter.bind(
        HighSkillCard(
            "high_shop",
            ("mid_search", "mid_buy"),
            "Search and verify",
            "Verify a product before buying.",
            ("Search for the product.", "Buy after verification."),
        ),
        query,
        candidate,
    )
    assert "red leather handbag under $50" in card.description
    assert "red leather handbag under $50" in card.procedure[0]


def test_task_condition_round_trip_preserves_open_domain_properties() -> None:
    condition = TaskCondition(
        "deploy service",
        "deploy service",
        ("DEPLOY",),
        (TaskProperty("region", ("ap-southeast-1",)),),
    )
    assert TaskCondition.from_record(condition.to_record()) == condition
    profile = TaskConditionProfile(
        "deployment",
        (SkillTaskCondition("skill", condition),),
    )
    assert TaskConditionProfile.from_record(profile.to_record()) == profile
