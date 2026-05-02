from src.models import ClassifierOutput, StubResponse


def run_stub(classifier_output: ClassifierOutput) -> StubResponse:
    return StubResponse(
        intent=classifier_output.intent,
        entities=classifier_output.entities,
        target_agent=classifier_output.target_agent,
        message=f"The '{classifier_output.target_agent.value}' agent is not implemented in this build.",
    )
