"""
templates.py

Contains the instruction template and output template
used to generate instruction-following training examples
for Week 3 of the internship project.
"""

SYSTEM_PROMPT = """You are an expert log analysis assistant.

Analyze the provided log entries and respond with:

1. SEVERITY: [CRITICAL | HIGH | MEDIUM | LOW | INFO]
2. INCIDENT_TYPE: [brief category]
3. ROOT_CAUSE: [most likely root cause]
4. SUMMARY: [2-3 sentence plain-English summary]
5. RECOMMENDED_ACTIONS: [numbered list of immediate steps]

Be concise and precise.
Do not include information not supported by the logs.
"""


OUTPUT_TEMPLATE = """SEVERITY: {severity}
INCIDENT_TYPE: {incident_type}
ROOT_CAUSE: {root_cause}
SUMMARY: {summary}
RECOMMENDED_ACTIONS:
{recommended_actions}
"""


def format_output(
    severity: str,
    incident_type: str,
    root_cause: str,
    summary: str,
    recommended_actions: list[str],
) -> str:
    """
    Formats the expected model output into the
    instruction-following format required by the dataset.
    """

    actions = "\n".join(
        f"{i + 1}. {action}"
        for i, action in enumerate(recommended_actions)
    )

    return OUTPUT_TEMPLATE.format(
        severity=severity,
        incident_type=incident_type,
        root_cause=root_cause,
        summary=summary,
        recommended_actions=actions,
    )