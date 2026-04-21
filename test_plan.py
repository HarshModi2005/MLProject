from orchestrator.state import UserIntent, OutreachGoal
from agents.planning_agent import PlanningAgent

intent = UserIntent(
    goal=OutreachGoal.RESEARCH_INTERNSHIP,
    research_domains=["Game Theory", "Machine Learning"],
    countries=["Punjab, India"]
)
agent = PlanningAgent()
plan = agent.generate_plan(intent)
agent.display_plan(plan)
