from orchestrator.state import UserIntent, OutreachGoal
from agents.planning_agent import PlanningAgent
import json
intent = UserIntent(
    goal=OutreachGoal.RESEARCH_INTERNSHIP,
    research_domains=["Game Theory", "Machine Learning"],
    countries=["Punjab, India"],
    timeline="Summer 2026",
    user_profile_summary="CS Student",
    outreach_per_week=10,
    follow_up_days=7
)
agent = PlanningAgent()
plan = agent.generate_plan(intent)
print("COUNTRIES IN PLAN:")
print(plan.search_strategy.country_filters)
