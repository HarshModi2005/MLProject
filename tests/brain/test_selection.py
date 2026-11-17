"""Wider but safer candidate selection: unknown locations kept (not "elsewhere"), page titles rejected as people,
guesses only where conventions are reliable, job boards treated as directories, requested sources first."""

from brain.discovery.reach import patterns
from brain.models import CampaignSpec, Contact, Evidence, FitCriterion, ProspectProfile, SenderProfile, TargetPersona
from brain.pipeline import plausible_person
from brain.playbooks import get_playbook
from brain.rank import filter_contacts, prescore


def _spec(**kw):
    base = dict(id="sel", pack="speakers", objective="Speakers for a student AI summit", persona=TargetPersona(description="ML speakers"),
                topics=["machine learning"], countries=["IN"], regions=["Bangalore"], target_count=5,
                sender=SenderProfile(name="T"))
    base.update(kw)
    return CampaignSpec(**base)


def test_people_with_unknown_location_are_kept_below_known_locals():
    pb = get_playbook("speakers")
    local = Contact(name="Asha Rao", country="IN", location_text="Bengaluru, Karnataka", bio="machine learning researcher")
    unknown = Contact(name="Ravi Iyer", country="IN", bio="machine learning researcher")
    elsewhere = Contact(name="Meena Das", country="IN", location_text="Kolkata", bio="machine learning researcher")
    kept, dropped = filter_contacts([local, unknown, elsewhere], _spec(), pb)
    assert [c.name for c in kept] == ["Asha Rao", "Ravi Iyer"] and dropped == {"region": 1}
    assert "region_unknown" in unknown.flags and "region_unknown" not in local.flags
    ranked = prescore(kept, _spec(), pb)
    assert ranked[0].name == "Asha Rao" and ranked[1].prescore < ranked[0].prescore
    unknown.location_text = "Kolkata, West Bengal"  # learned during enrichment: now known to be elsewhere
    kept, dropped = filter_contacts([unknown], _spec(), pb)
    assert kept == [] and dropped == {"region": 1}


def test_page_titles_are_not_people_but_handles_are_allowed():
    assert not plausible_person(Contact(kind="person", name="Partner Spotlight", sources=["orgs", "decision_maker"]))
    assert not plausible_person(Contact(kind="person", name="Customer Stories", sources=["orgs"]))
    assert plausible_person(Contact(kind="person", name="Savithri Parekh", sources=["orgs"]))
    assert plausible_person(Contact(kind="person", name="voxlab", sources=["huggingface"]))
    assert plausible_person(Contact(kind="org", name="Partner Spotlight"))


def test_no_conventional_guesses_for_journalists_or_outlets(monkeypatch):
    monkeypatch.setattr(patterns, "has_mx", lambda d: True)
    journo = Contact(kind="person", name="Supriya Roy", title="Journalist", org="Times of India", org_domain="indiatimes.com")
    assert patterns.guess(journo, None, accept_prior=True) is None
    founder = Contact(kind="person", name="Kiran Rao", title="Founder", org="Voxa", org_domain="voxa.ai")
    assert patterns.guess(founder, None, accept_prior=True).address == "kiran@voxa.ai"
    board = Contact(kind="person", name="Kai Brandt", title="Founder", org="Top Startups", org_domain="topstartups.io")
    assert patterns.guess(board, None, accept_prior=True) is None  # a job board is a directory


def test_job_boards_and_lists_are_directories():
    from brain.discovery.precision.orgtype import classify

    k = lambda d: classify(Contact(kind="org", name=d, org_domain=d), lambda x: False)[0]  # noqa: E731
    assert k("topstartups.io") == "directory" and k("aijobs.net") == "directory" and k("startupslist.in") == "directory"
    assert k("jobberwocky.ai") == "company" and k("voxa.ai") == "company"


def test_sources_a_criterion_asks_for_run_first():
    from brain.discovery.router import choose
    from brain.fit import profile_for

    pb = get_playbook("b2b_sales")
    spec = _spec(pack="b2b_sales", target_kind="org", regions=[], profile=ProspectProfile(
        criteria=[FitCriterion(name="publishes apps", signal="app_publisher")]))
    seed, _ = choose(spec, pb, profile_for(spec, pb), has_seeds=False)
    assert seed[:2] == ["memory", "google_play"]
