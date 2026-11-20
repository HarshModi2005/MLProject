"""Offline unit tests for the brain's deterministic parts (no network, no LLM)."""

from brain.emailfinder import (
    apply_patterns, attach_emails, extract_emails, infer_domain_patterns, is_role_address, name_match, score_email,
)
from brain.models import CampaignSpec, Contact, EmailCandidate, Evidence, SenderProfile
from brain.packs import get_pack
from brain.pages import contacts_from_page
from brain.rank import bm25_scores, diversify, filter_contacts, prescore
from brain.resolve import names_compatible, resolve
from brain.textutil import clean_name, looks_like_person_name, registrable_domain, split_locations

PROFILE = """<html><head><title>Dr. Sanjay Kumar | Department of CSE | IIT Ropar</title></head><body>
<nav><a href="/people">People</a><a href="mailto:office@iitrpr.ac.in">office</a></nav>
<h1>Dr. Sanjay Kumar</h1><p>Associate Professor, Computer Science</p>
<p>Email: sanjay [at] iitrpr [dot] ac [dot] in</p>
<p>Research Interests: graph neural networks, reinforcement learning, and formal verification</p>
<p>I lead the Graph Learning Lab where we study scalable message passing methods for molecular property
prediction and the theory of expressive power in graph neural networks, with students and collaborators.</p>
<footer>webmaster@iitrpr.ac.in</footer></body></html>"""


def directory_html(names):
    cards = "".join(
        f'<div class="card"><h3><a href="/faculty/{n.split()[0].lower()}">{n}</a></h3><p>Professor</p>'
        f'<a href="mailto:{n.split()[0].lower()}@iitrpr.ac.in">email</a></div>' for n in names)
    return f"<html><title>Faculty | IIT Ropar</title><body><h1>Faculty</h1>{cards}</body></html>"


NAMES = ["Asha Rani", "Vikram Singh", "Neha Gupta", "Rohit Sharma", "Priya Mehta", "Arjun Nair", "Kiran Rao",
         "Deepa Iyer", "Manoj Verma"]


# ── names, domains, locations ────────────────────────────────────────────────
def test_person_name_detection():
    for good in ["Prof. Sanjay Kumar", "Dr. A. P. J. Abdul Kalam", "Jean-Luc O'Neil", "Ludwig van Beethoven"]:
        assert looks_like_person_name(good), good
    for bad in ["Research Interests", "Contact Us", "Machine Learning", "Highlights", "Department of CSE", "Kumar"]:
        assert not looks_like_person_name(bad), bad
    assert clean_name("Prof. (Dr.) Sanjay K. Kumar, PhD") == "Sanjay K. Kumar"


def test_domains_and_locations():
    assert registrable_domain("https://cse.iitrpr.ac.in/x") == "iitrpr.ac.in"  # legacy code produced "AC"
    assert registrable_domain("www.cs.mit.edu") == "mit.edu"
    assert split_locations(["Punjab, India", "USA", "worldwide"]) == (["IN", "US"], ["Punjab"])


# ── emails ───────────────────────────────────────────────────────────────────
def test_email_deobfuscation_without_false_positives():
    text = "sk [at] iitrpr [dot] ac [dot] in and john.doe at cs dot mit dot edu; icon a@b.png"
    assert extract_emails(text) == ["sk@iitrpr.ac.in", "john.doe@cs.mit.edu"]
    assert extract_emails("We looked at results and dot products. Contact AT once.") == []


def test_email_scoring_prefers_the_persons_own_address():
    c = Contact(name="Prof. Sanjay Kumar", org_domain="iitrpr.ac.in")
    assert is_role_address("office@iitrpr.ac.in") and score_email("office@iitrpr.ac.in", c, "mailto", True) == 0
    own = score_email("sanjay@iitrpr.ac.in", c, "mailto", True)
    other = score_email("rahul@iitrpr.ac.in", c, "text", False)
    assert own >= 0.9 and other < 0.6
    assert name_match("Sanjay Kumar", "sk@x.org") == 0.5 and name_match("Sanjay Kumar", "rahul@x.org") == 0


def test_other_peoples_addresses_on_a_profile_are_demoted():
    c = Contact(name="Sanjay Kumar", org_domain="iitrpr.ac.in")
    attach_emails(c, ["sanjay@iitrpr.ac.in", "student.rahul@iitrpr.ac.in"], "mailto", "u", on_own_page=True)
    assert c.best_email.address == "sanjay@iitrpr.ac.in"
    assert [e for e in c.emails if e.address.startswith("student")][0].confidence < 0.6


def test_pattern_inference_is_flagged_and_low_confidence():
    known = [Contact(name=n, org_domain="acme.com",
                     emails=[EmailCandidate(address=f"{n.split()[0].lower()}.{n.split()[1].lower()}@acme.com",
                                            confidence=0.95, method="mailto")]) for n in NAMES[:4]]
    target = Contact(name="Tara Bose", org_domain="acme.com")
    patterns = infer_domain_patterns(known)
    assert patterns["acme.com"][0] == "first.last"
    assert apply_patterns([target], patterns) == 1
    assert target.best_email.address == "tara.bose@acme.com"
    assert target.best_email.confidence < 0.6 and "email_guessed" in target.flags


# ── pages ────────────────────────────────────────────────────────────────────
def test_profile_page():
    kind, people = contacts_from_page(PROFILE, "https://cse.iitrpr.ac.in/~sanjay/")
    assert kind == "profile" and len(people) == 1
    c = people[0]
    assert c.name == "Sanjay Kumar" and c.org == "IIT Ropar" and c.country == "IN"
    assert c.best_email.address == "sanjay@iitrpr.ac.in"  # not office@ (nav) or webmaster@ (footer)
    assert "graph neural networks" in c.topics and c.title == "Associate Professor"


def test_directory_page_yields_everyone_with_their_own_email():
    kind, people = contacts_from_page(directory_html(NAMES), "https://cse.iitrpr.ac.in/faculty")
    assert kind == "directory" and len(people) == len(NAMES)
    by_name = {c.name: c for c in people}
    assert by_name["Neha Gupta"].best_email.address == "neha@iitrpr.ac.in"
    assert by_name["Neha Gupta"].urls[0].endswith("/faculty/neha")


def test_non_people_page():
    kind, people = contacts_from_page("<html><title>News</title><body><h1>Latest News</h1></body></html>", "https://x.org/n")
    assert kind == "other" and people == []


# ── resolution ───────────────────────────────────────────────────────────────
def test_name_compatibility():
    assert names_compatible("S. Kumar", "Sanjay Kumar")
    assert names_compatible("Sanjay K. Kumar", "Sanjay Kumar")
    assert not names_compatible("Sanjay Kumar", "Sunil Kumar")
    assert not names_compatible("Kumar", "Sanjay Kumar")


def test_resolve_merges_same_person_across_sources_only():
    pubs = Contact(name="Sanjay Kumar", org="IIT Ropar", org_domain="iitrpr.ac.in", keys=["openalex:A1"],
                   evidence=[Evidence(kind="publication", text="Scalable GNNs")], sources=["openalex"])
    web = Contact(name="Dr. Sanjay Kumar", org_domain="iitrpr.ac.in", sources=["web"],
                  emails=[EmailCandidate(address="sanjay@iitrpr.ac.in", confidence=0.95, method="mailto")])
    namesake = Contact(name="Sanjay Kumar", org="IIT Delhi", org_domain="iitd.ac.in", sources=["web"])
    out = resolve([pubs, web, namesake])
    assert len(out) == 2
    merged = next(c for c in out if c.org_domain == "iitrpr.ac.in")
    assert merged.best_email.address == "sanjay@iitrpr.ac.in" and merged.evidence and set(merged.sources) == {"openalex", "web"}
    assert len({c.id for c in out}) == 2 and all(c.id for c in out)


# ── filtering and ranking ────────────────────────────────────────────────────
def _spec(**kw):
    base = dict(id="t", pack="academic", topics=["graph neural networks"], countries=["IN"], regions=["Punjab"],
                sender=SenderProfile(name="A"), target_count=5)
    base.update(kw)
    return CampaignSpec(**base)


def test_hard_filters():
    pack = get_pack("academic")
    ok = Contact(name="A B", country="IN", location_text="Rupnagar, Punjab, India", signals={"seniority": 0.8},
                 org_domain="iitrpr.ac.in")
    wrong_country = Contact(name="C D", country="US", location_text="Boston")
    wrong_region = Contact(name="E F", country="IN", location_text="Chennai, Tamil Nadu, India")
    web_unverified = Contact(name="K L", country="IN", org_domain="mnit.ac.in")  # web find, location unknown
    web_vouched = Contact(name="M N", country="IN", org_domain="iitrpr.ac.in")
    student = Contact(name="G H", country="IN", location_text="Punjab", signals={"seniority": 0.1})
    blocked = Contact(name="I J", country="IN", location_text="Punjab")
    kept, dropped = filter_contacts([ok, wrong_country, wrong_region, student, blocked, web_unverified, web_vouched],
                                    _spec(), pack, is_blocked=lambda c: "suppressed" if c.name == "I J" else None)
    # M N is vouched for by A B's confirmed organisation; K L's city is unknown (not known to be elsewhere), so they
    # stay, flagged for the pre-judge and the judge, and ranked below people known to be local.
    assert [c.name for c in kept] == ["A B", "K L", "M N"]
    assert "region_unknown" in web_unverified.flags and "region_unknown" not in web_vouched.flags
    assert dropped == {"country": 1, "region": 1, "seniority": 1, "suppressed": 1}


def test_prescore_orders_by_relevance():
    relevant = Contact(name="A B", evidence=[Evidence(kind="publication", text="Graph neural networks for molecules")],
                       signals={"matched_works": 3, "senior_positions": 2, "seniority": 0.8, "latest_year": 2026})
    unrelated = Contact(name="C D", evidence=[Evidence(kind="publication", text="Soil chemistry of wheat fields")],
                        signals={"matched_works": 1, "seniority": 0.8, "latest_year": 2026})
    ranked = prescore([unrelated, relevant], _spec(), get_pack("academic"))
    assert ranked[0].name == "A B" and ranked[0].prescore > ranked[1].prescore
    assert bm25_scores("graph networks", ["graph network", "wheat"]) == [1.0, 0.0]


def test_diversify_caps_per_organisation():
    people = [Contact(id=str(i), name=f"P {i}", org_domain="a.edu" if i < 5 else f"o{i}.edu") for i in range(8)]
    out = diversify(people, 5, max_per_org=2)
    assert [c.id for c in out] == ["0", "1", "5", "6", "7"]


def test_name_match_uses_whole_parts_not_substrings():
    assert name_match("Manoj Singh Gaur", "gauravsingal789@gmail.com") == 0  # caused a real mis-attribution
    assert name_match("Mahipal Jadeja", "mahmed.cse@mnit.ac.in") == 0
    assert name_match("Pramod Kumar Vishwakarma", "atul.kumar202@gmail.com") == 0  # shared middle name, other person
    assert name_match("Ravinder Kumar", "atul.kumar@x.in") == 0
    assert name_match("Ravinder Kumar", "kumar@x.in") == 0.5
    for name, addr in [("Mahipal Jadeja", "mahipaljadeja.cse@mnit.ac.in"), ("Tushar Dhar Shukla", "drtusharshukla@gmail.com"),
                       ("Maninder Kaur", "manindersohal@thapar.edu"), ("Joshuva Arockia Dhanraj", "joshuva1991@gmail.com"),
                       ("Jatin Bedi", "jatin.bedi@thapar.edu"), ("Sanjay Kumar", "skumar@x.in"),
                       ("Ravinder Kumar", "ravinder@thapar.edu"), ("Amandeep Kaur", "amandeep76@gmail.com")]:
        assert name_match(name, addr) == 1.0, (name, addr)


def test_one_wrong_email_cannot_chain_people_together():
    # A directory mis-assigns Gaurav's address to Manoj as well. They must stay two people.
    manoj = Contact(name="Manoj Singh Gaur", org_domain="mnit.ac.in", emails=[
        EmailCandidate(address="gauravsingal789@gmail.com", confidence=0.85, method="directory")])
    gaurav = Contact(name="Gaurav Singal", org_domain="mnit.ac.in", emails=[
        EmailCandidate(address="gauravsingal789@gmail.com", confidence=0.95, method="directory")])
    assert len(resolve([manoj, gaurav])) == 2
