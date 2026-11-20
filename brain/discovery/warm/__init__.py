"""
Warm paths: the user's own network, imported explicitly (`brain network-import <file>`), never read without it.

    importers.py  LinkedIn "Connections.csv" export, Google Contacts CSV, vCard (.vcf), plain CSV (name, email, company)
    paths.py      marks candidates the user knows (warm_path 1.0) or can be introduced to through someone they know
                  at the same organisation (0.6), with evidence the email may mention
Everything stays in the local store (brain_network); `brain network-clear` deletes it.
"""
