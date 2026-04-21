import sys

with open('main (3).tex.orig', 'r') as f:
    text = f.read()

# Fix 1: fig:layers
old_layers = """  \draw[diagarr] (main) -- (agents);
  \draw[diagarr] (agents) -- (orch);
  \draw[diagarr] (agents.south) -- ++(0,-0.32) -| (crawl.north);
  \draw[diagarr] (crawl) -- (core);
  \draw[diagarr] (crawl) -- (db);
  \draw[diagarr] (orch.south) -- ++(0,-0.38) -| (db.north);

  \draw[diagarr, dashed] (db.south) -- ++(0,-0.3) -| (ui.north);
  \draw[diagarr, dashed] (db.south west) -- ++(-0.15,-0.2) -| (email.north);
  \draw[diagarr, dashed] (sched.north) -- ++(0,0.55) -| (crawl.south east);
  \draw[diagarr, dashed] (sched.north west) -- ++(-0.2,0.65) -| (email.south east);"""

new_layers = """  \draw[diagarr] (main) -- (agents);
  \draw[diagarr] (agents) -- (orch);
  \draw[diagarr] (orch.south) -- ++(0,-0.32) -| (crawl.north);
  \draw[diagarr] (crawl) -- (core);

  \draw[diagarr, dashed] (main.south) -- ++(0,-0.8) -| (db.north west);
  \draw[diagarr, dashed] (ui.north) -- ++(0,0.3) -| (db.south west);
  
  \draw[diagarr, dashed] (sched.north) -- (db.south);
  \draw[diagarr, dashed] (sched.south) -- ++(0,-0.3) -| (email.south);"""

if old_layers in text:
    text = text.replace(old_layers, new_layers)
else:
    print("WARNING: Could not find old_layers block")

# Fix 2: fig:repo-tree
old_cap = r"\node[pkg, fill=blue!5, minimum width=6.9cm, minimum height=0.55cm] (cap) at (3.05,0.45) {\textbf{repository root (excerpt)}};"
new_cap = r"\node[pkg, fill=blue!5, minimum width=6.9cm, minimum height=0.55cm] (cap) at (3.05,0.85) {\textbf{repository root (excerpt)}};"
if old_cap in text:
    text = text.replace(old_cap, new_cap)
else:
    print("WARNING: Could not find old_cap")

old_foreach = r"\draw[diagarr, gray!55] (cap.south) -- ++(0,-0.06) -| (\n.north);"
new_foreach = r"\draw[diagarr, gray!55] (cap.south) -- ++(0,-0.15) -| (\n.north);"
if old_foreach in text:
    text = text.replace(old_foreach, new_foreach)
else:
    print("WARNING: Could not find old_foreach")

# Fix 3: cli-five caption
old_caption = r"\caption{Supervised \texttt{main.py start} pipeline as labeled in the CLI (matches \texttt{main.py}).}"
new_caption = r"\caption{Supervised pipeline phases (Steps 4 and 5 are explicitly labeled this way in the CLI).}"
if old_caption in text:
    text = text.replace(old_caption, new_caption)
else:
    print("WARNING: Could not find old_caption")

# Fix 4: intent-slots
old_f_arrow = r"\draw[diagarr] (f) |- (b.west);"
new_f_arrow = r"\draw[diagarr] (f.east) -| (b.south);"
if old_f_arrow in text:
    text = text.replace(old_f_arrow, new_f_arrow)
else:
    print("WARNING: Could not find old_f_arrow")

with open('main (3).tex', 'w') as f:
    f.write(text)

