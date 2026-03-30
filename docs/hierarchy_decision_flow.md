# Heading Hierarchy Decision Flow

Decision logic in `infer_heading_hierarchy` (`step_06_hierarchy_builder.py`).

Runs once per unique heading, in reading order. The **path_stack** holds the current ancestry chain from root to the most recently placed heading.

> **Preview:** [mermaid.live](https://mermaid.live) — paste either diagram block there. VS Code: install the "Markdown Preview Mermaid Support" extension.

---

## Part 1 — Decision tree

```mermaid
flowchart TD
    START(["New heading arrives"])
    START --> EMPTY{"path_stack empty?"}

    EMPTY -- yes --> ROOT["level = 1, no parent\npush to stack"]
    EMPTY -- no  --> PRIOR["prior = stack top"]

    PRIOR --> B1{"B1: curr_fp == prior_fp\nand both not NA?"}

    B1 -- yes --> NUM{"Both in\nNUMBERED_HEADING_TYPES?"}
    B1 -- no  --> B2

    NUM -- yes --> DEPTH{"Compare numeric depth"}
    NUM -- no  --> d_SIB(["SIBLING"])

    DEPTH -- "curr_nd > prior_nd" --> d_CHD(["CHILD"])
    DEPTH -- "curr_nd == prior_nd" --> d_SIB
    DEPTH -- "curr_nd < prior_nd" --> d_NPOP(["NUMBERED_POP"])

    B2{"B2: prior.block_role == heading\nAND lines consecutive?"}
    B2 -- yes --> d_CHD
    B2 -- no  --> B3

    B3{"B3: curr_fp in fp_path?"}
    B3 -- yes --> NUM2{"curr_type in\nNUMBERED_HEADING_TYPES?"}
    B3 -- no  --> B4

    NUM2 -- yes --> d_NREA(["NUMBERED_REATTACH"])
    NUM2 -- no  --> d_REA(["REATTACH"])

    B4{"B4/B5: curr_total vs\nprior.static_weight"}
    B4 -- "less than" --> d_CHD
    B4 -- "greater than" --> d_POP(["POP_UP"])
    B4 -- equal --> MEM{"B6: curr_fp was parent\nof prior_fp in memory?"}

    MEM -- yes --> d_POP
    MEM -- no  --> d_CHD

    classDef decision fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef outcome  fill:#f3f4f6,stroke:#6b7280,color:#111827
    class B1,B2,B3,B4,MEM,EMPTY,NUM,NUM2,DEPTH decision
    class d_SIB,d_CHD,d_REA,d_POP,d_NPOP,d_NREA outcome
```

---

## Part 2 — Outcome handlers

```mermaid
flowchart TD

    SIB(["SIBLING"])
    SIB --> SIB1{"stack depth > 1?"}
    SIB1 -- yes --> SIB2["parent = stack-2\nlevel = prior.level\nreplace stack top"]
    SIB1 -- no  --> SIB3["no parent, level = 1\nreplace stack top"]

    CHD(["CHILD"])
    CHD --> CHD1["parent = prior\nlevel = prior.level + 1\npush onto stack"]

    REA(["REATTACH"])
    REA --> REA1["Find last node in stack\nwhere node.fp == curr_fp"]
    REA1 --> REA2["Truncate stack to that node"]
    REA2 --> REA3{"node at stack root?"}
    REA3 -- no  --> REA4["parent = node before it\nlevel = node.level\nreplace that node"]
    REA3 -- yes --> REA5["no parent, level = 1\nreplace root node"]

    POP(["POP_UP"])
    POP --> POP1{"Last special node in stack?\ntype not free_form/table/figure"}
    POP1 -- yes --> POP2["Truncate to that node\nparent = that node\nlevel = that node.level + 1\npush onto stack"]
    POP1 -- no  --> POP3["Pop one node"]
    POP3 --> POP4{"curr is high rank\nOR stack now empty?"}
    POP4 -- yes --> POP5["no parent, level = 1\npush onto stack"]
    POP4 -- no  --> POP6["parent = new stack top\nlevel = top.level + 1\npush onto stack"]

    NPOP(["NUMBERED_POP"])
    NPOP --> NPOP1["Walk back: find deepest\nnumbered node with depth <= curr_nd"]
    NPOP1 --> NPOP2{"Found?"}
    NPOP2 -- no  --> NPOP3["Clear stack\nno parent, level = 1\npush onto stack"]
    NPOP2 -- yes --> NPOP4{"node.depth == curr_nd?"}
    NPOP4 -- "yes: sibling" --> NPOP5["Truncate to that node\nparent = node before it\nlevel = node.level\nreplace that node"]
    NPOP4 -- "no: child of shallower" --> NPOP6["Truncate to that node\nparent = that node\nlevel = node.level + 1\npush onto stack"]

    NREA(["NUMBERED_REATTACH"])
    NREA --> NREA1["Find deepest node:\nsame fp + numbered type\n+ depth <= curr_nd"]
    NREA1 --> NREA2{"Found?"}
    NREA2 -- "no: fallback" --> NREA3["Find last node\nwith same fp only"]
    NREA3 --> NREA4{"Found?"}
    NREA4 -- no  --> NREA5["no-op: heading unplaced"]
    NREA4 -- yes --> NREA6
    NREA2 -- yes --> NREA6{"node.depth == curr_nd?"}
    NREA6 -- "yes: sibling" --> NREA7["Truncate to that node\nparent = node before it\nlevel = node.level\nreplace that node"]
    NREA6 -- "no: child of shallower" --> NREA8["Truncate to that node\nparent = that node\nlevel = node.level + 1\npush onto stack"]

    classDef entry    fill:#f3f4f6,stroke:#6b7280,color:#111827
    classDef decision fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef terminal fill:#dcfce7,stroke:#16a34a,color:#14532d
    class SIB,CHD,REA,POP,NPOP,NREA entry
    class SIB1,REA3,POP1,POP4,NPOP2,NPOP4,NREA2,NREA4,NREA6 decision
    class SIB2,SIB3,CHD1,REA4,REA5,POP2,POP5,POP6,NPOP3,NPOP5,NPOP6,NREA5,NREA7,NREA8 terminal
```

---

## Decision priority summary

| # | Condition | Decision |
|---|-----------|----------|
| 1 | `curr_fp == prior_fp` — same fingerprint as stack top | **sibling** — or depth-routed for numbered sections |
| 2 | Prior is a real `heading` block and line IDs are consecutive | **child** |
| 3 | `curr_fp` appears anywhere earlier in the stack | **reattach** / **numbered_reattach** |
| 4 | `curr_total < prior.static_weight` | **child** |
| 5 | `curr_total > prior.static_weight` | **pop_up** |
| 6 | Equal weight | **pop_up** if curr was a known parent of prior, else **child** |

## Numbered section depth routing

Depth is extracted from the numeric prefix of the heading text:

| Text | Depth |
|------|-------|
| `1.` | 1 |
| `1.2.` | 2 |
| `1.2.3` | 3 |

Cross-article re-entry via **numbered_reattach** (branch 3):

```
ARTICLE VII        ← level 1
  Conditions       ← level 2
    7.1.           ← depth 1, level 3  ┐
    7.2.           ←                   ├ branch 1 sibling chain
    7.3.           ←                   ┘
ARTICLE VIII       ← branch 3 reattach    → level 1
  Termination      ← branch 4/5 weight   → level 2
    8.1.           ← branch 3 numbered_reattach, depth 1 → level 3
    8.2.           ← branch 1 sibling
```
