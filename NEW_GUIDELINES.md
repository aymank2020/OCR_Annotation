# 📋 Egocentric Annotation Program - Complete Guidelines

**Complete reference for the Egocentric Annotation Program**

---

## 🎯 Overview

The **Egocentric Annotation Program** is a manual video annotation workflow for reviewing text annotations on egocentric videos showing humans completing tasks from a first-person (ego) perspective.

### Your Role
As a reviewer, you are responsible for:
- ✅ Reviewing text annotations (labels) segment-by-segment
- ✅ Correcting labels when necessary
- ✅ Ensuring ego's main actions and objects are accurate
- ✅ Ensuring timestamps for each segment are correct

### What to Focus On
- ✅ **Main Actions:** Primary task being performed
- ✅ **Hand Dexterity:** Hands and meaningful object interactions
- ✅ **Primary Task:** The main goal/achievement

### What NOT to Focus On
- ❌ Movement through space (walking, navigating)
- ❌ Idle hand gestures unrelated to work environment

---

## 📝 Core Mental Model

### Definitions

- **Episode:** A full video task
- **Segment:** A continuous time span paired with one label
- **Core Mental Model:** A segment represents **one continuous interaction with a primary object toward a single goal**

### Segment Boundaries

A segment typically:
- **Begins:** When the hands engage the primary object
- **Ends:** When that interaction is complete, when the hands disengage, or when the interaction focus or goal changes

### Split Rules
**Split when:**
- ✅ Hands disengage and a new interaction begins
- ✅ A new goal/action begins that must be labeled separately
- ✅ Change in primary object or interaction focus

**Do NOT split just for:**
- ❌ Minor idle time inside segment
- ❌ To isolate "No Action" pauses

---

## ✍️ Label Format Rules

### 1. Imperative Voice ⭐ CRITICAL

Write labels as **commands**, not descriptions.

✅ **CORRECT:**
```
pick up spoon
place box on table
move mat to table
adjust cloth position
```

❌ **INCORRECT:**
```
picking up spoon
placing box on table
moving mat to table
adjusting cloth position
```

### 2. Consistency Rule

Use **consistent verbs and nouns** throughout an episode:

✅ **CORRECT:**
```
pick up blue shirt, place on table
pick up black shirt, place on table
pick up green shirt, place on table
```

❌ **INCORRECT:**
```
pick up blue shirt, place on table
take black shirt, put on table
grab green shirt, drop on table
```

### 3. Action Separators

When multiple actions in one label, separate with **comma** or **and**:

✅ **CORRECT:**
```
pick up cup, place cup on table
pick up cup and place cup on table
pick up cloth, wipe table, place cloth down
```

❌ **INCORRECT:**
```
pick up cup place cup on table  # No separator
```

### 4. No Numerals 🚨

**Always use words**, never digits:

✅ **CORRECT:**
```
pick up three knives
place five boxes
move two chairs
fold seven shirts
```

❌ **INCORRECT:**
```
pick up 3 knives
place 5 boxes
move 2 chairs
fold 7 shirts
```

**When to omit:**
- If quantity not required: `pick up knives`
- If ambiguous: `pick up blue knives`

### 5. No Intent-Only Language

Prefer **physical verbs** over mental state descriptions:

✅ **CORRECT:**
```
pick up scissors
cut tape
place sticker
```

❌ **INCORRECT:**
```
preparing to cut tape
getting ready to cut
thinking about cutting
```

---

## 🎯 Dense vs Coarse Labels

### Rule: **Either Dense OR Coarse — do not mix within single segment**

### When to Use Coarse

**Use coarse when:**
- ✅ A clear goal exists
- ✅ Listing atomic steps risks errors/hallucination
- ✅ The atomic steps are too many to list safely

**Examples:**
```
move mat to table (coarse)
move eggs in crate (coarse)
move box onto shelf (coarse)
clean table with yellow cloth (coarse)
```

**Why Coarse?**
- Reduces hallucinations
- More accurate for complex multi-step actions
- Clearer single goal

### When to Use Dense

**Use dense when:**
- ✅ Multiple distinct hand actions are required to be accurate
- ✅ No single goal verb fits

**Examples:**
```
pick up mat, place mat on table (dense)
pick up eggs, place eggs in crate (dense)
pick up cup, place cup on table (dense)
pick up cloth, wipe table, place cloth back (dense)
```

**When Dense Required:**
- Order of steps matters
- Multiple discrete interactions must be explicit
- No single verb covers entire interaction

### Length Guideline

- **Ideal:** ~20 words or ~4 atomic actions per label
- **Not strict:** Accuracy and completeness take priority
- **Long labels:** Consider coarse if becoming too long/complex

---

## 📚 Action Verb Rules

### ❌ FORBIDDEN VERBS

The following verbs **are NOT allowed**:

| Verb | Why Forbidden | Alternative |
|------|---------------|-------------|
| `inspect` | Visual judgment | `adjust` |
| `check` | Visual judgment | `adjust` |
| `examine` | Visual judgment | `adjust` |
| `reach` | Usually timestamp error | Fix timestamps |

**Exception:** `reach` may only be used when action is **truncated/cut off at end of episode** and no better verb possible.

### ✅ ALLOWED VERBS

#### **pick up**
- **Definition:** Object leaves a surface/container resting position
- **Usage:** Required when using dense and a pickup occurred
- **Examples:**
  ```
  pick up pen
  pick up cloth from counter
  pick up blue knife
  ```

#### **place**
- **Definition:** Object contacts surface and is released/positioned
- **Usage:** Required when using dense and a placement occurred
- **⭐ CRITICAL:** MUST include location
- **Examples:**
  ```
  place cup on table ✅
  place cup in bin ✅
  place box ⚠️ (missing location)
  place object ⚠️ (too general)
  place cup on table ✅
  place cup in bin ✅
  place box on shelf ✅
  place shoes by door ✅
  ```

#### **move**
- **Definition:** Coarse relocation describing pick up + place as one goal, OR repositioning without detailing steps
- **Usage:** ✅ Allowed coarse substitute for "pick up and place" when relocation is the goal
- **Examples (Coarse):**
  ```
  move mat to table ✅
  move box onto shelf ✅
  move eggs in crate ✅
  move chair to corner ✅
  ```
- **When dense, must be explicit:**
  ```
  pick up mat, place mat on table (dense)
  ```

#### **adjust**
- **Definition:** Small corrective change in position/orientation
- **Usage:** Use instead of inspect/check
- **Examples:**
  ```
  adjust shirt on board
  adjust cloth position
  adjust pen alignment
  ```

#### **hold**
- **Definition:** Maintain grip without relocating
- **Usage:** Only if task-relevant
- **Examples:**
  ```
  hold pen steady
  hold cloth while cutting
  ```

#### **grab**
- **Definition:** Grip itself is meaningful
- **Usage:** Rare; use sparingly
- **Examples:**
  ```
  grab handle
  grab tool
  ```

### Verb Attachment Rule

**Every verb should clearly apply to an object:**

✅ **CORRECT:**
```
pick up spoon
place cup on table
move mat to table
```

❌ **INCORRECT:**
```
pick up
place
move
```

---

## 🚫 No Action Rules

### When to Use "No Action"

**Use "No Action" only when:**
- ✅ Hands touch nothing
- ✅ Ego is idle / doing irrelevant behavior unrelated to the task

### No Action Rules

**Do NOT:**
- ❌ Split solely to isolate "No Action" pauses
- ❌ Combine "No Action" with real actions in single label
- ❌ Use "No Action" if ego is holding object and that hold is task-relevant

**Examples:**

✅ **CORRECT:**
```
Segment 1: pick up spoon, stir soup
Segment 2: No Action
Segment 3: add salt, stir soup
```

❌ **INCORRECT:**
```
Segment 1: pick up spoon, stir soup
Segment 2: No Action, check phone (combined)
Segment 3: add salt, stir soup
```

---

## 🎥 Object Guidelines

### Identification Rule

**Identify only what you can defend:**
- ✅ Clear objects: `spoon`, `cup`, `table`
- ⚠️ Unsure: Use general nouns (`tool`, `container`, `cloth`)

### Consistency Rule

**Stay consistent in object naming through episode:**

✅ **CORRECT:**
```
Segment 1: pick up blue cloth
Segment 2: wipe table with blue cloth
Segment 3: place blue cloth down
```

❌ **INCORRECT:**
```
Segment 1: pick up cloth
Segment 2: wipe table with rag
Segment 3: place towel down
```

### Adjective Rule

**Use adjectives only to disambiguate:**

✅ **NEEDED:**
```
blue cloth vs white cloth
left shoe vs right shoe
```

❌ **NOT NEEDED:**
```
cloth (if only one color)
knife (if only one type)
```

### "Place" Location Rule

**`place` always requires a location** (can be general):

✅ **CORRECT:**
```
place cup on table
place cup in bin
place object (too general)
```

### Left/Right Rule

**Allowed if accurate from ego view, but not required:**

```
pick up left shoe ✅
pick up right glove ✅
pick up shoe ✅ (acceptable without left/right)
```

### Body Parts Rule

**Avoid referencing body parts unless unavoidable:**

✅ **PREFERRED:**
```
apply glue to shoe
wipe table surface
```

✅ **ACCEPTABLE (if needed):**
```
apply glue to shoe with finger (if it's the only clear description)
```

---

## ⏱️ Segment Editing Rules

### Timestamps

**Start:**
- When action begins
- Hands begin engaging toward contact
- Cover full interaction

**End:**
- When hands disengage
- When interaction ends

**Minor idle time** inside segment is acceptable if still one continuous interaction

### Extend / Shorten

**Use to align boundaries to true action:**
- ✅ Align to when action actually begins/ends
- ❌ Don't extend into a new action
- ❌ Don't cut off completion of the action

### Merge (When Allowed)

**Merge adjacent segments only if:**
- ✅ Same action/goal
- ✅ Hands never disengage between them

### Do NOT Merge

**Do not merge when:**
- ❌ Repeated pick up → place cycles with clear disengagement
- ❌ Different objects
- ❌ Different goals

### Split (When Required)

**Split when:**
- ✅ Hands disengage and a new interaction begins
- ✅ A new goal/action begins that must be labeled separately

---

## 🔄 Repeated & Simultaneous Actions

### Repeated Actions

**Rule:**
- If ego disengages and repeats → **multiple segments**
- If ego never disengages → **one segment** (often coarse)

**Examples:**

Disengage + Repeat (Multiple Segments):
```
Segment 1: pick up cloth, wipe table, place down
Segment 2: No Action
Segment 3: pick up cloth, wipe table, place down
```

Never Disengages (One Segment):
```
Segment 1: wipe table with cloth continuously (coarse)
```

### Simultaneous Actions

**Capture all task-relevant actions:**
- ✅ Include all relevant simultaneous actions
- Ignore irrelevant side actions (phone, camera touch, etc.)

---

## ❌ Audit Fail Conditions

A segment **FAILS** audit if **ANY** of the following are true:

| Condition | Description |
|-----------|-------------|
| ❌ Missed action | Missed major task-relevant hand action |
| ❌ Hallucinated | Hallucinated (non-occurring) action/object |
| ❌ Timestamps | Timestamps cut off action or include different action |
| ❌ Forbidden verbs | Forbidden verbs used (`inspect`, `check`, `examine`, `reach`) |
| ❌ Mixed granularity | Dense/coarse mixed in one label |
| ❌ No Action combo | "No Action" combined with action |

---

## ✅ Ideal Segment Checklist

✅ **One goal**
✅ **Full action coverage**
✅ **Accurate verbs**
✅ **No hallucinated steps**
✅ **Dense OR coarse (not mixed)**
✅ **Imperative voice**
✅ **No numerals**
✅ **No forbidden verbs**
✅ **Objects clearly identified**
✅ **Timestamps accurate**

---

## 📚 Reference & Edge Cases

### Move vs Pick Up + Place

| **Move** (Coarse) | **Pick Up + Place** (Dense) |
|-------------------|----------------------------|
| `move mat to table` | `pick up mat, place mat on table` |
| `move eggs in crate` | `pick up eggs, place eggs in crate` |

**Use `move` when:**
- Goal is relocation
- Intermediate steps add no value
- Accuracy improves by abstraction

**Use `pick up + place` when:**
- Order matters
- Labeling densely
- Multiple discrete interactions must be explicit

### Merge vs Split Flow

```
1. Hands disengage? → Yes: Split, No: Continue
2. Same goal? → Yes: Merge/Keep, No: Split
3. Different object? → Yes: Split
```

**Never merge:**
- Repeated pick up → place cycles with disengagement
- Different goals "just to reduce count"

### Common Formatting Mistakes (Minor)

❌ `pick up 3 knives from table`
✅ `pick up three knives from table` (or `pick up knives from table`)

### When to Escalate

**Escalate via Discord if:**
- Object cannot be identified after reasonable effort
- Action cannot be labeled without guessing
- Segment cannot be made accurate via coarse abstraction

---

## 🎯 Summary Quality Rule

### Quality Over Quantity

**A well-labeled segment accurately captures the main hand-object interaction from start to finish, using clear and consistent language.**

**Remember:**
- Better to have fewer accurate segments than many inaccurate ones
- Coarse labels are often preferred for accuracy
- Consistency within episode is crucial
- When unsure, use coarse granularity

---

## 🔤 Quick Reference

### Voice
```
✅ pick up spoon (imperative)
❌ picking up spoon (participle)
```

### Verbs
```
✅ Allowed: pick up, place, move, adjust, hold, grab
❌ Forbidden: inspect, check, examine, reach
```

### Format
```
✅ pick up three knives (words)
❌ pick up 3 knives (numerals)
✅ pick up cup, place on table (separator)
```

### Granularity
```
✅ move mat to table (coarse)
✅ pick up mat, place on table (dense)
❌ pick up mat move to table (mixed)
```

### Objects
```
✅ place on table, in bin (location)
✅ blue cloth vs white cloth (adjective)
✅ consistent naming throughout episode
```

---

## 📚 External Resources

- **Training Hub:** https://audit.atlascapture.io/training/hub
- **Task Page:** https://audit.atlascapture.io/
- **Discord:** For questions and escalations

---

## 🎓 Complete Examples

### Example 1: Correct Episode

```
Episode: Pen Assembly Video

0:00.0-0:20.0#1 pick up black pen parts, assemble pen
0:20.0-0:51.0#2 pick up blue pen parts, assemble pen
0:51.0-1:15.0#3 place assembled pens in packaging box
```

Checklist: ✅ Imperative ✅ No numerals ✅ Allowed verbs ✅ Coarse/Dense consistent ✅ Objects clear

### Example 2: Common Mistakes

**❌ Original (Wrong):**
```
0:00.0-0:20.0#1 Assembling 3 black pens
0:20.0-0:51.0#2 Inspecting blue pens carefully
```

Issues:
- ❌ Participle voice
- ❌ Numeral "3"
- ❌ Forbidden verb "inspect"
- ❌ Intent language "carefully"

**✅ Corrected:**
```
0:00.0-0:20.0#1 pick up black pen parts, assemble pens
0:20.0-0:51.0#2 adjust blue pen alignment
```

---

**Remember: The goal is to capture the main hand-object interactions accurately and consistently. When in doubt, simplify with a coarse label.**

---

*Egocentric Annotation Program v2.0*
*Last Updated: 2026-02-02*