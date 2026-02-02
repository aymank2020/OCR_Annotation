# 🚀 OCR Annotation System - Egocentric Annotation Program

**AI-powered annotation system for egocentric videos, compliant with Egocentric Annotation Program guidelines**

---

## ✅ What's New (Updated 2026-02-02)

### Complete Overhaul - Egocentric Annotation Program Compliance

This project has been **completely updated** to follow the **Egocentric Annotation Program** guidelines from [Atlas Capture](https://audit.atlascapture.io/).

**Key Changes:**
- ✅ **Imperative Voice** labels (no more -ing verbs)
- ✅ **Dense vs Coarse** label distinction
- ✅ **Specific verb rules** (pick up, place, move, adjust, hold, grab)
- ✅ **Forbidden verbs** (inspect, check, examine, reach)
- ✅ **No Action** handling
- ✅ **Annotation validation** module for compliance checking

---

## 📋 Project Overview

This system provides:
- **AI Action Recognition** using VideoX/CLIP models
- **Annotation Validation** against Egocentric Annotation Program rules
- **Manual Annotation Review** tools and guidelines
- **Google Drive Integration** for video assets

### Focus Areas
- **Main Actions:** Primary task being performed
- **Hand Dexterity:** Hands and meaningful object interactions
- **Primary Task:** The main goal/achievement

### Don't Focus On
- ❌ Movement through space (walking, navigating)
- ❌ Idle hand gestures unrelated to work environment

---

## 🎬 Label Format Rules

### 1. Imperative Voice
Write labels as **commands**:

✅ **CORRECT:**
```
pick up spoon
place box on table
move mat to table
```

❌ **INCORRECT:**
```
picking up spoon
placing box on table
moving mat to table
```

### 2. Action Separators
Use comma or "and" for multiple actions:

✅ **CORRECT:**
```
pick up cup, place cup on table
pick up cup and place cup on table
```

❌ **INCORRECT:**
```
pick up cup place cup on table  # No separator
```

### 3. No Numerals
Use **words** instead of digits:

✅ **CORRECT:**
```
pick up three knives
place five boxes
move two chairs
```

❌ **INCORRECT:**
```
pick up 3 knives
place 5 boxes
move 2 chairs
```

### 4. No Intent-Only Language
Prefer **physical verbs** over mental state descriptions:

✅ **CORRECT:**
```
pick up scissors
cut tape
```

❌ **INCORRECT:**
```
preparing to cut tape
getting ready to cut
thinking about cutting
```

---

## 📝 Action Verb Rules

### ✅ Allowed Verbs

| Verb | Definition | Usage |
|------|------------|-------|
| **pick up** | Object leaves a surface/container | Required for dense when pickup occurs |
| **place** | Object contacts surface released | Required for dense when placement occurred (**requires location**) |
| **move** | Coarse relocation (pick up + place) | Allowed coarse substitute for relocation goal |
| **adjust** | Small corrective change | Use instead of "inspect" or "check" |
| **hold** | Maintain grip without relocating | Only if task-relevant |
| **grab** | Grip itself is meaningful | Rare; use sparingly |

### ❌ Forbidden Verbs
- `inspect` ❌
- `check` ❌
- `examine` ❌
- `reach` ❌ (except truncated at episode end)

### "Place" Rule
`place` **always requires a location**:

✅ **CORRECT:**
```
place cup on table
place cup in bin
place box on shelf
```

❌ **INCORRECT:**
```
place cup
place object
```

---

## 🎯 Dense vs Coarse Labels

### Rule: **Either Dense OR Coarse — do not mix**

### Coarse Labels
**Use when:**
- ✅ Clear goal exists
- ✅ Listing atomic steps risks errors/hallucination
- ✅ Too many atomic steps to list safely

**Examples:**
```
move mat to table
move eggs in crate
move box onto shelf
```

### Dense Labels
**Use when:**
- ✅ Multiple distinct hand actions required for accuracy
- ✅ No single goal verb fits

**Examples:**
```
pick up mat, place mat on table
pick up cup, place cup in bin
pick up cloth, wipe table, place cloth down
```

**Note:** Dense is not "better" than coarse. Coarse is often preferred for accuracy.

---

## 🚫 No Action Rules

### When to Use "No Action"
- ✅ Hands touch nothing
- ✅ Ego is idle / doing irrelevant behavior unrelated to task

### No Action Rules
- ❌ Do **not** split solely to isolate "No Action" pauses
- ❌ Do **not** combine "No Action" with real actions
- ❌ Do **not** use "No Action" if ego is holding object and that hold is task-relevant

---

## 📦 File Structure

```
OCR_Annotation/
├── config/
│   └── config.yaml              # Main config with Egocentric rules
├── src/
│   ├── annotation_validator.py  # NEW! Validation module
│   ├── model_architecture.py
│   ├── dataset_loader.py
│   ├── trainer_module.py
│   └── inference_module.py
├── data/
│   └── videos/                  # Video files (f1, f2, f3, etc.)
├── main.py                      # Main CLI interface
├── app.py                       # Flask API server
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── NEW_GUIDELINES.md            # Detailed Egocentric rules summary
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/aymank2020/OCR_Annotation.git
cd OCR_Annotation

# Install dependencies
pip install -r requirements.txt

# (Optional) Install VideoX
pip install torch transformers opencv-python
```

### 2. Download Videos from Google Drive

**Videos Folder:**
https://drive.google.com/drive/folders/193m6v05VrN-VivKinRF-w05uWzXAxKu2

**Available Videos:**
- `f1.mp4` (12.9 MB) - Pen assembly
- `f2.mp4` (18.3 MB) - Shirt ironing
- `f3.mp4` (14 MB) - Clothing folding
- `f4.mp4` (17.8 MB) - Garment ironing
- `f5.mp4` (15.6 MB) - Sweater and socks folding
- `f6.mp4` (22 MB) - Table cleaning
- `f7.mp4` (21.1 MB) - Bird nest cleaning
- `f100.mp4` (151.5 MB)
- `f101.mp4` (26 MB)

Download and place in `data/videos/`

### 3. Validate Annotations

```bash
# Test annotation validator
python src/annotation_validator.py

# Validate specific label
python -c "
from src.annotation_validator import AnnotationValidator
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

validator = AnnotationValidator(config)
result = validator.validate_label('pick up three knives')
print(validator.get_validation_report(result))
"
```

### 4. Run System

```bash
# Prepare data
python main.py --mode prepare

# Train model (optional)
python main.py --mode train --epochs 50

# Start API server
python app.py

# Access web interface
# Open browser: http://localhost:5000
```

---

## 📊 Annotation Examples

### Example 1: Pen Assembly (f1.mp4)
```
0:00.0-0:20.0#1 Pick up black pen assembly parts, align components
0:20.0-0:51.0#2 Pick up blue pen parts, assemble blue pens
0:51.0-1:15.0#3 Place assembled pens into packaging boxes
```

### Example 2: Shirt Ironing (f2.mp4)
```
0:00.0-0:16.0#1 Pick up blue shirt, place on ironing board
0:16.0-0:37.3#2 Smooth blue shirt, align edges
0:37.3-0:46.3#3 Smooth blue shirt surface
0:46.3-0:56.7#4 Fold blue shirt, adjust on board
0:56.7-2:00.0#5 Smooth shirt, fold on ironing table
```

### Example 3: Dense Label
```
pick up cloth, wipe table surface, place cloth back on counter
```

### Example 4: Coarse Label
```
clean table with yellow cloth
```

---

## ✅ Audit Fail Conditions

A segment **FAILS** audit if:

- ❌ Missed major task-relevant hand action
- ❌ Hallucinated (non-occurring) action/object
- ❌ Timestamps cut off action or include different action
- ❌ **Forbidden verbs used** (`inspect`, `check`, `examine`, `reach`)
- ❌ **Dense/coarse mixed in one label**
- ❌ **"No Action" combined with action**
- ❌ **Numerals present** (`3` instead of `three`)

---

## 🎯 Ideal Segment Checklist

✅ One goal
✅ Full action coverage
✅ Accurate verbs
✅ No hallucinated steps
✅ **Dense OR coarse (not mixed)**
✅ **Imperative voice**
✅ **No numerals**
✅ **No forbidden verbs**

---

## 🔧 Configuration

### Edit `config/config.yaml`:

```yaml
egocentric_annotation:
  label_format:
    voice: imperative
    no_numerals: true

  verbs:
    forbidden:
      - "inspect"
      - "check"
      - "examine"
      - "reach"

  validation:
    enable_validation: true
    strict_mode: false
```

---

## 📚 Documentation

### Core Documents:
- **`NEW_GUIDELINES.md`** - Complete Egocentric Annotation Program rules
- **`config/config.yaml`** - Full configuration with inline comments
- **`src/annotation_validator.py`** - Validation module with examples

### External Resources:
- **Atlas Capture:** https://audit.atlascapture.io/
- **Training Hub:** https://audit.atlascapture.io/training/hub
- **GitHub:** https://github.com/aymank2020/OCR_Annotation

---

## 🔍 Validation Module

### Features:
- ✅ Forbidden verb detection
- ✅ Numeral checking
- ✅ Imperative voice validation
- ✅ Object naming verification
- ✅ Verb compliance checking
- ✅ Length constraint validation
- ✅ No Action rule checking
- ✅ Timestamp validation
- ✅ Dense/Coarse mixing detection
- ✅ Episode consistency checking

### Usage:
```python
from src.annotation_validator import AnnotationValidator
import yaml

# Load config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Create validator
validator = AnnotationValidator(config)

# Validate label
result = validator.validate_label("pick up three knives")
print(validator.get_validation_report(result))
```

---

## 🤝 Contributing

When adding annotations or modifying code:

1. **Always use imperative voice** (commands, not descriptions)
2. **Run validation** before committing
3. **Test with multiple examples**
4. **Follow Egocentric Annotation Program guidelines**

---

## 📝 Migration from Old Format

### Before (Easy Mode):
```
0:00.0-0:20.0#1 Assembling black ballpoint pens
0:20.0-0:51.0#2 Assembling blue ballpoint pens
```
❌ Present participle (-ing)
❌ Descriptive sentences

### After (Egocentric):
```
0:00.0-0:20.0#1 Pick up black pen assembly parts, align components
0:20.0-0:51.0#2 Pick up blue pen parts, assemble blue pens
```
✅ Imperative voice
✅ Commands
✅ Specific verbs

---

## 🎓 Key Differences Summary

| Aspect | Old (Easy Mode) | New (Egocentric) |
|--------|-----------------|------------------|
| **Voice** | Present participle (-ing) | Imperative (commands) |
| **Verbs** | Any descriptive verb | Specific set only |
| **Forbidden** | General intent words | `inspect`, `check`, `examine`, `reach` |
| **Granularity** | Single mode | Dense vs Coarse distinction |
| **Numerals** | Allowed | Use words (`three` not `3`) |
| **Validation** | Basic | Comprehensive module |

---

## 🆘 Troubleshooting

### Validation Fails:
1. Check for forbidden verbs (`inspect`, `check`, etc.)
2. Convert numerals to words
3. Ensure imperative voice (remove -ing)
4. Verify `place` has location
5. Check no intent-only language

### Issues:
```bash
# Test validator
python src/annotation_validator.py

# Check syntax
python -m py_compile src/annotation_validator.py

# Verify config
python -c "import yaml; print(yaml.safe_load(open('config/config.yaml'))['egocentric_annotation'].keys())"
```

---

## 📞 Support

- **Documentation:** Check `NEW_GUIDELINES.md` and inline comments in `config/config.yaml`
- **GitHub Issues:** https://github.com/aymank2020/OCR_Annotation/issues
- **Project Site:** https://audit.atlascapture.io/

---

## 🎉 Summary

### What Changed:
- ✅ Complete config.yaml overhaul with Egocentric rules
- ✅ New annotation validation module
- ✅ Updated README with new guidelines
- ✅ Sample annotations updated to new format

### What's Included:
- ✅ AI action recognition (VideoX/CLIP)
- ✅ Annotation validation
- ✅ Egocentric Annotation Program compliance
- ✅ Google Drive video support

### Quality Over Quantity:
**A well-labeled segment accurately captures the main hand-object interaction from start to finish, using clear and consistent language.**

---

**🚀 Ready to annotate with confidence!**

---

*Last Updated: 2026-02-02*
*Compliance: Egocentric Annotation Program v2.0*