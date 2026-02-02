# Quick Fix Script for model_architecture.py
# اقرأ هذا الملف ثم طبّق التغييرات يدوياً

"""
========================================
إصلاح model_architecture.py
========================================

المشكلة:
--------
السطر 100 يحتوي على nn.Sigmoid() الذي يسبب خطأ في التدريب

الحل:
-----
احذف nn.Sigmoid() من boundary_detector

الكود القديم (خطأ):
------------------
"""

# OLD CODE (WRONG):
self.boundary_detector = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model // 2, 2),  # [start_prob, end_prob]
    nn.Sigmoid()  # ❌ THIS CAUSES THE ERROR!
)

"""
الكود الجديد (صحيح):
------------------
"""

# NEW CODE (CORRECT):
self.boundary_detector = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model // 2, 2)  # ✅ No Sigmoid - output logits
)

"""
========================================
خطوات التطبيق:
========================================

1. افتح الملف:
   E:\OCR_system-Atlas\src\model_architecture.py

2. ابحث عن السطر ~100 (boundary_detector)

3. احذف السطر:
   nn.Sigmoid()

4. احفظ الملف

5. أعد التدريب:
   python main.py --mode train --epochs 20

========================================
"""

print("""
✅ تعليمات الإصلاح:

1. افتح: src/model_architecture.py
2. ابحث عن: self.boundary_detector
3. احذف السطر: nn.Sigmoid()
4. احفظ الملف
5. شغّل: python main.py --mode train --epochs 20
""")
