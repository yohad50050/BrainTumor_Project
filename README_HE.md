# Brain Tumor MRI — AI FATE Phase A

הקוד משתמש בדאטה המקומי בלבד. אין בו קוד שמוריד דאטהסט.
המודל: EfficientNetB0 pretrained on ImageNet, עם ראש סיווג לארבע מחלקות.
תחילה מאמנים את ראש הסיווג ולאחר מכן מבצעים fine-tuning לשלושת בלוקי התכונות האחרונים.
משקולות ImageNet יורדות פעם אחת אוטומטית; נדרש אינטרנט בהתקנה ובהרצה הראשונה.

## הרצה ב-Ubuntu וב-VS Code

1. חלצו את חבילת הקוד. ב-VS Code בחרו File > Open Folder ופתחו את BrainTumor_Project.
2. פתחו Terminal > New Terminal. השתמשו ב-Python 3.11 או 3.12 (Ubuntu 24.04 מגיע בדרך כלל עם 3.12).
3. הריצו פעם אחת:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

אם יצירת הסביבה נכשלת בגלל ensurepip/venv:

```bash
sudo apt install python3-venv
```

ואז חזרו על יצירת הסביבה. ב-VS Code בחרו Ctrl+Shift+P > Python: Select Interpreter > .venv/bin/python.

4. בדקו את כרטיס המסך:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

אם כתוב False, הקוד עדיין יעבוד במעבד אך יהיה איטי יותר. הריצו nvidia-smi לבדיקת הדרייבר. ניתן לבחור התקנה מתאימה לדרייבר דרך https://pytorch.org/get-started/locally/ . אין צורך להתקין CUDA Toolkit בנפרד לצורך גלגלי PyTorch.
למחשב ללא NVIDIA אפשר להתקין במקום פקודת PyTorch למעלה:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

5. לפי צילום המסך שלך, זו פקודת האימון:

```bash
python brain_tumor.py train --data "$HOME/Downloads/archive" --output results
```

הקוד מאתר אוטומטית את brain-tumor-mri-deduplicated/data בתוך התיקייה. אין צורך להזין את שם המשתמש ואין צורך להריץ את קובץ preprocessing של יוצר הדאטה.
חלוקת הדאטה הצפויה בגרסה 2: train 4530, val 502, test 1475. הקוד סופר את הקבצים בפועל.
ניתן קודם לבצע בדיקת מבנה וכפילויות ללא הורדת משקולות וללא אימון:

```bash
python brain_tumor.py train --data "$HOME/Downloads/archive" --audit-only --output audit_results
```

## מה קורה באימון?

- בדיקת תמונות ותוויות ושמירת רשימת הקבצים.
- בדיקת כפילויות פיקסלים מדויקות בכל המחלקות והחלוקות. כפילויות מוסרות, תוך העדפת test על val על train; קבוצות עם תוויות סותרות מוסרות. כל שינוי מתועד ב-excluded_images.json. אין שינוי בקובצי המקור.
- השארת החלוקה שסופקה; אין ערבוב מחדש בין train, val, test.
- שמירת בחירה אקראית משוכבת וקבועה של 100 תמונות test מראש.
- שינוי גודל ל-224×224 עם ריפוד לשמירת יחס התמונה, נרמול ImageNet, והגדלות נתונים עדינות באימון בלבד.
- class weights מחושבים מאימון בלבד. Batch size 16 מתאים כנקודת התחלה לכרטיס 6GB.
- 3 epochs לראש הסיווג ועד 15 epochs ל-fine-tuning, עם early stopping לפי validation macro F1. epoch הוא מעבר מלא על תמונות האימון.
- המודל הטוב ביותר נבחר לפי validation בלבד. test משמש להערכה סופית.

אין זמן אימון מובטח: תלוי בכרטיס, בדרייבר ובמחשב. חברו את הלפטופ לחשמל. אם מתקבלת CUDA out of memory, נסו תיקיית פלט חדשה ו-batch קטן יותר:

```bash
python brain_tumor.py train --data "$HOME/Downloads/archive" --output results_batch8 --batch-size 8
```

לבדיקת הרצה קצרה (אינה תוצאת הגשה סופית):

```bash
python brain_tumor.py train --data "$HOME/Downloads/archive" --output quick_check --head-epochs 1 --epochs 0
```

הקוד מסרב לדרוס תיקיית תוצאות קיימת. להרצה נוספת בחרו למשל --output results_run2.
אין חידוש אימון אוטומטי אחרי עצירה; best_model.pt והיסטוריית epochs שהושלמו נשמרים.

## קובצי הפלט

| קובץ | שימוש |
|---|---|
| best_model.pt | המודל המאומן הטוב ביותר |
| predictions_100.csv | בדיוק 100 חיזויים לנספח A: שם תמונה, אמת, חיזוי, הצלחה וציוני המודל |
| predictions_all.csv | חיזויים לכל test שנותר אחרי הבדיקה |
| metrics.json | Accuracy, balanced accuracy, macro F1, precision/recall/F1 לכל מחלקה ומדדי זיהוי בינארי |
| evaluation_summary.txt | סיכום קריא לדוח |
| confusion_matrix.png | מטריצת בלבול |
| learning_curve.png | ביצועי validation לאורך האימון |
| history.csv | loss באימון ומדדי validation |
| manifest_original.csv / manifest_used.csv | קבצי המקור והקבצים שנכללו בניסוי |
| excluded_images.json | כפילויות/תוויות סותרות שהוחרגו |
| run_config.json | פרמטרים, גרסאות, התפלגות מחלקות ומגבלות |
| appendix_selection.csv | 100 התמונות שנבחרו מראש |

## חיזוי תמונה חדשה לאחר האימון

```bash
python brain_tumor.py predict --model results/best_model.pt --image "/full/path/to/image.jpg"
```

המודל מסווג glioma / meningioma / notumor / pituitary. חיזוי נוכחות גידול בינארי משתמש בסכום ציוני שלוש מחלקות הגידול ובסף קבוע 0.5; הוא עשוי להיות שונה מהמרת מחלקת argmax לבינארי. הציונים אינם הסתברויות קליניות מכוילות. הקוד אינו מסמן מיקום גידול או מבצע segmentation.

## התאמה לדוח הקורס

- התחום והמערכת: סיווג תמונות MRI, מוטיבציה, יתרונות ומגבלות — לנסח בדוח.
- הדאטה: לציין M.Y. LEE_77, הקישור המאושר, המקור Masoud Nickparvar, CC BY 4.0, וההתפלגות שהתקבלה בפועל.
- האלגוריתם: לציין PyTorch, EfficientNetB0, transfer learning, אימון הראש ו-fine-tuning, preprocessing, class weights ו-early stopping.
- ההערכה: להעתיק את התוצאות האמיתיות לאחר הרצה, להסביר macro F1 ו-recall לכל מחלקה לצד accuracy ולהוסיף confusion matrix.
- נספח A: predictions_100.csv. לא מאמנים על 100 תמונות בלבד — אלו דוגמאות מתועדות מתוך test.
- ההנחיות דורשות גם קישור לקוד/Notebook; לאחר ההרצה יש לפרסם את הקוד או לשתף קישור לפי דרך ההגשה בקורס. ה-ZIP לבדו אינו קישור ציבורי לקוד.

## מגבלות ונקודות AI FATE

זהו ניסוי לימודי ולא מערכת אבחון. אין כאן אימות על חולים חדשים או בתי חולים אחרים.
הכותרת deduplicated אינה הוכחה לעצמאות מלאה: היוצר השתמש ב-pHash במרחק אפס וציין שייתכנו near-duplicates נוספים. הבדיקה הנוספת בקוד מזהה רק פיקסלים זהים, לא כל דמיון חזותי או חפיפה בין מטופלים.
אין metadata מאומת של גיל/מין/מוצא ולכן אי אפשר לטעון שנבדקה הוגנות בין קבוצות אלו. אפשר להשוות בין מחלקות, אך זה אינו תחליף לבדיקת הוגנות דמוגרפית.
היוצר מציין הטיית מסגור תמונות בין tumor ו-notumor. יש להזכיר סיכון ללמידת מאפייני מקור/צילום במקום פתולוגיה.
אין Grad-CAM בחבילה זו; אין דרישה לכך בהנחיות שלב א׳ שסופקו.
אל תשנו היפרפרמטרים לפי ביצועי test ואל תדווחו על מדדי quick_check כתוצאות סופיות.

## מקורות

- דאטה מאושר: https://www.kaggle.com/datasets/mylee77/brain-tumor-mri-deduplicated-clean-version
- EfficientNetB0: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html
- התקנת PyTorch: https://pytorch.org/get-started/locally/
- הנחיות הקורס: המצגת וטופס דוח שלב א׳ שצורפו לשיחה.
