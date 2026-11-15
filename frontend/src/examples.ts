export interface Example {
  label: string
  kind: 'MCQ' | 'Yes/No/Maybe' | 'Open'
  question: string
  options?: string[]
}

export const EXAMPLES: Example[] = [
  {
    label: 'Flank pain + WBC casts',
    kind: 'MCQ',
    question:
      'A 26-year-old woman presents with a 2-day history of fever to 39.4°C, chills, right flank pain, and dysuria. ' +
      'Examination shows right costovertebral angle tenderness. Urinalysis shows pyuria, bacteriuria, and white blood cell casts. ' +
      'Which of the following is the most likely diagnosis?',
    options: ['Acute cystitis', 'Acute pyelonephritis', 'Nephrolithiasis', 'Acute interstitial nephritis', 'Renal cell carcinoma'],
  },
  {
    label: 'Statins & post-op AF',
    kind: 'Yes/No/Maybe',
    question:
      'Is preoperative statin therapy associated with a reduced incidence of postoperative atrial fibrillation after cardiac surgery?',
    options: ['yes', 'no', 'maybe'],
  },
  {
    label: 'Metformin & CRC',
    kind: 'Open',
    question: 'Does metformin use reduce colorectal cancer risk in type 2 diabetes?',
  },
  {
    label: 'Diabetic albuminuria',
    kind: 'MCQ',
    question:
      'A 58-year-old man with type 2 diabetes has a blood pressure of 148/92 mm Hg on repeated measurements and a urine ' +
      'albumin-to-creatinine ratio of 320 mg/g. Which of the following is the most appropriate first-line antihypertensive?',
    options: ['Hydrochlorothiazide', 'Amlodipine', 'Lisinopril', 'Metoprolol', 'Doxazosin'],
  },
  {
    label: 'H. pylori first line',
    kind: 'Open',
    question: 'What is the recommended first-line treatment for Helicobacter pylori infection in areas of high clarithromycin resistance?',
  },
]
