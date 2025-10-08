# An Investigation of Memorization Risk in Healthcare Foundation Models (NeurIPS 2025)

![Overview](overview.png)

---

## Abstract
Foundation models trained on de-identified Electronic Health Records (EHRs) hold great potential for advancing clinical AI. However, their ability to memorize and inadvertently disclose sensitive patient information raises significant privacy concerns.  
This repository implements a reproducible evaluation framework to **quantify and contextualize memorization risk** in EHR foundation models.  
It introduces a series of **black-box tests (T1–T6)** that distinguish between benign generalization and harmful patient-level memorization, evaluated on **EHRMamba2**, a public benchmark model trained on MIMIC-IV.

---

## Repository Structure

```

├── embedding_memorization/
│   ├── membership_inference.py     # Implements T4: membership inference attacks
│   └── probing_test.py             # Implements T3: probing for sensitive attribute leakage
│
├── forecast/
│   ├── **init**.py
│   └── ehrmamba2_forecast.ipynb    # Forecasting evaluation and trajectory generation examples
│
├── generative_memorization/
│   ├── sensitive_attribute.py      # Implements T2: sensitive attribute test
│   ├── trajectory_memorization.py  # Implements T1: trajectory reconstruction test
│   └── utils.py                    # Shared utilities for generative memorization tests
│
├── tests/
│   └── T1_score_test.py            # Unit test for similarity metric (dEMD)
│
├── overview.png                    # High-level visual overview of evaluation framework
└── README.md

````

---

## Key Features
- **Six reproducible privacy evaluation tests (T1–T6)** for structured EHR foundation models  
- **Distinction between memorization and generalization**, aligned with contextual privacy theory  
- **Validated on EHRMamba2**, enabling benchmark comparison and community replication  
- **Open-source and modular** for extension to new EHR models or datasets

---

## Authors

**Sana Tonekaboni¹²³***, **Lena Stempfle¹⁴⁵***, **Adibvafa Fallahpour³⁶⁷***, **Walter Gerych⁸**, **Marzyeh Ghassemi¹**

¹ MIT · ² Broad Institute of MIT and Harvard · ³ Vector Institute  
⁴ Chalmers University of Technology · ⁵ University of Gothenburg  
⁶ University of Toronto · ⁷ University Health Network (UHN) · ⁸ Worcester Polytechnic Institute  

\* Equal contribution

---

## Citation
If you use this work, please cite it as follows:

```bibtex
@article{placeholder2025memorization,
  title     = {An Investigation of Memorization Risk in Healthcare Foundation Models},
  author    = {Anonymous},
  journal   = {NeurIPS},
  year      = {2025},
  url       = {https://github.com/...}
}
````

---

**License:** MIT
