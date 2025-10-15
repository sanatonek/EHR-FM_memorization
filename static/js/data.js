const paper = {
  metadata: {
    title: "An Investigation of Memorization Risk in Healthcare Foundation Models",
    description:
      "Black-box tests to assess memorization risks in EHR foundation models, probing generative and embedding leakage and distinguishing memorization from generalization for patient-level privacy risk.",
    keywords: [
      "EHR Foundation Models",
      "Memorization",
      "Privacy",
      "Membership Inference",
      "Healthcare AI",
      "Black-box Evaluation"
    ],
  },

  navbar: {
    home_link: "https://bowang-lab.github.io",
    more_research: [
      { name: "MIT", link: "https://www.mit.edu" },
      { name: "Vector Institute", link: "https://vectorinstitute.ai" },
      { name: "University of Toronto", link: "https://www.utoronto.ca" }
    ],
  },

  authors: [
    { name: "Sana Tonekaboni",    superscript: "1,2,3*", website: "mailto:stonekab@mit.edu" },
    { name: "Lena Stempfle",      superscript: "1,4,5*", website: "mailto:stempfle@mit.edu" },
    { name: "Adibvafa Fallahpour",superscript: "3,6,7*", website: "https://www.linkedin.com/in/adibvafa-fallahpour" },
    { name: "Walter Gerych",      superscript: "8",      website: "mailto:wgerych@wpi.edu" },
    { name: "Marzyeh Ghassemi",   superscript: "1",      website: "mailto:mghassem@mit.edu" },
  ],

  affiliations: [
    { number: "1", name: "Massachusetts Institute of Technology (MIT)",         logo: "static/images/mit.png" },
    { number: "2", name: "Broad Institute of MIT and Harvard",                  logo: "static/images/broad.png" },
    { number: "3", name: "Vector Institute",                                    logo: "static/images/vector.png" },
    { number: "4", name: "Chalmers University of Technology",                   logo: "static/images/chalmers.png" },
    { number: "5", name: "University of Gothenburg",                            logo: "static/images/gothenburg.png" },
    { number: "6", name: "University of Toronto",                               logo: "static/images/uoft.png" },
    { number: "7", name: "University Health Network (UHN)",                     logo: "static/images/uhn.png" },
    { number: "8", name: "Worcester Polytechnic Institute",                     logo: "static/images/worcester.png" },
  ],

  author_notes: {
    equal_contribution: "Equal Contribution",
    equal_advising: "", // none in this paper; preserve style
  },

  link_items: [
    {
      name: "arXiv",
      icon: "ai ai-arxiv",
      link: "https://github.com/sanatonek/EHR-FM_memorization"
    },
    {
      name: "GitHub",
      icon: "fab fa-github",
      link: "https://github.com/sanatonek/EHR-FM_memorization"
    },
    {
      name: "NeurIPS 2025",
      icon: "fas fa-globe",
      link: "https://neurips.cc/virtual/2025/poster/118370"
    }
  ],

  content: {
    abstract:
      "Foundation models trained on large-scale de-identified electronic health records (EHRs) hold promise for clinical applications. However, their capacity to memorize patient information raises important privacy concerns. In this work, we introduce a suite of black-box evaluation tests to assess privacy-related memorization risks in foundation models trained on structured EHR data. Our framework includes methods for probing memorization at both the embedding and generative levels, and aims to distinguish between model generalization and harmful memorization in clinically relevant settings. We contextualize memorization in terms of its potential to compromise patient privacy, particularly for vulnerable subgroups. We validate our approach on a publicly available EHR foundation model and release an open-source toolkit to facilitate reproducible and collaborative privacy assessments in healthcare AI.",
  
    contributions: [
      "<strong>Black-box evaluation suite (T1–T6):</strong> Practical tests spanning <em>generative</em> leakage (trajectory reconstruction, sensitive attribute prediction) and <em>embedding</em> leakage (probing, membership inference).",
      "<strong>Risk-aware framing:</strong> Assesses privacy through contextual integrity, prioritizing patient-level harm rather than raw leakage counts.",
      "<strong>Memorization vs. generalization:</strong> Perturbation (T5) and subgroup (T6) tests to separate individual memorization from population-level trends.",
      "<strong>Toolkit & reproducibility:</strong> Open-source implementation for auditing EHR-FMs under prompt-only access.",
      "<strong>Empirical validation:</strong> Demonstration on a public EHR foundation model trained on structured codes."
    ],
  
    // All main-paper figures and tables
    sections: [
      {
        title: "Evaluation Framework and Objectives (T1–T6)",
        image: "./static/images/figure1.png",
        alt: "Schematic of tests T1–T6 grouped by two objectives",
        caption:
          "Figure 1: Our tests for EHR-FM memorization. Objective I quantifies leakage via generative (T1–T2) and embedding (T3–T4) tests. Objective II evaluates privacy risk with perturbation (T5) and subgroup (T6) analyses to distinguish patient-level memorization from generalization.",
        text:
          "We formalize two objectives: (I) measure whether prompts can trigger disclosure of training information in generations or embeddings, and (II) determine whether any leakage indicates harmful, patient-specific memorization versus acceptable population knowledge."
      },
      {
        title: "Trajectory Memorization (T1): Generated vs. Ground-Truth Trajectories",
        image: "./static/images/figure2.png",
        alt: "Distance curves across prompts and token frequency profile for best trajectory",
        caption:
          "Figure 2: (a) Average distance between generated and true trajectories across prompt setups (Random, Static, 10/20/50 codes). (b) Top-predicted token frequencies reveal bias toward frequent, routine codes (e.g., common labs) rather than rare/sensitive events.",
        text:
          "Providing more prompt context decreases trajectory distance, indicating better reconstruction. However, low distance often reflects benign clinical regularities (e.g., common labs/time tokens), so distance alone does not map directly to privacy risk."
      },
      {
        title: "Sensitive Attribute Leakage (T2): Likelihood Under Increasing Prompt Context",
        image: "./static/images/table1.png",
        alt: "Table summarizing AUROC/AUPRC/precision/recall for sensitive attributes at varying prompt lengths",
        caption:
          "Table 1: Sensitivity test for infectious disease, substance abuse, and mental health. With demographics-only prompts (Static), predictions are near random. As prompt length increases (10/20/50 codes), leakage likelihood rises, highlighting risk from richer patient context.",
        text:
          "We remove direct sensitive codes from prompts and test whether the model emits them anyway. Prompts that push predicted sensitive codes above a conservative threshold are flagged for further privacy analysis."
      },
      {
        title: "Embedding Probing (T3) and Membership Inference (T4)",
        image: "./static/images/figure3.png",
        alt: "Left: random-code frequency vs. training distribution; Right: MI score distributions",
        caption:
          "Figure 3: Left—randomly generated code frequencies compared with the training distribution. Right—membership inference detection scores show only slight, non-significant separation between member and non-member samples.",
        text:
          "Probing sensitive attributes from frozen embeddings yields near-random AUROC across prompt lengths in this setup, suggesting limited embedding leakage. MI based on confidence is weak here; risk becomes concerning if similar signals emerge with minimal input information."
      },
      {
        title: "Perturbation Test (T5): Disentangling Memorization from Generalization",
        image: "./static/images/figure4.png",
        alt: "Probability of sensitive predictions versus age for fixed prompts",
        caption:
          "Figure 4: Predicted probability of sensitive codes as a function of age while holding prompts fixed. Stability around the original identifiers suggests generalized patterns; sharp changes near the original identifiers indicate patient-specific memorization.",
        text:
          "We perturb personal identifiers (e.g., age) in prompts that triggered sensitive predictions in T2. If sensitive predictions collapse under small perturbations, the original behavior likely reflects memorization of an individual case rather than population-level reasoning."
      },
      {
        title: "Sub-population Test (T6): Memorization Risk in Rare or Identifiable Cohorts",
        image: "./static/images/figure5.png",
        alt: "Distributions of predicted probabilities for sensitive attributes under rare-condition prompts",
        caption:
          "Figure 5: Distribution of predicted probabilities for sensitive attributes when prompting with rare diagnoses/procedures. Low scores imply limited risk from rarity alone; high scores flag potential privacy concerns requiring case-by-case review.",
        text:
          "We analyze rare-condition prompts and elderly patients (long-tail cohorts). Elevated leakage within identifiable subgroups indicates amplified privacy risk and motivates targeted safeguards before releasing models."
      }
    ],
  
    conclusion:
      "We provide a practical, risk-aware framework to evaluate memorization in EHR foundation models using only black-box access. By combining generative and embedding tests with perturbation and subgroup analyses, we distinguish harmful patient-level memorization from acceptable generalization. On a public EHR-FM, we find that leakage increases with richer prompts and can concentrate in identifiable cohorts. Our open-source tooling enables reproducible audits and complements post-training safety layers, red-teaming, and retraining to harden EHR-FMs prior to deployment.",
  },
  
  // Keep BibTeX empty for now
  bibtex: ``,  
};
