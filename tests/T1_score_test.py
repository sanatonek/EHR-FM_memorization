from ehr_sim import meausre_distance, get_embed
import random

import ast
import numpy as np
import pandas as pd
import ot
import torch
import json
import os
import polars
import gzip
import sys
import pickle

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random_path = "/projects/ehrmamba_memorization/ehrmamba2/meta_vocab.json"
f = open(random_path) 
token_dict = json.load(f)

ehr_sequence = [
    "AGE//76",
    "DIAGNOSIS//99591",
    "DIAGNOSIS//5990",
    "LAB//50862//2",
    "LAB//50912//2",
    "LAB//50882//2",
    "LAB//50821//2",
    "LAB//50983//0",
    "LAB//51221//0",
    "LAB//50868//0",
    "MEDICATION//Ciprofloxacin//IV",
    "MEDICATION//Vancomycin//IV",
    "MEDICATION//Norepinephrine//IV",
    "PROCEDURE//BloodCulture",
    "PROCEDURE//UrineCulture",
    "LAB//50820//2",
    "LAB//50931//1",
    "LAB//50902//2",
    "MEDICATION//Acetaminophen//Oral",
    "LAB//50813//2",
    "LAB//50814//1",
    "MEDICATION//Pantoprazole//IV",
    "LAB//50878//1",
    "LAB//50971//0",
    "PROCEDURE//ICUTransfer",
    "LAB//50862//1",
    "LAB//50912//1",
    "LAB//50902//1",
    "LAB//50821//1",
    "LAB//50882//1",
    "MEDICATION//NormalSaline//IV",
    "MEDICATION//Insulin//SC",
    "MEDICATION//Heparin//Subcut",
    "TIME//48",
    "LAB//50862//1",
    "LAB//50912//1",
    "LAB//50931//1",
    "LAB//50983//1",
    "LAB//50820//1",
    "LAB//50868//1",
    "MEDICATION//Ciprofloxacin//Oral",
    "LAB//50813//1",
    "LAB//50902//1",
    "LAB//50821//1",
    "PROCEDURE//DailyVitals",
    "LAB//50882//1",
    "LAB//50814//1",
    "DIAGNOSIS//V5866"
]

similar_med_sequence = [
    "AGE//76",
    "DIAGNOSIS//99591",
    "DIAGNOSIS//5990",
    "LAB//50862//2",
    "LAB//50912//2",
    "LAB//50882//2",
    "LAB//50821//2",
    "LAB//50983//0",
    "LAB//51221//0",
    "LAB//50868//0",
    "MEDICATION//Levofloxacin//IV",  # Updated
    "MEDICATION//Linezolid//IV",     # Updated
    "MEDICATION//Dopamine//IV",      # Updated
    "PROCEDURE//BloodCulture",
    "PROCEDURE//UrineCulture",
    "LAB//50820//2",
    "LAB//50931//1",
    "LAB//50902//2",
    "MEDICATION//Ibuprofen//Oral",   # Updated
    "LAB//50813//2",
    "LAB//50814//1",
    "MEDICATION//Omeprazole//IV",    # Updated
    "LAB//50878//1",
    "LAB//50971//0",
    "PROCEDURE//ICUTransfer",
    "LAB//50862//1",
    "LAB//50912//1",
    "LAB//50902//1",
    "LAB//50821//1",
    "LAB//50882//1",
    "MEDICATION//NormalSaline//IV",
    "MEDICATION//Insulin//SC",
    "MEDICATION//Heparin//Subcut",
    "TIME//48",
    "LAB//50862//1",
    "LAB//50912//1",
    "LAB//50931//1",
    "LAB//50983//1",
    "LAB//50820//1",
    "LAB//50868//1",
    "MEDICATION//Amoxicillin-Clavulanate//Oral",  # Updated
    "LAB//50813//1",
    "LAB//50902//1",
    "LAB//50821//1",
    "PROCEDURE//DailyVitals",
    "LAB//50882//1",
    "LAB//50814//1",
    "DIAGNOSIS//V5866"
]

irrelevant_med_sequence = [
    "AGE//76",
    "DIAGNOSIS//99591",
    "DIAGNOSIS//5990",
    "LAB//50862//2",
    "LAB//50912//2",
    "LAB//50882//2",
    "LAB//50821//2",
    "LAB//50983//0",
    "LAB//51221//0",
    "LAB//50868//0",
    "MEDICATION//Fluoxetine//Oral",     # Antidepressant (SSRI)
    "MEDICATION//Hydroxyzine//Oral",    # Antihistamine, anxiety treatment
    "MEDICATION//Methotrexate//Oral",   # Immunosuppressant for RA or cancer
    "PROCEDURE//BloodCulture",
    "PROCEDURE//UrineCulture",
    "LAB//50820//2",
    "LAB//50931//1",
    "LAB//50902//2",
    "MEDICATION//Risperidone//Oral",    # Antipsychotic (schizophrenia/bipolar)
    "LAB//50813//2",
    "LAB//50814//1",
    "MEDICATION//Adalimumab//SC",       # TNF inhibitor for autoimmune diseases
    "LAB//50878//1",
    "LAB//50971//0",
    "PROCEDURE//ICUTransfer",
    "LAB//50862//1",
    "LAB//50912//1",
    "LAB//50902//1",
    "LAB//50821//1",
    "LAB//50882//1",
    "MEDICATION//Clobetasol//Topical",   # Corticosteroid for psoriasis/eczema
    "MEDICATION//Metformin//Oral",       # Anti-diabetic agent
    "MEDICATION//Alendronate//Oral",     # Osteoporosis treatment
    "TIME//48",
    "LAB//50862//1",
    "LAB//50912//1",
    "LAB//50931//1",
    "LAB//50983//1",
    "LAB//50820//1",
    "LAB//50868//1",
    "MEDICATION//Gabapentin//Oral",      # Neuropathic pain or epilepsy
    "LAB//50813//1",
    "LAB//50902//1",
    "LAB//50821//1",
    "PROCEDURE//DailyVitals",
    "LAB//50882//1",
    "LAB//50814//1",
    "DIAGNOSIS//V5866"
]


irrelevant_lab_sequence = [
    "AGE//76",
    "DIAGNOSIS//99591",
    "DIAGNOSIS//5990",
    "LAB//50862//0",  # Irrelevant WBC - Low instead of High
    "LAB//50912//0",  # Irrelevant Creatinine - Low instead of High
    "LAB//50882//1",  # Irrelevant Lactate - Normal instead of High
    "LAB//50821//0",  # Irrelevant BUN - Low instead of High
    "LAB//50983//2",  # Irrelevant Potassium - High instead of Low
    "LAB//51221//1",  # Irrelevant Hematocrit - Normal instead of Low
    "LAB//50868//2",  # Irrelevant Platelets - High instead of Low
    "MEDICATION//Ciprofloxacin//IV",
    "MEDICATION//Vancomycin//IV",
    "MEDICATION//Norepinephrine//IV",
    "PROCEDURE//BloodCulture",
    "PROCEDURE//UrineCulture",
    "LAB//50820//0",  # Irrelevant Bilirubin - Low instead of High
    "LAB//50931//0",  # Irrelevant Sodium - Low instead of Normal
    "LAB//50902//2",  # Irrelevant Bicarbonate - High instead of Normal
    "MEDICATION//Acetaminophen//Oral",
    "LAB//50813//0",  # Irrelevant ALT - Low instead of High
    "LAB//50814//2",  # Irrelevant AST - High instead of Normal
    "MEDICATION//Pantoprazole//IV",
    "LAB//50878//0",  # Irrelevant INR - Low instead of Normal
    "LAB//50971//2",  # Irrelevant Hemoglobin - High instead of Low
    "PROCEDURE//ICUTransfer",
    "LAB//50862//2",  # Irrelevant WBC - High after ICU, should be normal
    "LAB//50912//2",  # Irrelevant Creatinine - High after ICU, should be improving
    "LAB//50902//0",  # Irrelevant Bicarbonate - Low instead of improving
    "LAB//50821//0",  # Irrelevant BUN - Low after ICU, should be normal
    "LAB//50882//0",  # Irrelevant Lactate - Low instead of improving
    "MEDICATION//NormalSaline//IV",
    "MEDICATION//Insulin//SC",
    "MEDICATION//Heparin//Subcut",
    "TIME//48",
    "LAB//50862//0",  # Irrelevant WBC - Low instead of stabilizing
    "LAB//50912//0",  # Irrelevant Creatinine - Low instead of stabilizing
    "LAB//50931//2",  # Irrelevant Sodium - High instead of normal
    "LAB//50983//0",  # Irrelevant Potassium - Low instead of normal
    "LAB//50820//2",  # Irrelevant Bilirubin - High instead of stabilizing
    "LAB//50868//0",  # Irrelevant Platelets - Low instead of improving
    "MEDICATION//Ciprofloxacin//Oral",
    "LAB//50813//2",  # Irrelevant ALT - High instead of stabilizing
    "LAB//50902//1",  # Irrelevant Bicarbonate - Normal instead of stable
    "LAB//50821//0",  # Irrelevant BUN - Low instead of improving
    "PROCEDURE//DailyVitals",
    "LAB//50882//2",  # Irrelevant Lactate - High instead of normal
    "LAB//50814//0",  # Irrelevant AST - Low instead of stable
    "DIAGNOSIS//V5866"
]

overestiate_sequence = [
    "AGE//76",
    "DIAGNOSIS//99591",
    "DIAGNOSIS//5990",
    "LAB//50862//2",
    "LAB//50912//2",
    "LAB//50882//2",
    "LAB//50821//2",
    "LAB//50983//0",
    "LAB//51221//0",
    "LAB//50868//0",
    "MEDICATION//Ciprofloxacin//IV",
    "MEDICATION//Vancomycin//IV",
    "MEDICATION//Norepinephrine//IV",
    "PROCEDURE//BloodCulture",
    "PROCEDURE//UrineCulture",
    "LAB//50820//2",
    "LAB//50931//1",
    "LAB//50902//2",
    "MEDICATION//Acetaminophen//Oral",
    "LAB//50813//2",
    "LAB//50814//1",
    "MEDICATION//Pantoprazole//IV",
    "LAB//50878//1",
    "LAB//50971//0",
    "PROCEDURE//ICUTransfer",
    "LAB//50862//1",
    "LAB//50912//1",
    "LAB//50902//1",
    "LAB//50821//1",
    "LAB//50882//1",
    "MEDICATION//NormalSaline//IV",
    "MEDICATION//Insulin//SC",
    "MEDICATION//Heparin//Subcut",
    "TIME//48",
    "LAB//50862//1",
    "LAB//50912//1",
    "LAB//50931//1",
    "LAB//50983//1",
    "LAB//50820//1",
    "LAB//50868//1",
    "MEDICATION//Ciprofloxacin//Oral",
    "LAB//50813//1",
    "LAB//50902//1",
    "LAB//50821//1",
    "PROCEDURE//DailyVitals",
    "LAB//50882//1",
    "LAB//50814//1",
    "DIAGNOSIS//V5866",
    
    # 20 Random Tokens
    "LAB//50900//2",
    "PROCEDURE//Echo",
    "MEDICATION//Amoxicillin//Oral",
    "LAB//50865//0",
    "TIME//12",
    "LAB//50989//1",
    "DIAGNOSIS//25000",
    "PROCEDURE//MRI_Brain",
    "MEDICATION//Aspirin//Oral",
    "LAB//50921//2",
    "LAB//50834//1",
    "PROCEDURE//CT_Chest",
    "MEDICATION//Metformin//Oral",
    "LAB//50972//0",
    "TIME//24",
    "DIAGNOSIS//4019",
    "LAB//50955//1",
    "MEDICATION//Hydrochlorothiazide//Oral",
    "PROCEDURE//ABG",
    "LAB//50882//2"
]

p2_sequence = [
    "AGE//64",
    "DIAGNOSIS//25000",  # Type 2 Diabetes Mellitus
    "DIAGNOSIS//4019",   # Hypertension
    "DIAGNOSIS//70715",  # Diabetic foot ulcer
    "DIAGNOSIS//73007",  # Osteomyelitis of foot
    "LAB//50862//1",     # WBC - Normal (early infection)
    "LAB//50912//1",     # Creatinine - Normal (early CKD)
    "LAB//50882//2",     # Lactate - Elevated
    "LAB//50983//1",     # Potassium - Normal
    "LAB//50821//1",     # BUN - Normal
    "MEDICATION//Piperacillin-Tazobactam//IV",  # Broad-spectrum antibiotic
    "MEDICATION//Vancomycin//IV",               # MRSA coverage
    "PROCEDURE//BloodCulture",
    "PROCEDURE//WoundCulture",
    "LAB//50820//1",     # Bilirubin - Normal
    "LAB//50878//1",     # INR - Normal
    "MEDICATION//Insulin//SC",                  # Glycemic control
    "LAB//50971//1",     # Hemoglobin - Normal
    "LAB//51221//0",     # Hematocrit - Low
    "LAB//50813//1",     # ALT - Normal
    "PROCEDURE//CT_Foot",
    "PROCEDURE//SurgicalDebridement",
    "LAB//50902//2",     # Bicarbonate - High (DKA risk)
    "LAB//50931//1",     # Sodium - Normal
    "TIME//24",          # 24 hours after debridement
    "LAB//50862//1",     # WBC - Normal after antibiotics
    "LAB//50882//1",     # Lactate - Normalizing
    "LAB//50912//2",     # Creatinine - Elevated (CKD progression)
    "MEDICATION//Metformin//Oral",               # Diabetes maintenance
    "MEDICATION//Lisinopril//Oral",              # Hypertension control
    "MEDICATION//Aspirin//Oral",                  # Cardiovascular protection
    "TIME//48",
    "LAB//50862//1",     # WBC - Stable
    "LAB//50821//1",     # BUN - Stable
    "LAB//50983//1",     # Potassium - Stable
    "MEDICATION//Ciprofloxacin//Oral",           # Oral antibiotic for discharge
    "LAB//50931//1",     # Sodium - Normal
    "LAB//50813//1",     # ALT - Normal
    "LAB//50902//1",     # Bicarbonate - Normal
    "PROCEDURE//DailyVitals",
    "LAB//51221//1",     # Hematocrit - Improving
    "MEDICATION//Gabapentin//Oral",               # Neuropathic pain control
    "MEDICATION//Clopidogrel//Oral",              # Antiplatelet therapy
    "DIAGNOSIS//5853",   # CKD Stage 3
    "PROCEDURE//DischargeSummary"
]



ehr_labels = token_convertor(token_dict, ehr_sequence)
samples_embeddings = get_embed(ehr_labels, model='bert')
time_weight_gt = generate_time_counter(ehr_labels)
dist = meausre_distance(s_true=samples_embeddings, s_pred=samples_embeddings, embedded=True)
print("Distance to self: ", dist)

p2_labels = token_convertor(token_dict, p2_sequence)
p2_embeddings = get_embed(p2_labels, model='bert')
# time_weight_gt = generate_time_counter(ehr_labels)
dist = meausre_distance(s_true=samples_embeddings, s_pred=p2_embeddings, embedded=True)
print("Distance to other patients: ", dist)

s1_labels = token_convertor(token_dict, ehr_sequence[:45])
s1_embeddings = get_embed(s1_labels, model='bert') 
dist_1 = meausre_distance(s_true=samples_embeddings, s_pred=s1_embeddings, embedded=True)
print("Distance to half: ", dist_1)

overestimate_labels = token_convertor(token_dict, overestiate_sequence[:55])
overestimate_embeddings = get_embed(overestimate_labels, model='bert') 
dist_1 = meausre_distance(s_true=samples_embeddings, s_pred=overestimate_embeddings, embedded=True)
print("Distance to over-estimate: ", dist_1)

similar_med_labels = token_convertor(token_dict, similar_med_sequence)
similar_med_embeddings = get_embed(similar_med_labels, model='bert') 
dist_3 = meausre_distance(s_true=samples_embeddings, s_pred=similar_med_embeddings, embedded=True)
print("Distance to similar meds: ", dist_3)

irrelevant_med_labels = token_convertor(token_dict, irrelevant_med_sequence)
irrelevant_med_embeddings = get_embed(irrelevant_med_labels, model='bert') 
dist_4 = meausre_distance(s_true=samples_embeddings, s_pred=irrelevant_med_embeddings, embedded=True)
print("Distance to irrelevant meds: ", dist_4)

irrelevant_lab_labels = token_convertor(token_dict, irrelevant_lab_sequence)
irrelevant_lab_embeddings = get_embed(irrelevant_lab_labels, model='bert') 
dist_5 = meausre_distance(s_true=samples_embeddings, s_pred=irrelevant_lab_embeddings, embedded=True)
print("Distance to irrelevant labs: ", dist_5)



