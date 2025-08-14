# Data Directory

This directory contains all datasets used in the Drug-Drug Interaction Prediction project.

## Directory Structure

```
data/
├── raw/                    # Original, unprocessed datasets
│   ├── drugbank/          # DrugBank XML/CSV files
│   ├── chembl/            # ChEMBL bioactivity data
│   ├── faers/             # FDA FAERS adverse event data
│   └── structures/        # SMILES and molecular structure files
├── processed/             # Cleaned and preprocessed data
│   ├── interactions.csv   # Standardized interaction dataset
│   ├── drug_features.csv  # Molecular descriptors and features
│   └── graph_data.pkl     # Graph structures for GNN training
└── external/              # External validation datasets
    ├── twosides/          # TWOSIDES interaction database
    └── sider/             # SIDER side effect database
```

## Data Sources

### Primary Databases

1. **DrugBank** (https://go.drugbank.com/)
   - Comprehensive drug and drug interaction database
   - ~13,000 drug entries with detailed interaction information
   - Required: Academic license for full database access

2. **ChEMBL** (https://www.ebi.ac.uk/chembl/)
   - Large-scale bioactivity database
   - Molecular properties and bioactivity data
   - Free access via API or bulk download

3. **FDA FAERS** (https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html)
   - FDA Adverse Event Reporting System
   - Real-world adverse event data
   - Public access, quarterly updates

4. **PubChem** (https://pubchem.ncbi.nlm.nih.gov/)
   - Chemical structure database
   - SMILES strings and molecular identifiers
   - Free API access

### External Validation

1. **TWOSIDES** (http://tatonetti.com/data/twosides/)
   - Large-scale drug interaction database from FDA labels
   - Statistical significance testing included
   - Academic use permitted

2. **SIDER** (http://sideeffects.embl.de/)
   - Side effect database with drug interaction information
   - Mapped to medical ontologies
   - Free academic use

## Data Collection Instructions

### 1. DrugBank Data
```bash
# Download DrugBank full database (requires license)
# 1. Register at https://go.drugbank.com/
# 2. Request academic license
# 3. Download XML full database
# 4. Place in data/raw/drugbank/
```

### 2. ChEMBL Data
```bash
# Download via ChEMBL API or bulk download
# Focus on drug-like molecules with interaction data
# Extract bioactivity and molecular property data
```

### 3. FAERS Data
```bash
# Download quarterly FAERS data
# Extract drug combination adverse events
# Focus on serious outcomes and drug interactions
```

### 4. Molecular Structures
```bash
# Collect SMILES strings from PubChem
# Map drug names to chemical structures
# Validate structure quality with RDKit
```

## Data Processing Pipeline

1. **Raw Data Validation**
   - Check file integrity and format
   - Validate molecular structures
   - Identify missing or corrupted data

2. **Data Standardization**
   - Standardize drug names and identifiers
   - Normalize interaction severity scales
   - Map to common ontologies (ATC, MeSH)

3. **Feature Engineering**
   - Extract molecular descriptors
   - Calculate drug similarity metrics
   - Generate network topology features

4. **Quality Assessment**
   - Statistical validation of data distributions
   - Identify and handle outliers
   - Assess class imbalance and bias

## Statistical Considerations

### Sample Size Requirements
- Minimum 10,000 drug pairs for reliable training
- Power analysis for detecting clinically significant interactions
- Stratified sampling by drug class and interaction severity

### Bias Assessment
- Publication bias in interaction reporting
- Selection bias in database inclusion criteria
- Temporal bias from changing medical practices

### Validation Strategy
- Hold-out test set (20% of data)
- Time-split validation using interaction discovery dates
- External validation on independent datasets

## Data Usage Guidelines

### Privacy and Ethics
- All datasets contain no personally identifiable information
- FAERS data aggregated to protect patient privacy
- Academic use only for proprietary databases

### Reproducibility
- Document all data processing steps
- Version control for datasets and preprocessing scripts
- Provide checksums for data integrity verification

### Citation Requirements
- Cite original database sources in publications
- Acknowledge data providers and funding sources
- Follow database-specific citation guidelines

## Contact Information

For questions about data sources or processing:
- Technical issues: [your.email@domain.com]
- Data access: Follow individual database guidelines
- Collaboration: Open to academic partnerships