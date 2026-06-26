# MedGuards

The code repository of MedGuards.

## Installation

```commandline
pip install -r requirements.txt
```

For more requirements, please refer to requirements.txt

## Data Preparation

The MEDEC data is from https://github.com/abachaa/MEDEC.

The data has been put in the folder of "data/Multi_Lang/". We labeled the data with additional key_words section.

## How to run the program

For model running, the commandline is:

```commandline
python Doubao_MedGuards.py
```

## Keywords Extraction

For new datasets, we also provide a script to extract the keywords for KPCS metric calculation as shown by extract_keywords.py.
