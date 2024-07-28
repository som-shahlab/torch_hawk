## meds_reader: A Pythonic interface for MEDS datasets

[![pypy](https://img.shields.io/pypi/v/meds_reader.svg)](https://pypi.org/project/meds_reader/) [![docs](https://readthedocs.org/projects/meds_reader/badge/?version=latest)](https://meds-reader.readthedocs.io/en/latest/) [![Build wheels](https://github.com/EthanSteinberg/meds_reader/actions/workflows/python-build.yml/badge.svg?branch=main)](https://github.com/EthanSteinberg/meds_reader/actions/workflows/python-build.yml?query=branch%3Amain) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

meds_reader is a fast and easy-to-use library for reading and processing patient data stored in [MEDS (Medical Event Data Standard)](https://github.com/Medical-Event-Data-Standard/) format using a Python-native API.

Inspired by HuggingFace Datasets, meds_reader transforms MEDS datasets into collections of Python objects, that can then be processed using normal Python functions.

> [!WARNING]
> meds_reader is currently in early stage development and the API is still in flux. Please leave feedback and use at your own risk!
 
### Example

```bash
# First we need to convert to a database
# That requires running the following command in a terminal
$ meds_reader_convert mimic_meds mimic_meds_reader
```

```python
# Now we can use Python to interact with our data
import meds_reader

# We can now construct a database
database = meds_reader.PatientDatabase("mimic_meds_reader")

# Databases are dict-like objects that store patient data

# We can iterate through all the patient ids in the database
for patient_id in database:

    # We can retrieve patient data given a patient_id
    patient = database[patient_id]

    # Patient data can be manipulated with normal Python operations
    print(patient.patient_id)
    for event in patient.events:
        print(event.time, event.code)
```

<!-- Start OnlyGithub -->

To learn more, see our [full documentation](https://meds-reader.readthedocs.io/en/latest/).

<!-- End OnlyGithub -->

### Installation

meds_reader can be installed using pip.

```bash
pip install meds_reader
```

### Interactive Demo

An interactive demo can be found at [Google Colab](https://colab.research.google.com/drive/1R1LrDIzhQyWldQWM0lyfjeF_n9I_iZT3?usp=sharing).
