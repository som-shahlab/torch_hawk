API Reference
=============

Command Line Program
-------------------------------

.. py:attribute:: meds_reader_convert [source_meds_path] [destination_path] --num_threads

    Convert a MEDS dataset to a meds_reader PatientDatabase.

    See https://github.com/Medical-Event-Data-Standard/meds for the details of the expected input format.

    :param str source_meds_path: The path to the source MEDS dataset.
    :param str destination_path: The path of where to write the resulting meds_reader PatientDatabase.
    :param int num_threads: The number of threads to use.
    
-----

Python Module
-------------

.. automodule:: meds_reader

.. autoclass:: PatientDatabase
    :members:
    :special-members: __len__, __iter__, __getitem__


.. autoclass:: Patient
    :members:

.. autoclass:: Event
    :members:
    :special-members: __getattr__