**party**
========

party is **PA**ge-wise **R**ecognition of **T**ext-y. It is a replacement for
conventional text recognizers in ATR system using the baseline+bounding polygon
line data model where it eliminates the need for bounding polygons.

Installation
------------

::

        $ pip install .


Training
--------

Options are largely identical to those offered by `ketos train`, apart from
supporting only XML training data:

::

        $ party -d cuda train -f xml --workers 32 -t train.lst -e val.lst
