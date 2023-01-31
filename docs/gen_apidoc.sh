#!/bin/bash
sphinx-apidoc -f -P -o source ../encodermap/ ../encodermap/examples/ -V 3.0.0 -H EncoderMap --templatedir _templates
# make html
# make pdf
