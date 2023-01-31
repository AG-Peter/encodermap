#!/bin/bash
rm -rf runs/
# form some reason --clear-ouput is broken
# It will be fixed in v6.0 of jupyter_contrib_extensions
for file in *.ipynb ; do
    nbstripout $file
    # jupyter nbconvert --clear-output --inplace $file
done
