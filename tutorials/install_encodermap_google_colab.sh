#!/bin/bash

vercomp () {
    if [[ $1 == $2 ]]
    then
        echo 0
        exit
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    # fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++))
    do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++))
    do
        if [[ -z ${ver2[i]} ]]
        then
            # fill empty fields in ver2 with zeros
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]}))
        then
            echo 1
            exit
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]}))
        then
            echo 2
            exit
        fi
    done
    echo 0
    exit
}

# Get the current python version
echo "I will install EncoderMap in your Google colab notebook. Please stand by..."
current_py_ver=$(python -V)
current_py_ver=${current_py_ver#"Python "}
echo "Your Google colab notebook runs python $current_py_ver"

comp=$(vercomp ${current_py_ver} 3.9.0)
if [ "$comp" = "2" ] ; then
echo "EncoderMap needs at least python 3.9. I will install that version now."

# update apt and install python
apt-get update -y > /dev/notebook
apt-get install python3.9 > /dev/null
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 > /dev/null
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 > /dev/null

# make the Colab Kernel run with python
apt-get install python3.9-distutil
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
python -m pip install ipython ipython_genutils ipykernel jupyter_console prompt_toolkit httplib2 astor
ln -s /usr/local/lib/python3.8/dist-packages/google /usr/local/lib/python3.9/dist-packages/google

# check version
new_py_ver=$(python -V)
new_py_ver=${new_py_ver#"Python "}
echo -e "Succesfully installed python version $new_py_ver$. You can now also change the current python version by calling\n\n!sudo update-alternatives --config python3"
else
echo "This python version is sufficient for EncoderMap. I will install EncoderMap now"
fi

echo "Installing pip packages for EncoderMap"
pip install -r https://raw.githubusercontent.com/AG-Peter/encodermap/main/requirements.txt
pip install -r https://raw.githubusercontent.com/AG-Peter/encodermap/main/md_requirements.txt
git clone https://github.com/AG-Peter/encodermap.git
cd encodermap
pip install .
