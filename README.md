# Rxn_analysis

# Install

    virtualenv venv -p /opt/anaconda3/bin/python3.8 --no-pip
    source venv/bin/activate
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py pip==19.3.1
    pip install --upgrade pip
    pip install numpy
    pip install rdkit
    pip install torch torchvision torchaudio
    pip install dgl-cu102 dglgo -f https://data.dgl.ai/wheels/repo.html

# Dataset processing
Run #main.py# for .csv to .npz conversion with appropriate reaction function. 
