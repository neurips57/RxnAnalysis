# Rxn_analysis

# Install

    virtualenv venv -p /opt/anaconda3/bin/python3.8 --no-pip
    source venv/bin/activate
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py pip==19.3.1
    pip install --upgrade pip
    python -m pip install numpy==1.22.3
    pip install rdkit
    pip install torch torchvision torchaudio
    pip install dgl-cu102 dglgo -f https://data.dgl.ai/wheels/repo.html
    https://github.com/neurips57/RxnAnalysis.git
    cd RxnAnalysis

# Dataset processing
Run `main.py` for .csv to .npz conversion with appropriate reaction function. For example, use `generate_sc_total()` to create npz data for SC reaction. In the next step, run `get_data.py` inside `./data` forlder.

# Model Running
Run `run_code.py` with proper `model_*.py` 

# Subset creation
For subset making check `subset_creation/make_subset.ipynb`

# Dataset
All datasets are kept in `./Dataset` folder
