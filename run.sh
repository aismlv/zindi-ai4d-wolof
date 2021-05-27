#!/bin/bash
pip3 install -r requirements.txt --upgrade

# install kenlm and ctc-decoder
cd /tmp && git clone https://github.com/kpu/kenlm && cd kenlm && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make install -j$(nproc)
pip3 install https://github.com/kpu/kenlm/archive/master.zip
pip3 install git+https://github.com/ynop/py-ctc-decode --no-dependencies --upgrade

cd /content/zindi-ai4d-wolof
mkdir temp

python3 preprocess.py
# fit wav2vec2 on train
python3 train.py
# pseudo-label test
python3 predict.py --do-expand
# fit wav2vec2 on all data
python3 train.py data.train_dataset="train.dataset.expanded" run_name="step-2"
# predict
python3 predict.py
