# AI4D Baamtu Datamation - Automatic Speech Recognition in Wolof Solution

The repository contains the 4th place solution to Zindi's low-resource automatic speech recognition [competition](https://zindi.africa/competitions/ai4d-baamtu-datamation-automatic-speech-recognition-in-wolof). The goal of the competition was to build a system that allows people to navigate public transport without being able to read or speak French - the language heavily used in the naming of local landmarks.

The solution consists of a finetuned `wav2vec2-xlsr` model, with predictions decoded using an n-gram language model and a nearest-neighbour search. Read the short description of the solution [here](https://zindi.africa/competitions/ai4d-baamtu-datamation-automatic-speech-recognition-in-wolof/discussions/6290)

## Hardware
The script was tested on a high-RAM P100 Google Colab session. The full run should take about 6 hours.

## Steps
1. Place files in `/content/zindi-ai4d-wolof`
2. Place competition data in `/content/zindi-ai4d-wolof/data`
3. Run `run.sh`

The output file is `/content/zindi-ai4d-wolof/submission.csv`
