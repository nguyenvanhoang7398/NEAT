#Neural Side Effect Discovery from User Credibility and Experience-Assessed Online Health Discussions

Full workshop paper is available [here](https://www.aclweb.org/anthology/W18-5602/)

## Getting started
Download People-On-Drugs dataset [here](https://www.mpi-inf.mpg.de/impact/peopleondrugs/), unzip and move to `data/pod`

Download Glove [here](https://nlp.stanford.edu/projects/glove/), unzip and move to `data/glove.840B.300d.txt`

Neural ADR extractor can be found [here](https://github.com/Deep1994/An-Attentive-Neural-Model-for-labeling-Adverse-Drug-Reactions)

Extract side effects
```
python side_effects.py
```

Preprocess People-On-Drugs into docs
```
python pod.py
```

Run model training
```
python train.py --model_name=<neat_lstm|neat_wpe|neat_wpeu|neat_full|neat_cnn|neat_cnn_wpe|neat_cnn_wpeu|neat_fulll>
```

Run ADR extraction benchmark
```
python adr_extraction.py
```

Output of 3 approaches: UMLS tagging, ADR extractor - Ding et al. (2018), and NEAT's Attention for 5 drugs Alprazolam, 
Ibuprofen, Levothyroxine, Metoformin, Omeprazole are available in `test_alprazolam.json`, `test_ibuprofen.json`, 
`test_levothyroxine.json`, `test_metoformin.json`, `test_omeprazole.json` under the keys `umls`, `neural` and `neat` 
respectively.

Bag-of-word Random Forest implementation is available at `baseline.py`

## Environments
```
python==3.7
nltk==3.2.1
keras==2.1.3
tensorflow==1.13.1
spacy==2.2.3
pytorch=1.3.1
```

