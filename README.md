# SummaRuNNer

## Clone project
```bash
git clone https://github.com/Baragouine/SummaRuNNer.git
```

## Create environnement
```bash
conda create --name SummaRuNNer python=3.9
```

## Activate environnement
```bash
conda activate SummaRuNNer
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Convert initial dataset to valid pandas json
Download the [initial dataset](https://drive.google.com/file/d/1JgsboIAs__r6XfCbkDWgmberXJw8FBWE/view?usp=sharing).  
Copy train.json, val.json and test.json to `./data/ref` .  
Run the python script: `convert_ref_data_to_raw_data.py`:
```bash
python3 ./convert_ref_data_to_raw_data.py
```

## Compute own labels
```bash
python3 ./compute_own_labels_from_raw_data.py
```

## Eval RNN_RNN
accuracy = 0.7947603395657316+/-0.0003403645399815157
rouge1 = 0.2952507457561455+/-0.0006882599953940415
rouge2 = 0.15091452229881022+/-0.000400575299884331
rougeL = 0.198735038961782+/-0.0005179696499819547