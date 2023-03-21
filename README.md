# SummaRuNNer
paper: [SummaRuNNer](https://arxiv.org/pdf/1611.04230.pdf)

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
Download the [initial dataset](https://drive.google.com/file/d/1JgsboIAs__r6XfCbkDWgmberXJw8FBWE/view?usp=sharing) (DailyMail).  
Copy train.json, val.json and test.json to `./data/ref` .  
Run the python script: `convert_ref_data_to_raw_data.py`:
```bash
python3 ./convert_ref_data_to_raw_data.py
```

## Compute own labels
```bash
python3 ./compute_own_labels_from_raw_data.py
```

## Embeddings
For training you must use glove 100 embeddings, they must have the following path: `data/glove.6B/glove.6B.100d.txt`

## Training
Run `train_RNN_RNN.ipynb` to train the paper model.  
Run `train_SIMPLE_CNN_RNN.ipynb` to train the model whose first RNN is replaced by a single-layer CNN. 
Run `train_COMPLEX_CNN_RNN.ipynb` to train the model whose the first RNN is replaced by a complex CNN (3 layers).  
Run `train_RES_CNN_RNN.ipynb` to train the model whose first RNN is replaced by a CNN that uses residual connections (3 layers).  
  
The other notebooks are used to train SIMPLE_CNN_RNN that have been ablated to see the importance of each component of the model:
 * Run `train_SIMPLE_CNN_RNN_abs_pos_only.ipynb` to train a SIMPLE_CNN_RNN that uses only the absolute position to predict.  
 * Run `train_SIMPLE_CNN_RNN_rel_pos_only.ipynb` to train a SIMPLE_CNN_RNN that uses only the relative position to predict.
 * ...  
  
To find out what these notebooks are for, just look at the file name.

## Result
| model | ROUGE-1 | ROUGE-2 | ROUGE-L | Accuracy |  
|:-:    |:-:      |:-:      |:-:      |:-:       |  
|SummaRuNNer(Nallapati)|26.2|10.8|14.4|?|  
|RNN_RNN|29.5|15.1|19.9|0.795|  
|SIMPLE_CNN_RNN|29.3|15.0|19.8|0.795|  
|COMPLEX_CNN_RNN|29.2|15.0|19.7|0.795|  
|COMPLEX_CNN_RNN (max_pool)|29.2|15.0|19.7|0.795|  
|RES_CNN_RNN|29.2|15.0|19.7|0.795|  
|SIMPLE_CNN_RNN without text content (positions only)|28.6|14.6|19.3|0.794|  
|SIMPLE_CNN_RNN without positions (text content only)|29.2|15.0|19.7|0.795|  
|SIMPLE_CNN_RNN absolute position only|28.6|14.5|19.2|0.794|  
|SIMPLE_CNN_RNN relative position only|28.6|14.5|19.3|0.794|  
|SIMPLE_CNN_RNN without positions and content|29.4|15.1|19.9|0.795|  
|SIMPLE_CNN_RNN without positions and salience|29.2|15.0|19.7|0.796|  
|SIMPLE_CNN_RNN without position and novelty|29.2|15.0|19.7|0.796|  
|SIMPLE_CNN_RNN without position, content and salience (novelty only)|19.4|15.1|19.8|0.796|  
|SIMPLE_CNN_RNN without position, content and novelty (salience only)|29.1|15.0|19.7|0.795|  
|SIMPLE_CNN_RNN without position, salience and novelty (content only)|29.3|15.1|19.8|0.796|

## Influences of batch size
| batch size | training time (s) | Accuracy | ROUGE-1 | ROUGE-2 | ROUGE-L |  
|:-:         |:-:                |:-:       |:-:      |:-:      |:-:      |  
|8|28467|0.795|29.3|15.0|19.8|  
|16|25016|0.795|29.5|15.1|19.9|  
|32|21712|0.795|29.6|15.1|19.9|  
|64|21924|0.794|29.6|15.1|19.9|  


