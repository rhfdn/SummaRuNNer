# SummaRuNNer
paper: [SummaRuNNer](https://arxiv.org/pdf/1611.04230.pdf)

## Clone project
```bash
git clone https://github.com/Baragouine/SummaRuNNer.git
```

## Enter into the directory
```bash
cd SummaRuNNer
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

## Install nltk data
To install nltk data:
  - Open a python console.
  - Type ``` import nltk; nltk.download()```.
  - Download all data.
  - Close the python console.

## Convert initial dataset to valid pandas json
Download the [initial dataset]([https://drive.google.com/file/d/1JgsboIAs__r6XfCbkDWgmberXJw8FBWE/view?usp=sharing](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)) (CNN-DailyMail from Kaggle).  
Copy train.json, val.json and test.json to `./data/cnn_dailymail/raw/` .  
Run the notebook: `00-0-convert_raw_cnndailymail_to_json.ipynb`:

## Compute labels
```bash
python3 ./00-1-compute_label_cnndailymail.py
```
You can adapt this script for other dataset containing texts and summaries.

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
The `pt` files are located in `./checkpoints`, each training result is stored in a different sub directory.  

## Result

### CNN-DailyMail (full-length f1 rouge)
| model | ROUGE-1 | ROUGE-2 | ROUGE-L |  
|:-:    |:-:      |:-:      |:-:      |  
|SummaRuNNer(Nallapati)|39.6 &plusmn; 0.2|16.2 &plusmn; 0.2|35.3 &plusmn; 0.2|  
|RNN_RNN|39.7 &plusmn; 0.0|16.2 &plusmn; 0.0|24.4 &plusmn; 0.0|  
|SIMPLE_CNN_RNN|39.6 &plusmn; 0.0|16.2 &plusmn; 0.0|24.4 &plusmn; 0.0|  
|COMPLEX_CNN_RNN|39.6 &plusmn; 0.0|16.2 &plusmn; 0.0|24.4 &plusmn; 0.0|  
|COMPLEX_CNN_RNN (max_pool)|39.6 &plusmn; 0.0|16.2 &plusmn; 0.0|24.4 &plusmn; 0.0|  
|RES_CNN_RNN|39.6 &plusmn; 0.0|16.2 &plusmn; 0.0|24.4 &plusmn; 0.0|  
|SIMPLE_CNN_RNN without text content (positions only)|39.4 &plusmn; 0.0|16.0 &plusmn; 0.0|24.2 &plusmn; 0.0|  
|SIMPLE_CNN_RNN without positions (text content only)|39.6 &plusmn; 0.0|16.2 &plusmn; 0.0|24.4 &plusmn; 0.0|  
|SIMPLE_CNN_RNN absolute position only|39.4 &plusmn; 0.0|16.0 &plusmn; 0.0|24.3 &plusmn; 0.0|  
|SIMPLE_CNN_RNN relative position only|39.0 &plusmn; 0.0|15.8 &plusmn; 0.0|24.1 &plusmn; 0.0|  
|SIMPLE_CNN_RNN without positions and content|39.6 &plusmn; 0.0|16.2 &plusmn; 0.0|24.4 &plusmn; 0.0|  
|SIMPLE_CNN_RNN without positions and salience|39.6 &plusmn; 0.0|16.2 &plusmn; 0.0|24.4 &plusmn; 0.0|  
|SIMPLE_CNN_RNN without position and novelty|39.6 &plusmn; 0.0|16.2 &plusmn; 0.0|24.4 &plusmn; 0.0|  
|SIMPLE_CNN_RNN without novelty only|**40.0 &plusmn; 0.0**|**16.7 &plusmn; 0.0**|**25.3 &plusmn; 0.0**|  
|SIMPLE_CNN_RNN without salience only|39.6 &plusmn; 0.0|16.2 &plusmn; 0.0|24.4 &plusmn; 0.0|  
|SIMPLE_CNN_RNN without content only|39.6 &plusmn; 0.0|16.2 &plusmn; 0.0|24.4 &plusmn; 0.0|

### RNN_RNN on NYT50 (limited-length ROUGE Recall)
| model | ROUGE-1 | ROUGE-2 | ROUGE-L |  
|:-:    |:-:      |:-:      |:-:      |  
| HeterSUMGraph (Wang) | 46.89 | 26.26 | 42.58 |  
| RNN_RNN | **47.3 &plusmn; 0.0** | **26.7 &plusmn; 0.0** | **35.7\* &plusmn; 0.0** |  

*: maybe the ROUGE-L have changed in the rouge library I use.

### RNN_RNN on general geography, architecture town planning and geology french wikipedia articles (limited-length ROUGE Recall)
| dataset | ROUGE-1 | ROUGE-2 | ROUGE-L |  
|:-:      |:-:      |:-:      |:-:      |  
| Wikipedia-0.5 |31.4 &plusmn; 0.0|10.1 &plusmn; 0.0|20.0 &plusmn; 0.0|  
| Wikipedia-high-25 |24.4 &plusmn; 0.0|6.7 &plusmn; 0.0|15.0 &plusmn; 0.0|  
| Wikipedia-low-25 |32.3 &plusmn; 0.0|12.4 &plusmn; 0.0|21.9 &plusmn; 0.0|  

&ast; Wikipedia-0.5: general geography, architecture town planning and geology french wikipedia articles with len(summary)/len(content) <= 0.5.  
&ast; Wikipedia-high-25: first 25% of general geography, architecture town planning and geology french wikipedia articles sorted by len(summary)/len(content) descending.  
&ast; Wikipedia-low-25: first 25% of general geography, architecture town planning and geology french wikipedia articles sorted by len(summary)/len(content) ascending.  

See [CGN_ExSUM_NER repository](https://github.com/Baragouine/CGN_ExSUM_NER) for wikipedia scraping and preprocessing (that repository conatain script for scrapping and preprocessing).
