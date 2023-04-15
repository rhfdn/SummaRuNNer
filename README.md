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
The `pt` files are located in `./checkpoints`, each training result is stored in a different sub directory.  

## Result

### CNN/DailyMail
New values:
#### RNN_RNN
accuracy = 0.7976200912564225+/-0.0004832340762185265  
rouge1 = 0.33276916772892823+/-0.0015953728047048649  
rouge2 = 0.1650124669069412+/-0.0005460987962798178  
rougeL = 0.21914196049006843+/-0.0008741201553995432  

#### SIMPLE_CNN_RNN
accuracy = 0.7970514595349885+/-0.0006736440696233371  
rouge1 = 0.3339824317943772+/-0.0010694029431169377  
rouge2 = 0.16529564971790509+/-0.0006236033491182214  
rougeL = 0.21971106489318834+/-0.0006360028902625464  

#### COMPLEX_CNN_RNN (avg_pool)
accuracy = 0.7971761718946562+/-0.0005915452463302269  
rouge1 = 0.3324963352466784+/-0.0022225250902709456  
rouge2 = 0.16490786526510912+/-0.0008031419255145703  
rougeL = 0.21900282267006227+/-0.001227515328570861  

#### COMPLEX_CNN_RNN (max_pool)
accuracy = 0.7957616492316657+/-0.0014121695141981394  
rouge1 = 0.33093665491579627+/-0.0019473995929805597  
rouge2 = 0.1633502583908628+/-0.0012589288836950604  
rougeL = 0.21742414108726243+/-0.0014528877766677758  

#### RES_CNN_RNN
accuracy = 0.7976091896259884+/-0.0006301527081488897  
rouge1 = 0.33340365623882723+/-0.0015766784674160301  
rouge2 = 0.16518844272974606+/-0.0005769727919436565  
rougeL = 0.21941753965878888+/-0.0008228038903521313  

With bugs:
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
|SIMPLE_CNN_RNN without position, content and salience (novelty only)|29.4|15.1|19.8|0.796|  
|SIMPLE_CNN_RNN without position, content and novelty (salience only)|29.1|15.0|19.7|0.795|  
|SIMPLE_CNN_RNN without position, salience and novelty (content only)|29.3|15.1|19.8|0.796|

### NYT50
#### RNN_RNN
accuracy = 0.7687323796624504+/-0.014930723042429778  
rouge1 = 0.33962717772701917+/-0.008494910491265708  
rouge2 = 0.16428411138339447+/-0.009686278224850171  
rougeL = 0.22178566520400525+/-0.009984206003256462  

## Influences of batch size on CNN/DailyMail
| batch size | training time (s) | Accuracy | ROUGE-1 | ROUGE-2 | ROUGE-L |  
|:-:         |:-:                |:-:       |:-:      |:-:      |:-:      |  
|8|28467|0.795|29.3|15.0|19.8|  
|16|25016|0.795|29.5|15.1|19.9|  
|32|21712|0.795|29.6|15.1|19.9|  
|64|21924|0.794|29.6|15.1|19.9|  


