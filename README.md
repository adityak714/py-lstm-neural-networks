# **LSTM Neural Networks**
A self-learning project where using PyTorch, I train a Long Short Term Memory neural network, on the NY Stock Exchange dataset split on 5 day-batches, to predict the prices of AAPL.

Stochastic Gradient Descent:
> From how we learned the conventional gradient descent algorithm (also known as **batch gradient descent**), this is an algorithm that drastically reduces the amount of computations done to find the minimum of the cost function $J(\theta)$.

> Normal GD finds the sum of squared residuals for all $m$ data points. This would not scale well for, eg. 10000000 data points, and will be slow.Stochastic Gradient Descent (SDG) computes the squared residual distance ($J(\theta)=\frac{(h_w(x)-y)^2}{m}$) for only one datapoint (rather than $\Sigma ...$).

Then, **cost is calculated, derivative is calculated, and is attempted to be minimized**. Repeat this process for another random datapoint, and another random datapoint, for a set number of iterations.

Plus-sides:
- **Much faster**, and **efficient** than doing it for some iterations, and always adding $m$ terms in each iteration.

**`Adam()`** - Pytorch / Tensorflow
- An algorithm very similar to Stochastic Gradient Descent, but a more efficient version.

### Long short-term memory (LSTM) Networks
- A neural network where flow of information occurs in a neural network, and using weighted parameters, knowledge from previous inputs is remembered in a state "hidden state" (**long term memory**). This is useful for influencing the output of the prediction.


```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
!pip install pytorch_lightning

import pytorch_lightning as L
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader
```

    Collecting pytorch_lightning
      Downloading pytorch_lightning-2.1.0-py3-none-any.whl (774 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m774.6/774.6 kB[0m [31m10.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy>=1.17.2 in /usr/local/lib/python3.10/dist-packages (from pytorch_lightning) (1.23.5)
    Requirement already satisfied: torch>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from pytorch_lightning) (2.1.0+cu118)
    Requirement already satisfied: tqdm>=4.57.0 in /usr/local/lib/python3.10/dist-packages (from pytorch_lightning) (4.66.1)
    Requirement already satisfied: PyYAML>=5.4 in /usr/local/lib/python3.10/dist-packages (from pytorch_lightning) (6.0.1)
    Requirement already satisfied: fsspec[http]>2021.06.0 in /usr/local/lib/python3.10/dist-packages (from pytorch_lightning) (2023.6.0)
    Collecting torchmetrics>=0.7.0 (from pytorch_lightning)
      Downloading torchmetrics-1.2.0-py3-none-any.whl (805 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m805.2/805.2 kB[0m [31m21.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from pytorch_lightning) (23.2)
    Requirement already satisfied: typing-extensions>=4.0.0 in /usr/local/lib/python3.10/dist-packages (from pytorch_lightning) (4.5.0)
    Collecting lightning-utilities>=0.8.0 (from pytorch_lightning)
      Downloading lightning_utilities-0.9.0-py3-none-any.whl (23 kB)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from fsspec[http]>2021.06.0->pytorch_lightning) (2.31.0)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /usr/local/lib/python3.10/dist-packages (from fsspec[http]>2021.06.0->pytorch_lightning) (3.8.6)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.12.0->pytorch_lightning) (3.12.4)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.12.0->pytorch_lightning) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.12.0->pytorch_lightning) (3.2)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.12.0->pytorch_lightning) (3.1.2)
    Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.12.0->pytorch_lightning) (2.1.0)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>2021.06.0->pytorch_lightning) (23.1.0)
    Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>2021.06.0->pytorch_lightning) (3.3.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>2021.06.0->pytorch_lightning) (6.0.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>2021.06.0->pytorch_lightning) (4.0.3)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>2021.06.0->pytorch_lightning) (1.9.2)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>2021.06.0->pytorch_lightning) (1.4.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>2021.06.0->pytorch_lightning) (1.3.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.12.0->pytorch_lightning) (2.1.3)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->fsspec[http]>2021.06.0->pytorch_lightning) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->fsspec[http]>2021.06.0->pytorch_lightning) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->fsspec[http]>2021.06.0->pytorch_lightning) (2023.7.22)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.12.0->pytorch_lightning) (1.3.0)
    Installing collected packages: lightning-utilities, torchmetrics, pytorch_lightning
    Successfully installed lightning-utilities-0.9.0 pytorch_lightning-2.1.0 torchmetrics-1.2.0


## **Making By-Hand LSTM Model: Small-Scale Stock Predictor**

Dataset from NYSE values of S&P500 companies: Kaggle
https://www.kaggle.com/datasets/dgawlik/nyse/data

### By-hand: Initializing model class

```python
class LSTMByHand(L.LightningModule):
    def __init__(self) -> None:
        ## Inherit all attributes from the LightningModule class
        super().__init__()

        mean = torch.tensor(0.0) # Tensor is almost similar to a numpy-array
        std = torch.tensor(1.0)

        ## Forget Gate
        self.w11 = nn.Parameter(torch.normal(mean, std), requires_grad=True) ## Randomly initialise a number from a Gaussian Normal Distribution
        self.w12 = nn.Parameter(torch.normal(mean, std), requires_grad=True)## mu = 0, sigma = 1 --> will return random numbers, most probably closer to 0 and less close to 1
        # Bias for the first gate
        self.b1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        ## Input Gate
        self.w21 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.w22 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        # Bias for the first gate
        self.b2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        ## Candidate Long-Term Memory
        self.w31 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.w32 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        # Bias for the first gate
        self.b3 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        ## Output Gate
        self.w41 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.w42 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        # Bias for the first gate
        self.b4 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def lstm_unit(self, input_value, initial_short_memory, initial_long_memory):
        ## Forget gate
        # - input value x_t, and short-term memory is previous output, therefore h_t-1
        percent_longterm = torch.sigmoid((initial_short_memory * self.w11) + (input_value * self.w12) + self.b1)

        ## Input gate --> Creates a new potential long-term memory, and what percentage of THAT to remember.
        # The line below is basically the forget gate for this part of the LSTM.
        percent_remember_potential = torch.sigmoid((initial_short_memory * self.w21) + (input_value * self.w22) + self.b2)
        # Below is not a % value, but the actual value of the new potential memory to be added [-1,1], and reduced down to how much ever percent_remember_potential says.
        potential_memory = torch.tanh((initial_short_memory * self.w31) + (input_value * self.w32) + self.b3)

        ## Update long-term memory.
        # Add how much ever part of long_memory to keep, with how much ever part of new potential memory to add, and assign that as the updated value.
        updated_longterm = (percent_longterm * initial_long_memory) + (percent_remember_potential * potential_memory)

        # NOTE: This is the core of LSTM, with every new input gone through, the long term memory is increased/scaled down as per influence, and from past information.

        ## We use the updated long-term, and create a new short term memory.
        # This is again, controlled by its OWN forget gate, that decides how much info to keep, and how much to forget.
        percent_output = torch.sigmoid((initial_short_memory * self.w41) + (input_value * self.w42) + self.b4)
        updated_shortterm = torch.tanh(updated_longterm) * percent_output

        ## We return the changed longterm memory and shortterm memories, which could be passed down further layers, or returned as output
        # (updated_shortterm would be the output).
        return([updated_longterm, updated_shortterm])

    def forward(self, input: list):
        ## Initialise long + short term memory as 0
        long_memory = 0.0
        short_memory = 0.0

        ## Assuming input is in sequential order, and in the form of a list, we iterate from the first index to the length of the array,
        # of m items.
        for i in range(len(input)):
            # We call our previously created lstm_unit function, that updates the long+short term memory based on how it learns from
            # the data, and gives an output.
            long_memory, short_memory = self.lstm_unit(input[i], short_memory, long_memory)

        # REMEMBER: The short-term memory is our last recorded output, and is what is outputted during this activity.
        return short_memory

    def configure_optimizers(self):
        # The algorithm of PyTorch that tries to optimize the w parameters while training the LSTM.
        return Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        # Here we calculate things like the loss, to get the training progress of the LSTM.
        input_i, label_i = batch
        output_i = self.forward(input_i[0])

        ## Loss calculation
        loss = (output_i - label_i)**2

        self.log("train_loss", loss)

        if (label_i == 0):
            self.log("out_0_companyA", output_i)
        else:
            self.log("out_0_companyB", output_i)

        return loss
```


```python
## This is an example trial run of the LSTM, in action. Here, we simulate with a very small dataset, such as the stock market
# prices of a company, for 4 days.

model = LSTMByHand()
data1 = torch.tensor([0., 0.5, 0.25, 1.])
data2 = torch.tensor([1., 0.5, 0.25, 1.])

print("Company A: Expected: 0, Actual: ", model(data1).detach())
print("Company B: Expected: 1, Actual: ", model(data2).detach())
```

    Company A: Expected: 0, Actual:  tensor(0.4607)
    Company B: Expected: 1, Actual:  tensor(0.5484)


### Training of By-Hand LSTM

As can be seen from running the cell above, the model is quite inaccurate, as it does not return a value close to the observed (true) value. So we train it using the following methods...

Troubleshooting:
> If an error comes up, of the kind "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.", then add `.detach().numpy()` in all usages of Tensors, or go to `_tensors.py` in the `torch` library's source code, and modify the `__array__` function to return `self.detach().numpy()` instead of `self.numpy()`.


```python
X = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]], requires_grad=True)
y = torch.tensor([0., 1.], requires_grad=True)

# Bring X and y together as one matrix, done as a "TensorDataset", which is passed in, like any other nd-numpy array.
dataset = TensorDataset(X, y)

# Make the dataset be entered to a dataloader, for quick training, and sending data in parts (batches)
# while training etc.
dataloader = DataLoader(dataset)

## Training Area - doing backpropagation for 2000 epochs
# Initialize trainer instance (only done once at start, and then comment out the line)
trainer = L.Trainer(max_epochs=100)

# Allow model to retrain on an improved version, over and over, instead of resetting progress everytime.
model_last_checkpoint = trainer.checkpoint_callback.best_model_path

## Uncomment this --> And run if model is to be further trained FROM WHERE IT LEFT OF
# trainer = L.Trainer(max_epochs=6000)
# run trainer on LSTM "model", and use the training data, supplied by the loader "dataLoader"
trainer.fit(model, train_dataloaders=dataloader)
```

    INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
    INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
    INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:pytorch_lightning.callbacks.model_summary:
      | Name         | Type | Params
    --------------------------------------
      | other params | n/a  | 12    
    --------------------------------------
    12        Trainable params
    0         Non-trainable params
    12        Total params
    0.000     Total estimated model params size (MB)
    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:441: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=1` in the `DataLoader` to improve performance.
    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/loops/fit_loop.py:293: The number of training batches (2) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=100` reached.



```python
print(model(torch.tensor([0.,0.5,0.25,1.]))) # Should be a value very close to 0
print(model(torch.tensor([1.,0.5,0.25,1.]))) # Should be a value very close to 1
```

    tensor(0.3852, grad_fn=<MulBackward0>)
    tensor(0.5548, grad_fn=<MulBackward0>)


### Visualize the changes in the loss, and how the learning is going

We can use `TensorBoard` to be able to visualise, in the form of graphs, how the neural network is improving. There should be a directory with the name of `lightning_logs/` where 1) the above results that the model predicts, 2) the losses after each epoch etc. are stored.

Run in a terminal:
```
tensorboard --logdir=lightning_logs/
```

## **Extended exercise: Stock Predictor model using LSTM**
Here we will use the NYSE dataset, and put our LSTM to good use, by making it learn the closing prices of a particular company's stocks in a span of **1 week**, and use it to predict the price on the **8th day**. We will make m training examples, where m is the total number of collected time points (days, in this case it is data collected for about 1 year of prices), divided by 7 (as we group them into batches, to train the model with). The 8th column will represent the label, which is made to be the day that the closing price was observed to be (y).

### Data Observation
Getting the NYSE dataset, and analysing the amount of entries made for each company. From collecting the frequencies, it was found that AAPL (Apple Inc.) was one of the popular ones, and had a lot of entries, which could be used for training the model well.


```python
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import train_test_split
```


```python
def loadDataSet(dirname: str, filename: str):
    return pd.read_csv(dirname+filename)
```


```python
!wget https://github.com/adityak714/rust_multithr-hotel-reserv-system/raw/main/archive.zip
!unzip archive.zip
```

    --2023-10-27 13:39:35--  https://github.com/adityak714/rust_multithr-hotel-reserv-system/raw/main/archive.zip
    Resolving github.com (github.com)... 20.205.243.166
    Connecting to github.com (github.com)|20.205.243.166|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://raw.githubusercontent.com/adityak714/rust_multithr-hotel-reserv-system/main/archive.zip [following]
    --2023-10-27 13:39:35--  https://raw.githubusercontent.com/adityak714/rust_multithr-hotel-reserv-system/main/archive.zip
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 16177814 (15M) [application/zip]
    Saving to: ‘archive.zip.4’
    
    archive.zip.4       100%[===================>]  15.43M  --.-KB/s    in 0.06s   
    
    2023-10-27 13:39:35 (271 MB/s) - ‘archive.zip.4’ saved [16177814/16177814]
    
    Archive:  archive.zip
    replace archive/prices-split-adjusted.csv? [y]es, [n]o, [A]ll, [N]one, [r]ename: N



```python
prices_df = loadDataSet('./archive/','prices-split-adjusted.csv')

prices_df.columns

#fundamentals.columns

#fundamentals.corr()
```




    Index(['date', 'symbol', 'open', 'close', 'low', 'high', 'volume'], dtype='object')




```python
wltw_prices = prices_df['close'].loc[prices_df['symbol'] == 'AAPL']
wltw_prices = stats.zscore(wltw_prices)

wltw_prices # of type series
```




    254      -1.724049
    721      -1.722183
    1189     -1.739383
    1657     -1.741350
    2125     -1.734288
                ...   
    848767    1.310512
    849267    1.336640
    849767    1.318986
    850267    1.317927
    850767    1.285797
    Name: close, Length: 1762, dtype: float64



### Data Batching

Split day-wise close figures, to **batches of 5-day data**. Then the 6th index, will be considered as our output (y), and shall be used to check with what the model predicts.


```python
wltw_prices_list = wltw_prices.values.tolist()

wltw_new = np.zeros((220,6))
j = 0

for i in range(0, len(wltw_prices_list), 8):
    if len(wltw_prices_list[i:i+5]) < 5:
        break
    wltw_new[j,:5] = wltw_prices_list[i:i+5]
    wltw_new[j,5] = wltw_prices_list[i+5]
    j = j + 1

print(wltw_new)
```

    [[-1.72404938 -1.72218316 -1.73938285 -1.74134995 -1.73428846 -1.74372057]
     [-1.74715044 -1.76480409 -1.71885418 -1.73554946 -1.75401016 -1.80606319]
     [-1.75496846 -1.79829561 -1.83476297 -1.82129578 -1.81559618 -1.79859825]
     ...
     [ 1.12373679  1.14350881  1.13574117  1.13185733  1.09866839  1.06230191]
     [ 1.07854323  1.11667519  1.15516029  1.21977244  1.1968229   1.26355367]
     [ 1.29109333  1.31474915  1.32569434  1.32957818  1.30239166  1.3105122 ]]


### By-hand - Predicting stock prices of AAPL using self-made LSTM network


```python
model_ = LSTMByHand()

wltw_X = torch.tensor(wltw_new[:, :5], requires_grad=True)
wltw_y = torch.tensor(wltw_new[:, 5], requires_grad=True)

dataset_ = TensorDataset(wltw_X, wltw_y)
dataloader_ = DataLoader(dataset_)

## Training Area - doing backpropagation for 2000 epochs
# Initialize trainer instance (only done once at start, and then comment out the line)
trainer_ = L.Trainer(max_epochs=20)

# Allow model to retrain on an improved version, over and over, instead of resetting progress everytime.
model_last_checkpoint = trainer_.checkpoint_callback.best_model_path

## Uncomment this --> And run if model is to be further trained FROM WHERE IT LEFT OF
# trainer_ = L.Trainer(max_epochs=300)
# run trainer on LSTM "model", and use the training data, supplied by the loader "dataLoader"
trainer_.fit(model_, train_dataloaders=dataloader)
```

    INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
    INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
    INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:pytorch_lightning.callbacks.model_summary:
      | Name         | Type | Params
    --------------------------------------
      | other params | n/a  | 12    
    --------------------------------------
    12        Trainable params
    0         Non-trainable params
    12        Total params
    0.000     Total estimated model params size (MB)
    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:441: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=1` in the `DataLoader` to improve performance.
    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/loops/fit_loop.py:293: The number of training batches (2) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.



    Training: |          | 0/? [00:00<?, ?it/s]


    INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=20` reached.


### Pytorch - Predicting stock prices of AAPL using `torch.nn.LSTM()`


```python
class LSTM(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=1)

    def forward(self, input):
        input_trans = input.view(len(input), 1)
        lstm_out, temp = self.lstm(input_trans)

        prediction = lstm_out[-1]
        return prediction

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.0001)

    def training_step(self, batch, batch_idx):
        # Here we calculate things like the loss, to get the training progress of the LSTM.
        input_i, label_i = batch
        output_i = self.forward(input_i[0])

        ## Loss calculation
        loss_func = nn.MSELoss()
        loss = loss_func(input_i, output_i)
        loss.backward(retain_graph=True)

        self.log("train_loss", loss)
        self.log("pred_closing", output_i)

        loss_ = (output_i-label_i)**2

        return loss_
```


```python
model_ = LSTM()
wltw_X = (torch.tensor(stats.zscore(wltw_new[:, :5], axis=None), requires_grad=True).float())
wltw_y = (torch.tensor(stats.zscore(wltw_new[:, 5], axis=None), requires_grad=True).float())

print(wltw_X)
print(wltw_y)

dataset_ = TensorDataset(wltw_X, wltw_y)
dataloader_ = DataLoader(dataset_)

trainer2 = L.Trainer(max_epochs=250, log_every_n_steps=5)
trainer2.fit(model_, dataloader_)
```

    INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
    INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
    INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:pytorch_lightning.callbacks.model_summary:
      | Name | Type | Params
    ------------------------------
    0 | lstm | LSTM | 16    
    ------------------------------
    16        Trainable params
    0         Non-trainable params
    16        Total params
    0.000     Total estimated model params size (MB)


    tensor([[-1.7240, -1.7222, -1.7394, -1.7413, -1.7343],
            [-1.7472, -1.7648, -1.7189, -1.7355, -1.7540],
            [-1.7550, -1.7983, -1.8348, -1.8213, -1.8156],
            ...,
            [ 1.1237,  1.1435,  1.1357,  1.1319,  1.0987],
            [ 1.0785,  1.1167,  1.1552,  1.2198,  1.1968],
            [ 1.2911,  1.3147,  1.3257,  1.3296,  1.3024]],
           grad_fn=<ToCopyBackward0>)
    tensor([-1.7437, -1.8061, -1.7986, -1.7776, -1.7714, -1.6694, -1.6699, -1.6133,
            -1.5643, -1.4442, -1.5614, -1.5306, -1.5078, -1.5399, -1.4222, -1.5580,
            -1.5352, -1.4715, -1.4916, -1.5270, -1.5803, -1.4749, -1.3521, -1.3981,
            -1.2787, -1.2497, -1.2039, -1.2878, -1.2341, -1.1866, -1.1634, -1.1325,
            -1.0458, -1.0724, -1.0119, -1.0353, -0.9899, -1.0612, -1.0304, -1.0985,
            -1.1297, -1.0375, -1.0521, -1.1168, -1.0712, -1.1554, -1.1287, -0.9892,
            -0.8520, -0.8023, -0.9186, -0.9191, -0.9168, -0.8214, -0.7897, -0.9383,
            -0.7929, -0.7618, -0.8605, -0.9044, -0.8212, -0.8922, -0.7726, -0.6687,
            -0.6477, -0.5079, -0.2338, -0.1516, -0.0698,  0.2529,  0.2206,  0.3377,
             0.0225,  0.0476, -0.0491,  0.0830,  0.1236,  0.1509,  0.1851,  0.2479,
             0.0966,  0.3366,  0.4061,  0.6001,  0.5393,  0.7207,  0.5321,  0.3727,
             0.3077,  0.0110,  0.0500,  0.1486, -0.0848, -0.1798, -0.1610, -0.2680,
            -0.4993, -0.3828, -0.5297, -0.6565, -0.5049, -0.5708, -0.6059, -0.7926,
            -0.5563, -0.5647, -0.5582, -0.5919, -0.6257, -0.8034, -0.6483, -0.6902,
            -0.4705, -0.2891, -0.2666, -0.2905, -0.4597, -0.3988, -0.3338, -0.1813,
            -0.1805, -0.1775, -0.1618,  0.0212, -0.0255,  0.0262, -0.1013, -0.0492,
            -0.2180, -0.0496, -0.1492, -0.0969, -0.0839, -0.0859, -0.1910,  0.1930,
             0.1623,  0.2466,  0.3674,  0.4550,  0.3840,  0.5849,  0.4833,  0.6700,
             0.5415,  0.7474,  0.8437,  0.7858,  0.7890,  0.7138,  0.5952,  0.9652,
             1.0454,  1.2452,  1.2438,  1.0708,  1.1513,  1.0009,  1.0351,  1.3331,
             1.6057,  1.8631,  1.6664,  1.7324,  1.6583,  1.6841,  1.7377,  1.7406,
             1.7494,  1.8585,  1.7088,  1.7116,  1.6251,  1.6332,  1.6156,  1.2441,
             1.2907,  1.0694,  1.1619,  1.2021,  1.0909,  1.1368,  1.2745,  1.5241,
             1.1629,  1.3638,  1.3709,  0.9401,  0.9130,  0.6351,  0.7269,  0.5161,
             0.5952,  0.7459,  0.8070,  0.9437,  1.0736,  1.0750,  0.6503,  0.4727,
             0.5225,  0.6729,  0.6333,  0.5895,  0.5694,  0.7212,  0.8805,  1.0383,
             1.0577,  0.9426,  1.0079,  1.1760,  1.1880,  1.3469,  1.2385,  1.1177,
             1.0824,  1.0623,  1.2636,  1.3105], grad_fn=<ToCopyBackward0>)



    Training: |          | 0/? [00:00<?, ?it/s]


    /usr/local/lib/python3.10/dist-packages/torch/nn/modules/loss.py:535: UserWarning: Using a target size (torch.Size([1])) that is different to the input size (torch.Size([1, 5])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
      return F.mse_loss(input, target, reduction=self.reduction)
    INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=250` reached.


### Retraining from where the model last left off
Only run if the above is true.


```python
model_last_checkpoint2 = trainer2.checkpoint_callback.best_model_path

## Uncomment this --> And run if model is to be further trained FROM WHERE IT LEFT OF
trainer2 = L.Trainer(max_epochs=500, log_every_n_steps=5)
# run trainer on LSTM "model", and use the training data, supplied by the loader "dataLoader"

wltw_X = torch.tensor(wltw_new[:, :5], requires_grad=True).float()
wltw_y = torch.tensor(wltw_new[:, 5], requires_grad=True).float()

new_dataloader = DataLoader(TensorDataset(wltw_X, wltw_y))

trainer2.fit(model_, dataloader_, ckpt_path=model_last_checkpoint2)
```

    INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
    INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs
    INFO:pytorch_lightning.utilities.rank_zero:Restoring states from the checkpoint path at /content/lightning_logs/version_10/checkpoints/epoch=249-step=55000.ckpt
    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/callbacks/model_checkpoint.py:345: The dirpath has changed from '/content/lightning_logs/version_10/checkpoints' to '/content/lightning_logs/version_11/checkpoints', therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded.
    INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:pytorch_lightning.callbacks.model_summary:
      | Name | Type | Params
    ------------------------------
    0 | lstm | LSTM | 16    
    ------------------------------
    16        Trainable params
    0         Non-trainable params
    16        Total params
    0.000     Total estimated model params size (MB)
    INFO:pytorch_lightning.utilities.rank_zero:Restored all states from the checkpoint at /content/lightning_logs/version_10/checkpoints/epoch=249-step=55000.ckpt
    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:441: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=1` in the `DataLoader` to improve performance.



    Training: |          | 0/? [00:00<?, ?it/s]


    /usr/local/lib/python3.10/dist-packages/torch/nn/modules/loss.py:535: UserWarning: Using a target size (torch.Size([1])) that is different to the input size (torch.Size([1, 5])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
      return F.mse_loss(input, target, reduction=self.reduction)
    INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=500` reached.


### Accuracy of Model


```python
print("Predicted: ")

pred_ = np.zeros((wltw_new.shape[0]))
real_ = np.zeros((wltw_new.shape[0]))

for i in range(wltw_new.shape[0]):
  pred_[i] = model_(torch.tensor((wltw_new[i, :5])).float()).detach().numpy().item()
  real_[i] = (wltw_new[i, 5])

accuracy = np.sum(pred_/real_)/wltw_new.shape[0]
print(accuracy)
```

    Predicted: 
    0.9722046504895187

