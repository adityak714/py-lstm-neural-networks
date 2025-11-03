# SEE running of sections of code in cells as per README.md
# All code modules aggregated below in one collective .py file
################################################

import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# !pip install pytorch_lightning
import pytorch_lightning as L
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader

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
    
## This is an example trial run of the LSTM, in action. Here, we simulate with a very small dataset, such as the stock market
# prices of a company, for 4 days.

model = LSTMByHand()
data1 = torch.tensor([0., 0.5, 0.25, 1.])
data2 = torch.tensor([1., 0.5, 0.25, 1.])

print("Company A: Expected: 0, Actual: ", model(data1).detach())
print("Company B: Expected: 1, Actual: ", model(data2).detach())

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

print(model(torch.tensor([0.,0.5,0.25,1.]))) # Should be a value very close to 0
print(model(torch.tensor([1.,0.5,0.25,1.]))) # Should be a value very close to 1

################################################

def loadDataSet(dirname: str, filename: str):
    return pd.read_csv(dirname+filename)

prices_df = loadDataSet('./archive/','prices-split-adjusted.csv')
print(prices_df.columns)

#fundamentals.columns
#fundamentals.corr()

wltw_prices = prices_df['close'].loc[prices_df['symbol'] == 'AAPL']
wltw_prices = stats.zscore(wltw_prices)

# print(wltw_prices) # of type series

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

model_2 = LSTMByHand()

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
trainer_.fit(model_2, train_dataloaders=dataloader)

################################################

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
    
model_ = LSTM()
wltw_X = (torch.tensor(stats.zscore(wltw_new[:, :5], axis=None), requires_grad=True).float())
wltw_y = (torch.tensor(stats.zscore(wltw_new[:, 5], axis=None), requires_grad=True).float())

print(wltw_X)
print(wltw_y)

dataset_ = TensorDataset(wltw_X, wltw_y)
dataloader_ = DataLoader(dataset_)

trainer2 = L.Trainer(max_epochs=250, log_every_n_steps=5)
trainer2.fit(model_, dataloader_)

model_last_checkpoint2 = trainer2.checkpoint_callback.best_model_path

## Uncomment this --> And run if model is to be further trained FROM WHERE IT LEFT OF
trainer2 = L.Trainer(max_epochs=500, log_every_n_steps=5)
# run trainer on LSTM "model", and use the training data, supplied by the loader "dataLoader"

wltw_X = torch.tensor(wltw_new[:, :5], requires_grad=True).float()
wltw_y = torch.tensor(wltw_new[:, 5], requires_grad=True).float()

new_dataloader = DataLoader(TensorDataset(wltw_X, wltw_y))

trainer2.fit(model_, dataloader_, ckpt_path=model_last_checkpoint2)

print("Predicted: ")

pred_ = np.zeros((wltw_new.shape[0]))
real_ = np.zeros((wltw_new.shape[0]))

for i in range(wltw_new.shape[0]):
  pred_[i] = model_(torch.tensor((wltw_new[i, :5])).float()).detach().numpy().item()
  real_[i] = (wltw_new[i, 5])

accuracy = np.sum(pred_/real_)/wltw_new.shape[0]
print(accuracy)