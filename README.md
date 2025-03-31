# Time-Series

## LSTM Forecasting
The process of creating sequences before feeding data into an LSTM model is crucial for capturing the temporal relationships in time series data. LSTMs (Long Short-Term Memory networks) are designed to learn dependencies across time steps, making them suitable for tasks where the order and context of data points matter, such as forecasting. Below, I’ll explain the process step-by-step in detail with an example.

---

### **Understanding Why Sequences Are Needed**

LSTMs operate on **sequential data**, meaning they need a fixed-length input sequence for each prediction. This is because:
1. **Temporal Context:** Each time step in the sequence contributes to the understanding of patterns or trends over time (e.g., a sudden spike in CPU usage may be caused by a recent event).
2. **Sliding Window Approach:** By creating sequences, you allow the model to focus on recent trends while generating predictions (rather than treating time points as independent and disconnected).

---

### **Step-by-Step Explanation**

#### Step 1: **Raw Data**
Suppose you have a dataset like this:

| Date       | CPU Used |
|------------|----------|
| 2023-01-01 | 100      |
| 2023-01-02 | 120      |
| 2023-01-03 | 90       |
| 2023-01-04 | 110      |
| 2023-01-05 | 105      |
| 2023-01-06 | 115      |
| 2023-01-07 | 130      |

Here, the "CPU Used" column is the target variable we aim to predict. Each row represents a daily observation of CPU usage.

---

#### Step 2: **Preprocessing**
- **Scaling Data:** LSTMs work best with normalized data. Using MinMaxScaler or StandardScaler, we scale the "CPU Used" column to a range (e.g., [0, 1]) so the numerical values don’t overwhelm the gradients during backpropagation.

Example after scaling:

| Date       | Scaled CPU Used |
|------------|-----------------|
| 2023-01-01 | 0.1             |
| 2023-01-02 | 0.5             |
| 2023-01-03 | 0.2             |
| 2023-01-04 | 0.4             |
| 2023-01-05 | 0.3             |
| 2023-01-06 | 0.6             |
| 2023-01-07 | 0.7             |

---

#### Step 3: **Creating Sequences**
To enable the LSTM to understand temporal relationships, we transform the data into sequences using a **sliding window approach**. For example, with a lookback window of 3 days, the input sequences and corresponding targets look like this:

| Input Sequence           | Target |
|---------------------------|--------|
| [0.1, 0.5, 0.2]           | 0.4    |
| [0.5, 0.2, 0.4]           | 0.3    |
| [0.2, 0.4, 0.3]           | 0.6    |
| [0.4, 0.3, 0.6]           | 0.7    |

Here:
- **Input Sequence:** A group of 3 days of past CPU usage data.
- **Target:** The CPU usage value for the next day.

Why this works:
- The sequences represent the model's input, while the target is the output we want the model to learn to predict based on the pattern in the input.

---

#### Step 4: **Splitting Data**
We split these sequences into training and validation sets. For example:
- Training: First 3 sequences
- Validation: Last sequence

---

#### Step 5: **Feeding into LSTM**
After splitting:
1. **Reshaping:** LSTMs require input in the form `(samples, timesteps, features)`. For our example:
   - Number of samples: 4 (number of sequences)
   - Timesteps: 3 (length of each sequence)
   - Features: 1 (CPU usage is the only feature here)

   Shape of `X_train` becomes `(3, 3, 1)`.

2. **LSTM Training:** The reshaped sequences are fed into the LSTM model. The architecture might look like:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, input_shape=(3, 1)))  # 50 LSTM units, input shape = (timesteps, features)
model.add(Dense(1))  # Output layer predicts the target
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=1)
```

---

#### Step 6: **Forecasting**
After training, the model can predict CPU usage for the next day. For multi-step forecasts (e.g., 30 days), we use an **iterative approach**:
1. Start with the last sequence from the data.
2. Predict the next day’s CPU usage.
3. Update the sequence by dropping the oldest value and appending the prediction.
4. Repeat for the desired number of days.

---

### **Example Output**
After running predictions for 30 days, you might get something like this:

| Date       | Forecasted CPU Used |
|------------|---------------------|
| 2023-01-08 | 125                 |
| 2023-01-09 | 130                 |
| 2023-01-10 | 128                 |
| 2023-01-11 | 135                 |
| 2023-01-12 | 140                 |
| ...        | ...                 |

### **Key Takeaways**
1. **Sequences Help Capture Temporal Dependencies:** LSTMs rely on sequences to learn trends and patterns.
2. **Sliding Window:** Allows the model to focus on recent data when predicting future values.
3. **Time Steps Matter:** Longer lookback windows may capture more context but risk adding noise; shorter windows may lose critical information.
