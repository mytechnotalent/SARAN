# SARAN: Shallow Auto-Regressive Attention Network

The dominant paradigms in sequence transduction - Recurrent Neural Networks and deep Transformer architectures - rely on complex, multi-layered structures to achieve performance, often at the cost of interpretability and computational transparency. In this work, we introduce the Shallow Auto-Regressive Attention Network (SARAN), a minimalist architecture that reduces the Transformer decoder to its fundamental components. SARAN is defined by a strictly linear, 15-stage computational graph that maps input embeddings directly to output probabilities via a single, isolated block of masked self-attention. We present a "first principles" derivation of the network's training dynamics, explicitly defining the manual backpropagation algorithm through the attention mechanism without reliance on automatic differentiation engines. By stripping away deep layer stacking and feed-forward networks, SARAN demonstrates that a solitary attention block is sufficient to mechanically derive autoregressive properties, providing a transparent and rigorous baseline for understanding the mechanics of attention-based sequence modeling.


### [dataset](https://www.kaggle.com/datasets/mytechnotalent/mary-had-a-little-lamb)

Author: [Kevin Thomas](mailto:ket189@pitt.edu)

License: MIT


## Constants


```python
PI = 3.141592653589793
RNG_STATE = 12345
```


## Functions


```python
def tokens_to_ids(tokens, word_to_idx):
    unk = word_to_idx.get('<UNK>')
    return [word_to_idx.get(w, unk) for w in tokens]


def random():
    global RNG_STATE
    RNG_STATE = (1103515245 * RNG_STATE + 12345) % (2**31)
    return RNG_STATE / (2**31)


def sqrt(x):
    if x == 0:
        return 0
    guess = x / 2.0
    for _ in range(20):
        guess = (guess + x / guess) / 2.0
    return guess


def exp(x):
    if x < -10:
        return 1.0 / exp(-x)
    result = 1.0
    term = 1.0
    for i in range(1, 50):
        term *= x / i
        result += term
        if abs(term) < 1e-10:
            break
    return result


def log(x):
    if x <= 0:
        return float('-inf')
    guess = 0.0
    for _ in range(50):
        guess = guess + 2 * (x - exp(guess)) / (x + exp(guess))
    return guess


def cos(x):
    x = x % (2 * PI)
    result = 1.0
    term = 1.0
    for i in range(1, 20):
        term *= -x * x / ((2 * i - 1) * (2 * i))
        result += term
    return result


def randn():
    u1 = random()
    u2 = random()
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2)


def parse_json_array(text):
    text = text.strip()
    if not text.startswith('[') or not text.endswith(']'):
        return []
    text = text[1:-1]  # remove [ ]
    result = []
    in_string = False
    current = ""
    escape = False
    for char in text:
        if escape:
            current += char
            escape = False
        elif char == '\\':
            escape = True
        elif char == '"':
            if in_string:
                result.append(current)
                current = ''
            in_string = not in_string
        elif in_string:
            current += char
    return result


def zeros(rows, cols=None):
    if cols is None:
        return [0.0 for _ in range(rows)]
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def matmul(A, B):
    if isinstance(A[0], list) and isinstance(B[0], list):
        # Matrix @ Matrix
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        result = zeros(rows_A, cols_B)
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    elif isinstance(A[0], list):
        # Matrix @ Vector
        result = [0.0 for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B)):
                result[i] += A[i][j] * B[j]
        return result
    else:
        # Vector @ Matrix
        result = [0.0 for _ in range(len(B[0]))]
        for j in range(len(B[0])):
            for i in range(len(A)):
                result[j] += A[i] * B[i][j]
        return result


def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def outer(a, b):
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


def add_matrices(A, B):
    if isinstance(A[0], list):
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    else:
        return [A[i] + B[i] for i in range(len(A))]


def sub_matrices(A, B):
    if isinstance(A[0], list):
        return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    else:
        return [A[i] - B[i] for i in range(len(A))]


def mul_matrices(A, B):
    if isinstance(A[0], list):
        return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    else:
        return [A[i] * B[i] for i in range(len(A))]


def div_scalar(A, scalar):
    if isinstance(A[0], list):
        return [[A[i][j] / scalar for j in range(len(A[0]))] for i in range(len(A))]
    else:
        return [A[i] / scalar for i in range(len(A))]


def mul_scalar(A, scalar):
    if isinstance(A[0], list):
        return [[A[i][j] * scalar for j in range(len(A[0]))] for i in range(len(A))]
    else:
        return [A[i] * scalar for i in range(len(A))]


def copy_matrix(A):
    if isinstance(A[0], list):
        return [[A[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    else:
        return [A[i] for i in range(len(A))]


def triu(n, value, k=1):
    matrix = zeros(n, n)
    for i in range(n):
        for j in range(n):
            if j >= i + k:
                matrix[i][j] = value
    return matrix


def argmax(arr):
    max_val = arr[0]
    max_idx = 0
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_idx


def argsort(arr):
    return sorted(range(len(arr)), key=lambda i: arr[i])


def softmax(z):
    max_z = max(z)
    exp_z = [exp(zi - max_z) for zi in z]
    sum_exp = sum(exp_z)
    return [e / sum_exp for e in exp_z]


def save_weights(model_name, vocab, w_embed, w_pos, w_q, w_k, w_v, w_out, b_out, vocab_size, d_model, context_len):
    with open('w_embed.txt', 'w') as f:
        for row in w_embed:
            f.write(','.join([str(x) for x in row]) + '\n')
    with open('w_pos.txt', 'w') as f:
        for row in w_pos:
            f.write(','.join([str(x) for x in row]) + '\n')
    with open('w_q.txt', 'w') as f:
        for row in w_q:
            f.write(','.join([str(x) for x in row]) + '\n')
    with open('w_k.txt', 'w') as f:
        for row in w_k:
            f.write(','.join([str(x) for x in row]) + '\n')
    with open('w_v.txt', 'w') as f:
        for row in w_v:
            f.write(','.join([str(x) for x in row]) + '\n')
    with open('w_out.txt', 'w') as f:
        for row in w_out:
            f.write(','.join([str(x) for x in row]) + '\n')
    with open('b_out.txt', 'w') as f:
        f.write(','.join([str(x) for x in b_out]) + '\n')
    with open('vocab.txt', 'w') as f:
        f.write(','.join(vocab) + '\n')
    with open('w_attn_out.txt', 'w') as f:
        f.write(f'{vocab_size}\n{d_model}\n{context_len}\n')
    print(f'Trained weights and biases for {model_name} saved to disk!')


def load_weights():
    with open('w_attn_out.txt', 'r') as f:
        vocab_size = int(f.readline().strip())
        d_model = int(f.readline().strip())
        context_len = int(f.readline().strip())
    with open('vocab.txt', 'r') as f:
        vocab = f.readline().strip().split(',')
    with open('w_embed.txt', 'r') as f:
        w_embed = [[float(x) for x in line.strip().split(',')] for line in f.readlines()]
    with open('w_pos.txt', 'r') as f:
        w_pos = [[float(x) for x in line.strip().split(',')] for line in f.readlines()]
    with open('w_q.txt', 'r') as f:
        w_q = [[float(x) for x in line.strip().split(',')] for line in f.readlines()]
    with open('w_k.txt', 'r') as f:
        w_k = [[float(x) for x in line.strip().split(',')] for line in f.readlines()]
    with open('w_v.txt', 'r') as f:
        w_v = [[float(x) for x in line.strip().split(',')] for line in f.readlines()]
    with open('w_out.txt', 'r') as f:
        w_out = [[float(x) for x in line.strip().split(',')] for line in f.readlines()]
    with open('b_out.txt', 'r') as f:
        b_out = [float(x) for x in f.readline().strip().split(',')]
    print('Trained weights and biases for SARAN loaded from disk!')
    return w_embed, w_pos, w_q, w_k, w_v, w_out, b_out, vocab, vocab_size, d_model, context_len

```


## Load Data


```python
# 1. Data Loading and Preparation
with open('corpus.json') as f:
    corpus = parse_json_array(f.read())
words = set()
for line in corpus:
    for word in line.split():
        words.add(word)
vocab = ['<UNK>'] + sorted(words)
vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}
context_len = 4
data = []
for line in corpus:
    tokens = line.split()
    for i in range(len(tokens) - context_len):
        context = [word_to_idx[tokens[j]] for j in range(i, i + context_len)]
        target = word_to_idx[tokens[i + context_len]]
        data.append((context, target))
print(f'Vocabulary: {vocab}')
print(f'Training samples: {len(data)}')
```

**Output:**
```
Vocabulary: ['<UNK>', 'a', 'against', 'and', 'as', 'at', 'children', 'day', 'everywhere', 'fleece', 'followed', 'go', 'had', 'her', 'it', 'its', 'lamb', 'laugh', 'little', 'made', 'mary', 'one', 'play', 'rules', 'school', 'see', 'snow', 'sure', 'that', 'the', 'to', 'was', 'went', 'which', 'white']
Training samples: 26
```


```python
# # 1. Data Loading and Preparation
# with open('tinystories.txt', encoding='utf-8') as f:
#     corpus = [line.strip() for line in f if line.strip()]
# words = set()
# for line in corpus:
#     for word in line.split():
#         words.add(word)
# vocab = ['<UNK>'] + sorted(words)
# vocab_size = len(vocab)
# word_to_idx = {w: i for i, w in enumerate(vocab)}
# idx_to_word = {i: w for i, w in enumerate(vocab)}
# context_len = 4
# data = []
# for line in corpus:
#     tokens = line.split()
#     for i in range(len(tokens) - context_len):
#         context = [word_to_idx[tokens[j]] for j in range(i, i + context_len)]
#         target = word_to_idx[tokens[i + context_len]]
#         data.append((context, target))
# print(f'Vocabulary size: {vocab_size}, Training samples: {len(data)}')
```


## Train/Validation Split


```python
# 2. Train/Validation Split (80/20)
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
val_data = data[split_idx:]
print(f'Train samples: {len(train_data)}, Val samples: {len(val_data)}')
```

**Output:**
```
Train samples: 20, Val samples: 6
```


## Hyperparameters


```python
# 3. Define Model Hyperparameters
model_name = 'SARAN'
n = 0.01  # learning rate
epochs = 300  # cycles through the training dataset
d_model = 32  # embedding dimension
w_embed = [[randn() * 0.1 for _ in range(d_model)] for _ in range(vocab_size)]  # token embeddings (vocab_size x d_model)
w_pos = [[randn() * 0.1 for _ in range(d_model)] for _ in range(context_len)]  # positional embeddings (context_len x d_model)
w_q = [[randn() * 0.1 for _ in range(d_model)] for _ in range(d_model)]  # query projection weights (d_model x d_model)
w_k = [[randn() * 0.1 for _ in range(d_model)] for _ in range(d_model)]  # key projection weights (d_model x d_model)
w_v = [[randn() * 0.1 for _ in range(d_model)] for _ in range(d_model)]  # value projection weights (d_model x d_model)
w_out = [[randn() * 0.1 for _ in range(vocab_size)] for _ in range(d_model)]  # output projection weights (d_model x vocab_size)
b_out = zeros(vocab_size)  # output bias (vocab_size)
```


## Training w/ Validation


```python
sqrt_d_model = sqrt(d_model)

# 4. Training Loop
for epoch in range(epochs):
    total_cost = 0
    correct = 0
    
    # Training Loop
    for context, target in train_data:
        # 1. Input Tokens
        token_ids = context
        
        # 2. Token Embeddings
        token_emb = [[w_embed[tid][j] for j in range(d_model)] for tid in token_ids]
        
        # 3. Positional Encodings
        pos_emb = [[w_pos[i][j] for j in range(d_model)] for i in range(len(token_ids))]
        
        # 4. Embedding Summation
        x = add_matrices(token_emb, pos_emb)
        
        # 5. Query Projection
        q = matmul(x, w_q)
        
        # 6. Key Projection
        k = matmul(x, w_k)
        
        # 7. Value Projection
        v = matmul(x, w_v)
        
        # 8. Attention Score Calculation
        k_T = transpose(k)
        scores = matmul(q, k_T)
        scores = div_scalar(scores, sqrt_d_model)
        
        # 9. Causal Masking
        mask = triu(context_len, -1e9, k=1)
        scores = add_matrices(scores, mask)
        
        # 10. Softmax
        attn = [softmax(row) for row in scores]
        
        # 11. Attention Output Calculation
        attn_out = matmul(attn, v)
        
        # 12. Last Token Selection
        last = attn_out[-1]
        
        # 13. Output Projection
        logits = matmul(last, w_out)
        
        # 14. Bias Addition
        logits = add_matrices(logits, b_out)
        
        # 15. Softmax Activation
        o = softmax(logits)
        
        # Compute Cost
        c = -log(o[target] + 1e-8)
        total_cost += c
        
        # Compute Accuracy
        if argmax(o) == target:
            correct += 1
        
        # 1. Compute Output Gradient (dL/do)
        do = copy_matrix(o)
        do[target] -= 1
        
        # 2. Compute Output Weight Gradients (dw_out, db_out)
        dw_out = outer(last, do)
        db_out = copy_matrix(do)
        
        # 3. Backprop to Last Token (dlast)
        w_out_T = transpose(w_out)
        dlast = matmul(do, w_out_T)
        
        # 4. Backprop to Attention Output (dattn_out)
        dattn_out = zeros(context_len, d_model)
        dattn_out[-1] = dlast
        
        # 5. Compute Value Gradients (dv, dw_v)
        attn_T = transpose(attn)
        dv = matmul(attn_T, dattn_out)
        x_T = transpose(x)
        dw_v = matmul(x_T, dv)
        
        # 6. Backprop to Attention Weights (dattn)
        v_T = transpose(v)
        dattn = matmul(dattn_out, v_T)
        
        # 7. Compute Score Gradients (dscores)
        attn_dattn = mul_matrices(dattn, attn)
        sum_attn_dattn = [sum(row) for row in attn_dattn]
        dattn_adjusted = [[dattn[i][j] - sum_attn_dattn[i] for j in range(len(dattn[0]))] for i in range(len(dattn))]
        dscores = mul_matrices(attn, dattn_adjusted)
        dscores = div_scalar(dscores, sqrt_d_model)
        
        # 8. Compute Query and Key Gradients (dq, dk, dw_q, dw_k)
        dq = matmul(dscores, k)
        dscores_T = transpose(dscores)
        dk = matmul(dscores_T, q)
        dw_q = matmul(x_T, dq)
        dw_k = matmul(x_T, dk)
        
        # 9. Backprop to Embeddings (dx, dw_embed, dw_pos)
        w_q_T = transpose(w_q)
        w_k_T = transpose(w_k)
        w_v_T = transpose(w_v)
        dx1 = matmul(dq, w_q_T)
        dx2 = matmul(dk, w_k_T)
        dx3 = matmul(dv, w_v_T)
        dx = add_matrices(add_matrices(dx1, dx2), dx3)
        dw_embed = zeros(vocab_size, d_model)
        for i, tid in enumerate(token_ids):
            for j in range(d_model):
                dw_embed[tid][j] += dx[i][j]
        dw_pos = copy_matrix(dx)
        
        # 10. Update Parameters
        w_out = sub_matrices(w_out, mul_scalar(dw_out, n))
        b_out = sub_matrices(b_out, mul_scalar(db_out, n))
        w_v = sub_matrices(w_v, mul_scalar(dw_v, n))
        w_q = sub_matrices(w_q, mul_scalar(dw_q, n))
        w_k = sub_matrices(w_k, mul_scalar(dw_k, n))
        w_embed = sub_matrices(w_embed, mul_scalar(dw_embed, n))
        w_pos = sub_matrices(w_pos, mul_scalar(dw_pos, n))
    
    train_accuracy = correct / len(train_data) * 100
    
    # Validation Loop (No Backpropagation)
    val_correct = 0
    val_cost = 0
    for context, target in val_data:
        # 1. Input Tokens
        token_ids = context
        
        # 2. Token Embeddings
        token_emb = [[w_embed[tid][j] for j in range(d_model)] for tid in token_ids]
        
        # 3. Positional Encodings
        pos_emb = [[w_pos[i][j] for j in range(d_model)] for i in range(len(token_ids))]
        
        # 4. Embedding Summation
        x = add_matrices(token_emb, pos_emb)
        
        # 5. Query Projection
        q = matmul(x, w_q)
        
        # 6. Key Projection
        k = matmul(x, w_k)
        
        # 7. Value Projection
        v = matmul(x, w_v)
        
        # 8. Attention Score Calculation
        k_T = transpose(k)
        scores = matmul(q, k_T)
        scores = div_scalar(scores, sqrt_d_model)
        
        # 9. Causal Masking
        mask = triu(context_len, -1e9, k=1)
        scores = add_matrices(scores, mask)
        
        # 10. Softmax
        attn = [softmax(row) for row in scores]
        
        # 11. Attention Output Calculation
        attn_out = matmul(attn, v)
        
        # 12. Last Token Selection
        last = attn_out[-1]
        
        # 13. Output Projection
        logits = matmul(last, w_out)
        
        # 14. Bias Addition
        logits = add_matrices(logits, b_out)
        
        # 15. Softmax Activation
        o = softmax(logits)
        
        # Compute Validation Cost
        c = -log(o[target] + 1e-8)
        val_cost += c
        
        # Compute Validation Accuracy
        if argmax(o) == target:
            val_correct += 1
    
    val_accuracy = val_correct / len(val_data) * 100
    
    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch+1}: Train Cost={total_cost:.4f}, Train Acc={train_accuracy:.2f}%, Val Cost={val_cost:.4f}, Val Acc={val_accuracy:.2f}%')
```

**Output:**
```
Epoch 50: Train Cost=59.0611, Train Acc=15.00%, Val Cost=20.0826, Val Acc=0.00%
Epoch 100: Train Cost=47.2471, Train Acc=15.00%, Val Cost=19.3142, Val Acc=0.00%
Epoch 150: Train Cost=32.2077, Train Acc=45.00%, Val Cost=16.1187, Val Acc=16.67%
Epoch 200: Train Cost=19.9995, Train Acc=70.00%, Val Cost=14.2715, Val Acc=16.67%
Epoch 250: Train Cost=11.2064, Train Acc=95.00%, Val Cost=13.4749, Val Acc=33.33%
Epoch 300: Train Cost=4.1649, Train Acc=100.00%, Val Cost=12.2693, Val Acc=66.67%
```


## Save Weights


```python
save_weights(model_name, vocab, w_embed, w_pos, w_q, w_k, w_v, w_out, b_out, vocab_size, d_model, context_len)
print(f'Model saved to disk!')
```

**Output:**
```
Trained weights and biases for SARAN saved to disk!
Model saved to disk!
```


## Inference


```python
w_embed, w_pos, w_q, w_k, w_v, w_out, b_out, vocab, vocab_size, d_model, context_len = load_weights()
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}
print(f'Loaded vocabulary with {len(vocab)} words')
print(f'Model parameters: vocab_size={vocab_size}, d_model={d_model}, context_len={context_len}')
```

**Output:**
```
Trained weights and biases for SARAN loaded from disk!
Loaded vocabulary with 35 words
Model parameters: vocab_size=35, d_model=32, context_len=4
```


## Test Prediction


```python
# Inference: Predict the next word for 'mary had a little'
test_input = ['mary', 'had', 'a', 'little']

token_ids = tokens_to_ids(test_input, word_to_idx)

print(f'Token IDs: {token_ids}')
print(f'Tokens mapped: {[idx_to_word[tid] for tid in token_ids]}')

# Forward Pass
# 1. Input Tokens
# (token_ids already set above)

# 2. Token Embeddings
token_emb = [[w_embed[tid][j] for j in range(d_model)] for tid in token_ids]

# 3. Positional Encodings
pos_emb = [[w_pos[i][j] for j in range(d_model)] for i in range(len(token_ids))]

# 4. Embedding Summation
x = add_matrices(token_emb, pos_emb)

# 5. Query Projection
q = matmul(x, w_q)

# 6. Key Projection
k = matmul(x, w_k)

# 7. Value Projection
v = matmul(x, w_v)

# 8. Attention Score Calculation
k_T = transpose(k)
scores = matmul(q, k_T)
scores = div_scalar(scores, sqrt_d_model)

# 9. Causal Masking
mask = triu(context_len, -1e9, k=1)
scores = add_matrices(scores, mask)

# 10. Softmax
attn = [softmax(row) for row in scores]

# 11. Attention Output Calculation
attn_out = matmul(attn, v)

# 12. Last Token Selection
last = attn_out[-1]

# 13. Output Projection
logits = matmul(last, w_out)

# 14. Bias Addition
logits = add_matrices(logits, b_out)

# 15. Softmax Activation
o = softmax(logits)

print(f'\nInput: {" ".join(test_input)}')
print(f'Predicted: {idx_to_word[argmax(o)]}')
print(f'\nTop 5 predictions:')
top5_idx = argsort(o)[-5:][::-1]
for idx in top5_idx:
    print(f'  {idx_to_word[idx]}: {o[idx]:.4f}')
print(f'\nSum: {sum(o)}')
```

**Output:**
```
Token IDs: [20, 12, 1, 18]
Tokens mapped: ['mary', 'had', 'a', 'little']

Input: mary had a little
Predicted: lamb

Top 5 predictions:
  lamb: 0.9338
  went: 0.0612
  as: 0.0022
  school: 0.0012
  laugh: 0.0007

Sum: 1.0
```

