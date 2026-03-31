# 🚀 Advait-LM

Welcome to **Advait-LM**! This is a super fun, lightweight, character-level language model built with PyTorch. If you've ever wondered how models like GPT actually work under the hood, you're in the right place! This little repo is designed to show you the absolute basics of how AI learns to read, train on, and generate text, one letter at a time. 😊

## 🛠️ What's Inside? (The Code Blocks)

We kept things nice and tidy! Here’s a quick tour of the files in this project and what they do:

### 1. `config.py`
Think of this as the model's control panel. It holds all the knobs and dials you can twist to change how the model learns:
- **`batch_size` & `block_size`**: Decides how much text the model looks at during one training step, and how far back it can "remember" characters.
- **`n_embd`, `n_head`, `n_layer`**: These decide how "big" and "smart" the brain of our neural network is.
- **`device`**: Automatically figures out if you have a fancy GPU (CUDA/MPS) to use, or if it should just stick to your CPU so things don't crash.

### 2. `model.py`
This is where the magic happens! It contains the actual PyTorch brain (the Neural Network architecture).
- **`CausalSelfAttention`**: The part of the brain that looks at the text and tries to figure out relationships between characters. The "causal" part just means it can't peek into the future to cheat! 
- **`FeedForward`**: A tiny little network that helps process what the attention part just learned.
- **`Block`**: The building blocks of our GPT. We stack a bunch of these together to make the model smarter.
- **`GPTModel`**: The main boss! This wraps everything up, taking in your text, pushing it through the blocks, and guessing what the next character should be. It also has the cool `generate()` function that actually writes text for you.

### 3. `train.py`
This script takes the model to school! 🏫 
- It reads whatever text you put in `data/data.txt` and learns every single unique character you used.
- It chops the data up so it can practice on most of it and test itself on the rest.
- Then, it runs the training loop! You'll see it slowly get better and better at guessing letters.
- Finally, it saves its best brain state into `saved/advait.pt` and its character dictionary into `saved/tokenizer.json` so you can use it later.

### 4. `generate.py`
Time to see what the model learned!
- This script loads up the saved brain (`advait.pt`) and dictionary (`tokenizer.json`).
- You can talk to it by giving it a starting prompt.
- It will autoregressively (a fancy word for "one by one") spit out character after character until you tell it to stop!

---

## 🏃‍♂️ How to Run

Ready to give it a spin? Just follow these steps!

```bash
# 1. activate your venv
source venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. train
python train.py

# 4. generate with empty prompt
python generate.py

# 5. generate with a starting word
python generate.py --prompt "Neo" --temperature 0.8 --top_k 30

# 6. try different settings
python generate.py --prompt "Morpheus" --temperature 0.6 --max_tokens 200
```

---

## 🎨 Train on Your Own Data & Play Around!

The absolute best way to learn how language models work is to mess around with it yourself! Since this model learns *character by character*, it can learn anything! English dialogue, Shakespeare, code, emojis—you name it.

### Step 1: Bring your own data!
1. Go into the `data/` folder and open `data.txt`.
2. Delete what's there and paste in whatever text you want! A book, your favorite movie script, or all your text messages. (Try to use between 1MB and 5MB of text for best results).
3. Run `python train.py` again to take the model back to school on your new data.

### Step 2: Have fun experimenting!
- **Check out the Tokenizer:** After training, peek into `saved/tokenizer.json`. You'll see how the AI numbered every single character it found in your text! It literally just views your text as a giant list of math numbers.
- **Twist the Knobs in `config.py`:**
   - Is the model taking too long to train? Lower the `n_layer` to make the brain smaller.
   - Output still looking like gibberish? Try giving it a bigger brain by increasing `n_layer` to 6 or `n_embd` to 256, or let it train longer by turning up `max_iters`!
- **Play with the "Vibe" (Temperature):** When running `generate.py`, try changing the `--temperature`. 
  - A high temperature (like `1.2`) makes the model super chaotic and "creative," which often means funny typos and nonsense! 🤪
  - A lower temperature (like `0.2`) makes it super strict and confident, but it might get stuck repeating the exact same words over and over. 🤖
