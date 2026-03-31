# Advait-LM

This is a tiny language model, I created out of curiosity to see how lanugage models actually work, well this is just a tip of tip of the ice-berg but it's good starting point to understand how things actually work. Well this LM generate gibberish output obviously but still a good starting point just like learning ABCD.

## What's Inside? (The Code Blocks)

### 1. `config.py`

- **`batch_size` & `block_size`**: Decides how much text the model looks at during one training step, and how far back it can "remember" characters.
- **`n_embd`, `n_head`, `n_layer`**: These decide how "big" and "smart" the brain of our neural network is.
- **`device`**: Automatically figures out if you have a fancy GPU (CUDA/MPS) to use, or if it should just stick to your CPU so things don't crash.

### 2. `model.py`

- **`CausalSelfAttention`**: The part of the brain that looks at the text and tries to figure out relationships between characters. The "causal" part just means it can't peek into the future to cheat!
- **`FeedForward`**: A tiny little network that helps process what the attention part just learned.
- **`Block`**: The building blocks of our GPT. We stack a bunch of these together to make the model smarter.
- **`GPTModel`**: The main boss! This wraps everything up, taking in your text, pushing it through the blocks, and guessing what the next character should be. It also has the cool `generate()` function that actually writes text for you.

### 3. `train.py`

- It reads whatever text you put in `data/data.txt` and learns every single unique character you used.
- It chops the data up so it can practice on most of it and test itself on the rest.
- Then, it runs the training loop! You'll see it slowly get better and better at guessing letters.
- Finally, it saves its best brain state into `saved/advait.pt` and its character dictionary into `saved/tokenizer.json` so you can use it later.

### 4. `generate.py`

- This script loads up the saved brain (`advait.pt`) and dictionary (`tokenizer.json`).
- You can talk to it by giving it a starting prompt.
- It will autoregressively (a fancy word for "one by one") spit out character after character until you tell it to stop!

---

## How to Run

Clone the repo and take this for a run but please make sure your system is capable of running this.

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

## Train on Your Own Data & Play Around!

If you are curios enough then give it a try with your own text file, 15k to 20k words should be enough to give it a test drive.

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
  - A high temperature (like `1.2`) makes the model super chaotic and "creative," which often means funny typos and nonsense!
  - A lower temperature (like `0.2`) makes it super strict and confident, but it might get stuck repeating the exact same words over and over.
