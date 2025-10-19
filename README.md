# Urdu Transformer Chatbot

A simple Streamlit web application for an Urdu language chatbot using a custom-trained Transformer model.

## Features

 **Urdu Input/Output** - Full support for Urdu text with right-to-left (RTL) rendering
**Multiple Decoding Strategies**:
   - Greedy Decoding (fastest, deterministic)
   - Beam Search (balanced, explores top candidates)
   - Sampling (creative, with temperature, top-k, and repetition penalties)
**Conversation History** - View all your chat interactions
**Adjustable Parameters** - Customize max length, beam width, temperature, etc.
**Clean UI** - Simple and easy-to-use interface

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the App

Simply run:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

1. **Select Decoding Strategy** - Choose from Greedy, Beam Search, or Sampling in the sidebar
2. **Adjust Parameters** - Modify settings like max length, temperature, beam width based on your strategy
3. **Enter Urdu Text** - Type your message in the input box (Urdu text is supported)
4. **Generate Response** - Click the "Generate" button to get a response
5. **View History** - All conversations are saved in the session and displayed above

## Model Files Required

Make sure you have the following files in the `Model/` directory:
- `urdu_transformer.pt` - Trained model weights
- `urdu_tokenizer.model` - SentencePiece tokenizer
- `model_config.json` - Model configuration

## Decoding Strategies Explained

### Greedy Decoding
- Always selects the token with highest probability
- Fastest and most deterministic
- Good for consistent outputs

### Beam Search
- Maintains multiple candidate sequences
- Explores top-k paths simultaneously
- Balance between speed and quality

### Sampling (Default)
- Samples from probability distribution
- Most creative and diverse outputs
- Configurable with temperature, top-k, and penalties

## Tips for Best Results

- Use **Sampling** for more natural and diverse responses
- Increase **Temperature** (0.9-1.5) for more creative outputs
- Lower **Temperature** (0.5-0.8) for more focused responses
- Adjust **Repetition Penalty** (1.2-1.5) to reduce repetitive text
- Use **Beam Search** (beam width 3-5) for more balanced outputs

Built with ❤️ using PyTorch and Streamlit
