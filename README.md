# News Headline Classifier

A Streamlit web application that classifies news headlines into different categories using a fine-tuned BERT model. The app leverages Hugging Face's Transformers library with a BERT base uncased model fine-tuned on the AG News dataset.

## Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: PyTorch
- **NLP Model**: BERT base uncased (fine-tuned)
- **Tokenizer**: Hugging Face Transformers
- **Environment**: Python 3.8+
- **Dataset**: AG News (2000 training samples, 200 test samples)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**:
   \`\`\`bash
   git clone https://github.com/RAFI0-ABDUL/News-Headlines-Classifier.git
   cd News-Headlines-Classifier
   \`\`\`

2. **Create a virtual environment**:
   \`\`\`bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   \`\`\`

3. **Install dependencies**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Download the model**:
   - Place your fine-tuned BERT model in the \`news_classifier_model/\` directory
   - The model should include:
     - \`config.json\`
     - \`model.safetensors\` 
     - \`tokenizer_config.json\`
     - \`special_tokens_map.json\`
     - \`vocab.txt\`

## Usage

1. **Run the application**:
   \`\`\`bash
   streamlit run news_predictor.py
   \`\`\`

2. **Open your browser** and navigate to local link

3. **Enter a news headline** in the text box or try the sample headlines provided

4. **Click \"Predict Category\"** to see the classification result

##  Contributing

1. Fork the repository
2. Create a feature branch (\`git checkout -b feature/amazing-feature\`)
3. Commit your changes (\`git commit -m 'Add amazing feature'\`)
4. Push to the branch (\`git push origin feature/amazing-feature\`)
5. Open a Pull Request

## Acknowledgments

- Hugging Face for the Transformers library
- Google Research for the BERT model
- AG News dataset providers
- Streamlit for the amazing deployment framework

⭐ Star this repo if you found it helpful!
