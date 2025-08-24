import streamlit as st
from transformers import pipeline
import time

# Page configuration
st.set_page_config(
    page_title="News Headline Classifier",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #ffffff;
        border: 2px solid #2E86C1;
        border-radius: 8px;
        color: black;
    }
    .stButton>button {
        background: linear-gradient(to right, #2E86C1, #3498DB);
        color: white;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: bold;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #2874A6, #2E86C1);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #e6f3ff 0%, #d6eaf8 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #2E86C1;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .title {
        color: #2C3E50;
        text-align: center;
        font-size: 2.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .header {
        color: #2E86C1;
        text-align: center;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }
    .social-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 16px;
        margin: 0 8px;
        border-radius: 25px;
        color: white;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .social-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 2rem;
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 2.5rem;
        border-radius: 0 0 12px 12px;
    }
    .navbar-title {
        display: flex;
        align-items: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .navbar-logo {
        margin-right: 12px;
        font-size: 2rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .social-container {
        display: flex;
        align-items: center;
    }
    .github-btn {
        background: linear-gradient(135deg, #24292e 0%, #343434 100%);
    }
    .github-btn:hover {
        background: linear-gradient(135deg, #000000 0%, #24292e 100%);
    }
    .linkedin-btn {
        background: linear-gradient(135deg, #0077B5 0%, #005983 100%);
    }
    .linkedin-btn:hover {
        background: linear-gradient(135deg, #006097 0%, #00476e 100%);
    }
    .sample-btn {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        color: #495057;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .sample-btn:hover {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Navigation bar with social buttons
st.markdown("""
    <style>
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #1e293b; /* Dark blue-gray */
            padding: 8px 16px;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
            font-family: 'Segoe UI', sans-serif;
        }
        .navbar-title {
            font-size: 15px;
            font-weight: 500;
            color: #f1f5f9;
            display: flex;
            align-items: center;
        }
        .navbar-logo {
            font-size: 18px;
            margin-right: 6px;
        }
        .social-container {
            display: flex;
            gap: 12px;
        }
        .social-button {
            display: flex;
            align-items: center;
            font-size: 13px;
            text-decoration: none;
            color: #f1f5f9;
            background: #334155;
            padding: 5px 10px;
            border-radius: 20px;
            transition: 0.3s;
        }
        .social-button img {
            margin-right: 6px;
        }
        .social-button:hover {
            background: #475569;
            transform: translateY(-2px);
        }
    </style>

    <div class="navbar">
        <div class="navbar-title">
            <span class="navbar-logo">📰</span>
            News Classifier
        </div>
        <div class="social-container">
            <a href="https://github.com/RAFIQ-ABDUL" target="_blank" class="social-button">
                <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="16">
                GitHub
            </a>
            <a href="https://www.linkedin.com/in/abdul-manan-710448258/" target="_blank" class="social-button">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="16">
                LinkedIn
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)



# Load the pre-trained model and tokenizer from the local directory
@st.cache_resource
def load_model():
    model_path = "news_classifier_model"
    try:
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device="cpu"
        )
        return classifier
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


# Define category mapping
category_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
category_icons = {
    "World": "🌍",
    "Sports": "⚽",
    "Business": "💼",
    "Sci/Tech": "🔬"
}


# Function to predict news category
def predict_news_category(text, classifier):
    try:
        result = classifier(text)[0]
        label = int(result['label'].split('_')[-1])
        return category_map[label]
    except Exception as e:
        return f"Error in prediction: {str(e)}"


# App layout
st.markdown('<h1 class="title">News Headline Classifier</h1>', unsafe_allow_html=True)

# Display info about the model
with st.expander("ℹ️ About this app"):
    st.write("""
    This app uses a fine-tuned BERT model trained on the AG News dataset to classify news headlines into one of four categories:
    - 🌍 World News
    - ⚽ Sports
    - 💼 Business
    - 🔬 Science/Technology

    Simply type a news headline in the text box below and click 'Predict' to see the category.
    """)

# Load model
classifier = load_model()

if classifier is not None:
    # Initialize session state for user input if it doesn't exist
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    # Text input area with improved UI
    user_input = st.text_area(
        "**Enter News Headline:**",
        value=st.session_state.user_input,
        placeholder="Type your news headline here...",
        height=100,
        key="text_input"
    )

    # Update session state with current input
    st.session_state.user_input = user_input

    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🚀 Predict Category", use_container_width=True)

    # Handle prediction
    if predict_button:
        if user_input.strip():
            with st.spinner('Analyzing headline...'):
                # Add a small delay for better UX
                time.sleep(0.5)
                category = predict_news_category(user_input, classifier)

                if not category.startswith("Error"):
                    # Display prediction with styling
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: #2C3E50; margin-bottom: 10px;">Prediction Result</h3>
                        <p style="font-size: 18px; margin-bottom: 5px;">The news headline is categorized as:</p>
                        <h2 style="color: #2E86C1; margin-top: 5px;">{category_icons[category]} {category}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(category)
        else:
            st.warning("⚠️ Please enter a news headline before predicting.")

    # Add some sample headlines for quick testing
    st.markdown("---")
    st.subheader("💡 Try these sample headlines:")

    samples = [
        "Stock markets reach all-time high amid economic recovery",
        "Scientists discover new species in Amazon rainforest",
        "Local team wins championship after dramatic final match",
        "International summit addresses climate change policies"
    ]

    # Create columns for sample buttons
    sample_cols = st.columns(2)
    for i, sample in enumerate(samples):
        with sample_cols[i % 2]:
            if st.button(sample, key=f"sample_{i}"):
                st.session_state.user_input = sample
                st.rerun()

else:
    st.error("Model could not be loaded. Please check if the model files are in the correct location.")