# Import necessary library
from transformers import pipeline

# Load the zero-shot classification pipeline with BART model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the candidate categories for classification
categories = ["Entertainment", "Politics", "Health", "Education", "Sports"]

def classify_comment(text):
    """
    Classifies the given comment into one of the predefined categories.

    Parameters:
        text (str): The input comment to classify.

    Returns:
        str: Predicted category.
    """
    result = classifier(text, candidate_labels=categories)
    return result['labels'][0]  # Top predicted label

# --------- Example Usage ---------

# Sample comment
# comment = "The new movie is breaking box office records."
comment = "India won the ICC champions trophy in 2025 ."

# Get predicted category
predicted_category = classify_comment(comment)

# Print the result
print("Comment:", comment)
print("Predicted Category:", predicted_category)
        