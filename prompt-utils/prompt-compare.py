from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

def get_sentences():
    # Check if sufficient arguments are provided
    if len(sys.argv) == 3:
        sentence1, sentence2 = sys.argv[1], sys.argv[2]
    else:
        # Prompt user for input if arguments are not provided
        sentence1 = input("Please enter the first sentence: ")
        sentence2 = input("Please enter the second sentence: ")

    return sentence1, sentence2



def calculate_similarity(sentence1=None, sentence2=None):
    if sentence1 is None or sentence2 is None:
        print("Error: You have to pass two sentences for calculating similarity", file=sys.stderr)
        return None, None
    
    # Vectorize the sentences
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([sentence1, sentence2])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    
    # Convert to percentage
    percentage_similarity = 50 * (cosine_sim + 1)  # Scale from 0% to 100%
    
    return cosine_sim, percentage_similarity

def main():
    sentence1, sentence2 = get_sentences()
    print(f"Sentence 1: {sentence1}")
    print(f"Sentence 2: {sentence2}")
    similarity_score, similarity_percentage = calculate_similarity(sentence1, sentence2)
    print(f"Cosine Similarity Score: {similarity_score}")
    print(f"Similarity Percentage: {similarity_percentage}%")

if __name__ == "__main__":
    main()
    
# Example usage
#similarity_score, similarity_percentage = calculate_similarity("Generate an impressionist painting of a strawberry field with gentle rolling hills in the background", "A photograph of a strawberry field in a valley surrounded by hills")
#print(f"Cosine Similarity Score: {similarity_score}")
#print(f"Similarity Percentage: {similarity_percentage}%")
