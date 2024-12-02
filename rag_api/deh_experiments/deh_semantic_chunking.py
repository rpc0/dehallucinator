"""
Semantic Chunker based on:
https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/a4570f3c4883eb9b835b0ee18990e62298f518ef/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

"""

from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


class SemanticChunker:

    # ============================================================================
    # Constructor to initialize the attributes of the class
    def __init__(self, embedding_model_name="mxbai-embed-large"):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = OllamaEmbeddings(model=self.embedding_model_name)

    # ============================================================================
    def combine_sentences(self, sentences, buffer_size=1):
        # Go through each sentence dict
        for i in range(len(sentences)):

            # Create a string that will hold the sentences which are joined
            combined_sentence = ''

            # Add sentences before the current one, based on the buffer size.
            for j in range(i - buffer_size, i):
                # Check if the index j is not negative (to avoid index out of range like on the first one)
                if j >= 0:
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += sentences[j]['sentence'] + ' '

            # Add the current sentence
            combined_sentence += sentences[i]['sentence']

            # Add sentences after the current one, based on the buffer size
            for j in range(i + 1, i + 1 + buffer_size):
                # Check if the index j is within the range of the sentences list
                if j < len(sentences):
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += ' ' + sentences[j]['sentence']

            # Then add the whole thing to your dict
            # Store the combined sentence in the current sentence dict
            sentences[i]['combined_sentence'] = combined_sentence

        return sentences
    
    # ============================================================================
    def calculate_cosine_distances(self, sentences):
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['combined_sentence_embedding']
            embedding_next = sentences[i + 1]['combined_sentence_embedding']
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            
            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)

            # Store distance in the dictionary
            sentences[i]['distance_to_next'] = distance

        # Optionally handle the last sentence
        # sentences[-1]['distance_to_next'] = None  # or a default value

        return distances, sentences

    # ============================================================================
    # Method to do semantic chunking on a string (=text)
    def split_text(self, text):
        single_sentences_list = re.split(r'(?<=[.?!])\s+', text)
        sentences = [{'sentence': x, 'index': i} for i, x in enumerate(single_sentences_list)]
        sentences = self.combine_sentences(sentences)
        embeddings = self.embedding_model.embed_documents([x['combined_sentence'] for x in sentences])
        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]

        print(f"In split_text, sentences: {sentences[0]}")
        
        distances, sentences = self.calculate_cosine_distances(sentences)
        breakpoint_percentile_threshold = 95
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold) # If you want more chunks, lower the percentile cutoff
        # num_distances_above_theshold = len([x for x in distances if x > breakpoint_distance_threshold]) # The amount of distances above your threshold        
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list        

        # Initialize the start index
        start_index = 0

        # Create a list to hold the grouped sentences
        chunks = []

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            
            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        # grouped_sentences now contains the chunked sentences

        return chunks
    