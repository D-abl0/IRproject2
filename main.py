import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def get_doc_id(self, doc):
        """
        Splits each line of the document into doc_id and text content.
        """
        arr = doc.split("\t")
        return int(arr[0]), arr[1]

    def tokenizer(self, text):
        """
        Processes and tokenizes document text.
        - Converts text to lowercase
        - Removes special characters, keeping only alphanumeric characters and spaces
        - Removes stopwords
        - Applies stemming to each token
        - Removes duplicate tokens to ensure unique terms
        """
        # Convert to lowercase and remove special characters
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        # Tokenize by whitespace
        tokens = cleaned_text.split()
        # Remove stop words and apply stemming, then ensure tokens are unique
        processed_tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        # Remove duplicates by converting to set, then back to list
        unique_tokens = list(set(processed_tokens))
        return unique_tokens



import math

class Node:
    def __init__(self, value=None, next=None):
        """
        Node structure for each element in a linked list (postings list).
        Parameters:
        - value: Document ID
        - next: Pointer to the next node in the list
        Additional attributes:
        - skip: Optional skip pointer for faster traversal
        - tf_idf: Placeholder for tf-idf score (to be set in Indexer)
        """
        self.value = value
        self.next = next
        self.skip = None  # Optional skip pointer for optimized retrieval
        self.tf_idf = 0.0  # Placeholder for tf-idf score

import math

import math

class LinkedList:
    """
    Class to define a linked list (postings list). Each element in the linked list is of the type 'Node'.
    Each term in the inverted index has an associated linked list object.
    Feel free to add additional functions to this class.
    """
    def __init__(self):
        self.start_node = None
        self.end_node = None
        self.length, self.n_skips, self.idf = 0, 0, 0.0
        self.skip_length = None

    def traverse_list(self):
        traversal = []
        if self.start_node is None:
            return traversal
        else:
            """ Write logic to traverse the linked list.
                To be implemented."""
            current = self.start_node
            while current:
                traversal.append(current.value)
                current = current.next
            return traversal

    def traverse_skips(self):
        traversal = []
        if self.start_node is None:
            return traversal
        else:
            """ Write logic to traverse the linked list using skip pointers.
                To be implemented."""
            current = self.start_node
            while current:
                traversal.append(current.value)
                # Use skip pointer if available, otherwise move to the next node
                current = current.skip if current.skip else current.next
            return traversal

    def add_skip_connections(self):
        n_skips = math.floor(math.sqrt(self.length))
        if n_skips * n_skips == self.length:
            n_skips = n_skips - 1
        """ Write logic to add skip pointers to the linked list.
            This function does not return anything.
            To be implemented."""
        self.n_skips = n_skips
        self.skip_length = math.ceil(self.length / self.n_skips) if self.n_skips else self.length

        current = self.start_node
        count = 0
        last_skip_node = None

        while current:
            if count % self.skip_length == 0:
                if last_skip_node:
                    last_skip_node.skip = current
                last_skip_node = current
            count += 1
            current = current.next

    def insert_at_end(self, value):
        """
        Write logic to add new elements to the linked list.
        Insert the element at an appropriate position, such that elements to the left are lower than the inserted
        element, and elements to the right are greater than the inserted element.
        To be implemented.
        """
        new_node = Node(value)
        if self.start_node is None:
            # Initialize start and end if list is empty
            self.start_node = self.end_node = new_node
        else:
            # Insert in ascending order
            if value < self.start_node.value:
                new_node.next = self.start_node
                self.start_node = new_node
            else:
                current = self.start_node
                while current.next and current.next.value < value:
                    current = current.next
                new_node.next = current.next
                current.next = new_node
                if new_node.next is None:
                    self.end_node = new_node
        self.length += 1




import math
from collections import OrderedDict

import math
from collections import OrderedDict

class Indexer:
    def __init__(self):
        """ Add more attributes if needed"""
        self.inverted_index = OrderedDict({})
        self.doc_lengths = {}  # Additional attribute for document length tracking, if needed

    def get_index(self):
        """ Function to get the index.
            Already implemented."""
        return self.inverted_index

    def generate_inverted_index(self, doc_id, tokenized_document):
        """ This function adds each tokenized document to the index. This in turn uses the function add_to_index
            Already implemented."""
        for t in tokenized_document:
            self.add_to_index(t, doc_id)

    def add_to_index(self, term_, doc_id_):
        """ This function adds each term & document id to the index.
            If a term is not present in the index, then add the term to the index & initialize a new postings list (linked list).
            If a term is present, then add the document to the appropriate position in the postings list of the term.
            To be implemented."""
        if term_ not in self.inverted_index:
            self.inverted_index[term_] = LinkedList()  # Initialize new postings list if term is new
        self.inverted_index[term_].insert_at_end(doc_id_)  # Add doc_id to the term’s postings list
        self.doc_lengths[doc_id_] = self.doc_lengths.get(doc_id_, 0) + 1  # Update document length tracking

    def sort_terms(self):
        """ Sorting the index by terms.
            Already implemented."""
        sorted_index = OrderedDict({})
        for k in sorted(self.inverted_index.keys()):
            sorted_index[k] = self.inverted_index[k]
        self.inverted_index = sorted_index

    def add_skip_connections(self):
        """ For each postings list in the index, add skip pointers.
            To be implemented."""
        for postings_list in self.inverted_index.values():
            postings_list.add_skip_connections()  # Add skip pointers in each postings list

    def calculate_tf_idf(self):
        """ Calculate tf-idf score for each document in the postings lists of the index.
            To be implemented."""
        total_docs = len(self.doc_lengths)  # Total number of documents
        for term, postings_list in self.inverted_index.items():
            doc_count = postings_list.length  # Number of documents containing the term
            idf = math.log(total_docs / doc_count) if doc_count else 0
            postings_list.idf = idf  # Assign idf to the postings list for the term

            current_node = postings_list.start_node
            while current_node:
                tf = 1 / self.doc_lengths[current_node.value]  # Calculate tf as binary term frequency
                current_node.tf_idf = tf * idf  # Calculate and assign tf-idf
                current_node = current_node.next


from tqdm import tqdm
from collections import OrderedDict
import inspect as inspector
import sys
import argparse
import json
import time
import random
import flask
from flask import Flask
from flask import request
import hashlib

app = Flask(__name__)

class ProjectRunner:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.indexer = Indexer()

    def _merge(self, list1, list2):
        """
        Merge two postings lists while tracking comparisons properly.
        Preserves the maximum tf-idf value of a document.
        """
        merged_list = []
        comparisons = 0
        ptr1, ptr2 = list1.start_node, list2.start_node

        while ptr1 and ptr2:
            comparisons += 1
            if ptr1.value == ptr2.value:
                # When equal, store both doc_id and tf-idf score
                merged_list.append((ptr1.value, max(ptr1.tf_idf, ptr2.tf_idf)))
                ptr1, ptr2 = ptr1.next, ptr2.next
            elif ptr1.value < ptr2.value:
                if ptr1.skip and ptr1.skip.value <= ptr2.value:
                    while ptr1.skip and ptr1.skip.value <= ptr2.value:
                        ptr1 = ptr1.skip
                        comparisons += 1
                else:
                    ptr1 = ptr1.next
            else:
                if ptr2.skip and ptr2.skip.value <= ptr1.value:
                    while ptr2.skip and ptr2.skip.value <= ptr1.value:
                        ptr2 = ptr2.skip
                        comparisons += 1
                else:
                    ptr2 = ptr2.next

        return merged_list, comparisons

    def _daat_and(self, term_lists):
        """
        Implement the DAAT AND algorithm, which merges the postings list of N query terms.
        Use appropriate parameters & return types.
        To be implemented.
        """
        if not term_lists:
            return [], 0
        merged_list = term_lists[0]
        total_comparisons = 0

        for postings_list in term_lists[1:]:
            merged_list, comparisons = self._merge(merged_list, postings_list)
            total_comparisons += comparisons

        # Only document IDs needed for DAAT result
        final_results = [doc[0] for doc in merged_list]
        return final_results, total_comparisons


    def _get_postings(self, term):
        """
        Function to get the postings list of a term from the index.
        Use appropriate parameters & return types.
        To be implemented.
        """
        postings_list = self.indexer.inverted_index.get(term, LinkedList())
        return postings_list, postings_list

    def _output_formatter(self, op):
        """ This formats the result in the required format.
            Do NOT change.
        """
        if op is None or len(op) == 0:
            return [], 0
        op_no_score = [int(i) for i in op]
        results_cnt = len(op_no_score)
        return op_no_score, results_cnt

    def run_indexer(self, corpus):
        """
        This function reads & indexes the corpus. After creating the inverted index,
        it sorts the index by the terms, add skip pointers, and calculates the tf-idf scores.
        Already implemented, but you can modify the orchestration, as you seem fit.
        """
        with open(corpus, 'r') as fp:
            for line in tqdm(fp.readlines()):
                doc_id, document = self.preprocessor.get_doc_id(line)
                tokenized_document = self.preprocessor.tokenizer(document)
                self.indexer.generate_inverted_index(doc_id, tokenized_document)
        self.indexer.sort_terms()
        self.indexer.add_skip_connections()
        self.indexer.calculate_tf_idf()


    def _output_formatter(self, op):
        """
        This formats the result in the required format.
        Do NOT change.
        """
        if op is None or len(op) == 0:
            return [], 0
        op_no_score = [int(i) for i in op]
        results_cnt = len(op_no_score)
        return op_no_score, results_cnt

    def run_queries(self, query_list):
        output_dict = {
            'postingsList': {},
            'postingsListSkip': {},
            'daatAnd': {},
            'daatAndSkip': {},
            'daatAndTfIdf': {},
            'daatAndSkipTfIdf': {}
        }

        for query in tqdm(query_list):
            input_term_arr = self.preprocessor.tokenizer(query)
            
            term_postings = []
            term_postings_skip = []
            term_tf_idfs = {}  # Store tf-idf scores for each term

            for term in input_term_arr:
                postings, skip_postings = self._get_postings(term)
                term_postings.append(postings)
                term_postings_skip.append(skip_postings)
                
                # Store tf-idf scores for the term
                current = postings.start_node
                while current:
                    if current.value not in term_tf_idfs:
                        term_tf_idfs[current.value] = 0
                    term_tf_idfs[current.value] = max(term_tf_idfs[current.value], current.tf_idf)
                    current = current.next

                output_dict['postingsList'][term] = postings.traverse_list()
                output_dict['postingsListSkip'][term] = skip_postings.traverse_skips()

            # Regular DAAT AND
            and_op_no_skip, and_comparisons_no_skip = self._daat_and(term_postings)
            and_op_no_skip.sort()  # Sort by doc ID in ascending order

            # DAAT AND with skip
            and_op_skip, and_comparisons_skip = self._daat_and(term_postings_skip)
            and_op_skip.sort()  # Sort by doc ID in ascending order

            # For TF-IDF sorting, create list of (doc_id, tf_idf_score) tuples
            and_op_tfidf = [(doc_id, term_tf_idfs[doc_id]) for doc_id in and_op_no_skip]
            and_op_skip_tfidf = [(doc_id, term_tf_idfs[doc_id]) for doc_id in and_op_skip]

            # Sort by tf-idf score in descending order
            and_op_tfidf.sort(key=lambda x: (-x[1], x[0]))
            and_op_skip_tfidf.sort(key=lambda x: (-x[1], x[0]))

            # Extract just the doc IDs after sorting
            and_op_no_skip_sorted = [doc_id for doc_id, _ in and_op_tfidf]
            and_op_skip_sorted = [doc_id for doc_id, _ in and_op_skip_tfidf]

            # Use output formatter
            and_op_no_score_no_skip, and_results_cnt_no_skip = self._output_formatter(and_op_no_skip)
            and_op_no_score_skip, and_results_cnt_skip = self._output_formatter(and_op_skip)
            and_op_no_score_no_skip_sorted, and_results_cnt_no_skip_sorted = self._output_formatter(and_op_no_skip_sorted)
            and_op_no_score_skip_sorted, and_results_cnt_skip_sorted = self._output_formatter(and_op_skip_sorted)

            # Populate output_dict
            output_dict['daatAnd'][query.strip()] = {
                'results': and_op_no_score_no_skip,
                'num_docs': and_results_cnt_no_skip,
                'num_comparisons': and_comparisons_no_skip
            }
            output_dict['daatAndSkip'][query.strip()] = {
                'results': and_op_no_score_skip,
                'num_docs': and_results_cnt_skip,
                'num_comparisons': and_comparisons_skip
            }
            output_dict['daatAndTfIdf'][query.strip()] = {
                'results': and_op_no_score_no_skip_sorted,
                'num_docs': and_results_cnt_no_skip_sorted,
                'num_comparisons': and_comparisons_no_skip
            }
            output_dict['daatAndSkipTfIdf'][query.strip()] = {
                'results': and_op_no_score_skip_sorted,
                'num_docs': and_results_cnt_skip_sorted,
                'num_comparisons': and_comparisons_skip
            }

        return output_dict



@app.route("/execute_query", methods=['POST'])
def execute_query():
    """ This function handles the POST request to your endpoint.
        Do NOT change it."""
    start_time = time.time()

    queries = request.json["queries"]


    """ Running the queries against the pre-loaded index. """
    output_dict = runner.run_queries(queries)

    """ Dumping the results to a JSON file. """
    with open(output_location, 'w') as fp:
        json.dump(output_dict, fp)

    response = {
        "Response": output_dict,
        "time_taken": str(time.time() - start_time),
        "username_hash": username_hash
    }
    return flask.jsonify(response)



import hashlib
from project_runner import ProjectRunner  # Assuming you have a ProjectRunner module

if __name__ == "__main__":
    """Driver code for the project, defines global variables, and initializes the project."""

    # Directly specify file paths and username
    corpus = "input_corpus.txt"  # File in the same directory
    output_location = "project2_output.json"
    username = "cgurram"  # Replace with your actual UB username
    
    # Generate a username hash
    username_hash = hashlib.md5(username.encode()).hexdigest()
    
    # Initialize and run the project
    runner = ProjectRunner()
    runner.run_indexer(corpus)
    
    # Start the app
    app.run(host="0.0.0.0", port=9999)
