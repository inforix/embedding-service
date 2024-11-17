import unittest
from langchain_openai.embeddings.base import OpenAIEmbeddings

class TestOpenAIEmbeddings(unittest.TestCase):
    def setUp(self):
        """
        Setup before each test. Initialize OpenAIEmbeddings instance.
        """
        self.embedding_service = OpenAIEmbeddings(
            model="tao-8k",  # Specify the OpenAI embedding model
            openai_api_key="your_openai_api_key_here",  # Provide your API key
            base_url='http://localhost:8000/v1'
        )
        self.test_text = "This is a test sentence."

    # def test_embedding_generation(self):
    #     """
    #     Test if the embedding is successfully generated.
    #     """
    #     embedding = self.embedding_service.embed_query(self.test_text)
    #     self.assertIsInstance(embedding, list, "Embedding should be a list of floats.")
    #     self.assertTrue(len(embedding) > 0, "Embedding list should not be empty.")
    #     self.assertTrue(all(isinstance(x, float) for x in embedding), "All elements in the embedding should be floats.")

    def test_batch_embedding_generation(self):
        """
        Test batch embedding generation for multiple sentences.
        """
        sentences = ["This is the first sentence.", "Here is another one."]
        embeddings = self.embedding_service.embed_documents(sentences)
        self.assertIsInstance(embeddings, list, "Embeddings should be a list of lists.")
        self.assertEqual(len(embeddings), len(sentences), "Embeddings should match the number of input sentences.")
        for embedding in embeddings:
            self.assertIsInstance(embedding, list, "Each embedding should be a list of floats.")
            self.assertTrue(len(embedding) > 0, "Each embedding list should not be empty.")
            self.assertTrue(all(isinstance(x, float) for x in embedding), "All elements in embeddings should be floats.")

if __name__ == "__main__":
    unittest.main()
