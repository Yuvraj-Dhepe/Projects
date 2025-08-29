import sys
import time

from qdrant_client import models


class Retriever:
    def __init__(self, vector_db, embeddata, bq=False):
        self.vector_db = vector_db
        self.embeddata = embeddata
        self.bq = bq

    def search(self, query):
        # Use hf function to get the query embedding
        query_embedding = self.embeddata.embed_model.get_query_embedding(
            query
        )
        print(
            f"Size of the vector embedding {sys.getsizeof(query_embedding)}"
        )
        #
        start_time = time.time()
        result = self.vector_db.client.query_points(
            collection_name=self.vector_db.collection_name,
            query=query_embedding,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=self.bq, rescore=True, oversampling=2.0
                )
                # NOTE: Ignore quantization during search for high precision
                # NOTE: Rescore the results after the initial quantized search
                # for better accuracy
                # NOTE: Oversampling to fetch additional candidates to improve
                # result quality
            ),
            timeout=1000,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Execution time for search: {elapsed_time:.2f} seconds")

        return result
