import os

from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

from raggie.utils import batch_iterate

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))


class QdrantVDB:
    def __init__(
        self, collection_name, vector_dim=768, batch_size=512, bq=False
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self.bq = bq

    # TODO: As the client and collections have a one to one mapping with the
    # class instance and are never changed, we can make them private
    # and internal variables
    def define_client(self):
        self.client = QdrantClient(
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            # grpc in simple words is faster version of https for a local setup
            prefer_grpc=True,
        )

    def create_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                # NOTE: We use similarity search with dot product, and store
                # the vectors on disk instead of memory to optimize
                # memory usage for large datasets
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.DOT,
                    on_disk=True,
                ),
                # NOTE: Optimizer config is necessary to optimize storage
                # and indexing performance
                # NOTE: By default the vector db will refresh it's indexes
                # whenever a new vector comes in, and we don't want that many
                # frequent updates so we set a higher value of indexing_threshold
                # until which the collection should not update it's indexes.
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=9, indexing_threshold=30000
                ),
                # NOTE: Adding quantization config to
                # enable binary quantization
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                )
                if self.bq
                else None,
            )

    def ingest_data(self, embeddata):
        # Zip the contexts and embeddings into pairs
        # (eagerly convert to list for len())
        paired_data = list(
            zip(embeddata.contexts, embeddata.embeddings, strict=False)
        )
        # Iterate over zipped batches of (context, embedding) pairs
        for batch in tqdm(
            batch_iterate(paired_data, self.batch_size),
            total=len(paired_data) // self.batch_size,
            desc="Ingesting in batches",
        ):
            # Unzip the batch into separate lists of contexts and embeddings
            batch_contexts, batch_embeddings = zip(*batch, strict=False)

            # Upload the batch to the collection
            # For each batch, we invoke the .client.upload_collection
            # to store the embeddings and their associated metadata (payload).
            # Payload stores metadata such as the original context for each vector.

            self.client.upload_collection(
                collection_name=self.collection_name,
                vectors=batch_embeddings,  # List of embedding vectors
                payload=[
                    {"context": context} for context in batch_contexts
                ],  # Associated metadata
            )

        # Configuration to update the collection only if the total
        # data ingested in the latest run exceeds a certain threshold.
        # We specify the threshold, so that we are not updating the
        # vector db as soon as a new entry is added, but rather
        # after a certain number of entries have been added.
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=30000
            ),
        )
