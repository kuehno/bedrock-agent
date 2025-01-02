## Local RAG Example using pgvector

This example demonstrates how to use pgvector for a local Retrieval-Augmented Generation (RAG) setup.

### Prerequisites

- Docker
- Docker Compose

### Setup

1. Install the required Python packages:
    ```sh
    pip install pgvector==0.3.6 psycopg2-binary==2.9.10
    ```

2. Install bedrock_agent (see documentation in module)

3. Start the PostgreSQL database using Docker Compose:
    ```sh
    docker-compose -f docker-compose.yaml up
    ```

### Running the Example

1. Open the Jupyter notebook demo_rag_pgvector.ipynb.

2. Follow the steps in the notebook to:
    - Generate and ingest facts into the vector database.
    - Query the knowledge base for similar facts.

#### Configuration

Update the connection parameters in the notebook:
```python
conn_params = {
    'dbname': 'postgres', # default db name
    'host': '172.18.0.2', # IP address of the Postgres container
    'user': 'postgres',   # default user name
    'password': 'example' # some password
}
```

Set the table name and dimensions:
```python
table_name = "rag_example" # name of the embedding table
dimensions = 256
```
