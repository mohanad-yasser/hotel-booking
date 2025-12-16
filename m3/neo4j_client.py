# neo4j_client.py
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def run(self, query, params=None):
        if params is None:
            params = {}

        with self.driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]

neo4j_client = Neo4jClient()