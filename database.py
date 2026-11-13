
import sqlite3
import numpy as np

class FaceDatabase:
    def __init__(self):
        self.conn = sqlite3.connect("faces.db")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)

        self.conn.commit()

    def add_embedding(self, name, embedding):
        # Convert NumPy array into binary data
        vector = np.asarray(
            embedding, dtype=np.float32
        ).reshape(1, -1)

        self.conn.execute(
            """INSERT INTO face_embeddings (name, embedding)
               VALUES (?, ?)""",
            (name, vector.tobytes())
        )

        self.conn.commit()

    def load_embeddings(self):
        rows = self.conn.execute(
            "SELECT name, embedding FROM face_embeddings"
        ).fetchall()

        result = []

        for name, data in rows:
            vector = np.frombuffer(
                data, dtype=np.float32
            ).copy().reshape(1, -1)

            result.append((name, vector))

        return result

    def close(self):
        self.conn.close()
