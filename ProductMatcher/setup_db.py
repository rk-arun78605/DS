"""Run once to create required PostgreSQL tables."""
import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'dbname': 'barcode',
}

CREATE_UPLOADED_ITEMS = """
CREATE TABLE IF NOT EXISTS uploaded_items (
    id              SERIAL PRIMARY KEY,
    session_id      VARCHAR(64),
    barcode         VARCHAR(100),
    barcode1        VARCHAR(100),
    item_description TEXT,
    uploaded_at     TIMESTAMP DEFAULT NOW()
);
"""

CREATE_ITEMMASTER = """
CREATE TABLE IF NOT EXISTS itemmaster (
    item_code   VARCHAR(50) PRIMARY KEY,
    item_name   VARCHAR(255) NOT NULL,
    barcode     VARCHAR(100),
    status      VARCHAR(20) DEFAULT 'active'
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_uploaded_items_session ON uploaded_items(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_itemmaster_barcode ON itemmaster(barcode);",
]


def setup():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(CREATE_UPLOADED_ITEMS)
    cur.execute(CREATE_ITEMMASTER)
    for idx_sql in CREATE_INDEXES:
        cur.execute(idx_sql)
    conn.commit()
    cur.close()
    conn.close()
    print("Tables created successfully.")
    print("  - uploaded_items  (barcode, barcode1, item_description)")
    print("  - itemmaster      (item_code, item_name, barcode, status)")


if __name__ == '__main__':
    setup()
