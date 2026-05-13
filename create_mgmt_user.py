"""
Run once to:
  1. Create the mgmt user (Hello@123)
  2. Grant mgmt master access to ALL dashboards
"""
import psycopg2
import hashlib
from psycopg2.extras import execute_values

DB = dict(host='localhost', port=3307, user='postgres', password='hello', database='salesdata')

ALL_DASHBOARD_IDS = [
    'melcom_star',
    'kpi_dashboard',
    'stst',
    'barcode_matcher',
    'century_penetration',
    'loading_offloading',
    'invoice_scanning',
    'cost_control',
    'pi_dashboard',
    'serial_tracker',
]

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

conn = psycopg2.connect(**DB)
cur  = conn.cursor()

# Ensure tables exist
cur.execute("""
    CREATE TABLE IF NOT EXISTS portal_users (
        username      VARCHAR(50) PRIMARY KEY,
        display_name  VARCHAR(100) NOT NULL,
        password_hash VARCHAR(64)  NOT NULL,
        is_admin      BOOLEAN DEFAULT FALSE,
        created_at    TIMESTAMP DEFAULT NOW()
    )
""")
cur.execute("""
    CREATE TABLE IF NOT EXISTS portal_access (
        username      VARCHAR(50) REFERENCES portal_users(username) ON DELETE CASCADE,
        dashboard_id  VARCHAR(50) NOT NULL,
        granted_at    TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (username, dashboard_id)
    )
""")

# Create / update mgmt user
cur.execute("""
    INSERT INTO portal_users (username, display_name, password_hash, is_admin)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (username) DO UPDATE
      SET display_name  = EXCLUDED.display_name,
          password_hash = EXCLUDED.password_hash,
          is_admin      = EXCLUDED.is_admin
""", ('mgmt', 'Management', hash_pw('Hello@123'), False))

# Grant master access — all dashboards
cur.execute("DELETE FROM portal_access WHERE username = 'mgmt'")
execute_values(
    cur,
    "INSERT INTO portal_access (username, dashboard_id) VALUES %s",
    [('mgmt', did) for did in ALL_DASHBOARD_IDS]
)

conn.commit()
cur.close()
conn.close()

print("✓ User created / updated")
print("  Username : mgmt")
print("  Password : Hello@123")
print("  Role     : Regular user (non-admin)")
print(f"  Access   : {len(ALL_DASHBOARD_IDS)} dashboards granted")
for did in ALL_DASHBOARD_IDS:
    print(f"             ✓ {did}")
