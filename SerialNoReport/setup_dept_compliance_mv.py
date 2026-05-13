"""
Setup script: populates item_dept_map in WH from century_penetration.sales,
then creates/refreshes mv_dept_compliance.

Run once to set up, then re-run whenever you want to refresh the mapping.
"""
import psycopg2
import sys

WH_CONFIG = dict(host='localhost', port=3307, user='postgres', password='hello', database='WH')
CP_CONFIG = dict(host='localhost', port=3307, user='postgres', password='hello', database='century_penetration')


def log(msg):
    print(msg, flush=True)


def main():
    # ── 1. Pull distinct item_code → dept/grp/sub_group from century_penetration ──
    log("Connecting to century_penetration ...")
    conn_cp = psycopg2.connect(**CP_CONFIG)
    cur_cp = conn_cp.cursor()

    log("Fetching item dept mapping from century_penetration.sales ...")
    cur_cp.execute("""
        SELECT DISTINCT
            TRIM(item_code)    AS item_code,
            MAX(dept)          AS dept,
            MAX(groups)        AS grp,
            MAX(sub_group)     AS sub_group
        FROM sales
        WHERE item_code IS NOT NULL AND TRIM(item_code) <> ''
          AND dept IS NOT NULL AND TRIM(dept) <> ''
        GROUP BY TRIM(item_code)
        ORDER BY item_code
    """)
    rows = cur_cp.fetchall()
    conn_cp.close()
    log(f"  Fetched {len(rows)} distinct item codes with dept info.")

    # ── 2. Upsert into WH.item_dept_map ──
    log("Connecting to WH database ...")
    conn_wh = psycopg2.connect(**WH_CONFIG)
    cur_wh = conn_wh.cursor()

    cur_wh.execute("""
        CREATE TABLE IF NOT EXISTS item_dept_map (
            item_code   TEXT PRIMARY KEY,
            dept        TEXT,
            grp         TEXT,
            sub_group   TEXT
        )
    """)

    cur_wh.execute("""
        CREATE INDEX IF NOT EXISTS idx_item_dept_map_dept     ON item_dept_map(dept);
    """)
    cur_wh.execute("""
        CREATE INDEX IF NOT EXISTS idx_item_dept_map_grp      ON item_dept_map(grp);
    """)
    cur_wh.execute("""
        CREATE INDEX IF NOT EXISTS idx_item_dept_map_subgroup ON item_dept_map(sub_group);
    """)

    log("Upserting item dept map ...")
    upserted = 0
    for row in rows:
        cur_wh.execute("""
            INSERT INTO item_dept_map (item_code, dept, grp, sub_group)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (item_code) DO UPDATE
              SET dept      = EXCLUDED.dept,
                  grp       = EXCLUDED.grp,
                  sub_group = EXCLUDED.sub_group
        """, row)
        upserted += 1

    conn_wh.commit()
    log(f"  Upserted {upserted} rows into item_dept_map.")

    # ── 3. Create / Refresh materialized view ──
    log("Creating materialized view mv_dept_compliance ...")
    cur_wh.execute("""
        SELECT COUNT(*) FROM pg_matviews
        WHERE schemaname = 'public' AND matviewname = 'mv_dept_compliance'
    """)
    mv_exists = cur_wh.fetchone()[0] > 0

    if mv_exists:
        log("  MV exists — refreshing ...")
        cur_wh.execute("REFRESH MATERIALIZED VIEW mv_dept_compliance")
    else:
        cur_wh.execute("""
            CREATE MATERIALIZED VIEW mv_dept_compliance AS
            SELECT
                DATE(s.bill_date)                                               AS activity_date,
                s.shop_code,
                COALESCE(m.dept,     'Unclassified')                           AS dept,
                COALESCE(m.grp,      'Unclassified')                           AS grp,
                COALESCE(m.sub_group,'Unclassified')                           AS sub_group,
                COUNT(*)                                                        AS total_serials,
                SUM(CASE WHEN UPPER(TRIM(s.serial_check)) = 'Y' THEN 1 ELSE 0 END) AS total_yes,
                ROUND(
                    SUM(CASE WHEN UPPER(TRIM(s.serial_check)) = 'Y' THEN 1 ELSE 0 END)::NUMERIC
                    / NULLIF(COUNT(*), 0) * 100, 1
                )                                                               AS compliance_pct
            FROM serialno_check_yes_no s
            LEFT JOIN item_dept_map m ON TRIM(s.item_code) = TRIM(m.item_code)
            WHERE s.bill_date IS NOT NULL
              AND s.item_code IS NOT NULL
              AND TRIM(s.item_code) <> ''
            GROUP BY DATE(s.bill_date), s.shop_code, m.dept, m.grp, m.sub_group
            WITH DATA
        """)

        log("  Creating indexes on mv_dept_compliance ...")
        for idx_sql in [
            "CREATE INDEX idx_mv_dept_compliance_date      ON mv_dept_compliance(activity_date)",
            "CREATE INDEX idx_mv_dept_compliance_dept      ON mv_dept_compliance(dept)",
            "CREATE INDEX idx_mv_dept_compliance_grp       ON mv_dept_compliance(grp)",
            "CREATE INDEX idx_mv_dept_compliance_subgroup  ON mv_dept_compliance(sub_group)",
            "CREATE INDEX idx_mv_dept_compliance_shop      ON mv_dept_compliance(shop_code)",
            "CREATE INDEX idx_mv_dept_compliance_date_dept ON mv_dept_compliance(activity_date, dept)",
        ]:
            cur_wh.execute(idx_sql)

    conn_wh.commit()

    # ── 4. Quick sanity check ──
    cur_wh.execute("SELECT COUNT(*) FROM mv_dept_compliance")
    total = cur_wh.fetchone()[0]
    cur_wh.execute("SELECT dept, SUM(total_serials), ROUND(SUM(total_yes)::NUMERIC/NULLIF(SUM(total_serials),0)*100,1) FROM mv_dept_compliance GROUP BY dept ORDER BY SUM(total_serials) DESC")
    rows_check = cur_wh.fetchall()
    conn_wh.close()

    log(f"\n✅ mv_dept_compliance ready — {total} rows total.")
    log("\nDept summary:")
    for r in rows_check:
        log(f"  {str(r[0]):35s}  serials={r[1]:6d}  compliance={r[2]}%")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log(f"\n❌ Error: {e}")
        sys.exit(1)
