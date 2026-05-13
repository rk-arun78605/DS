-- ============================================================
-- Cost Control Portal - PostgreSQL Database Setup
-- Run against: localhost:3307 as postgres user
-- ============================================================

-- Create the database
CREATE DATABASE cost_control_portal;

-- Connect to the new database before running the rest
\c cost_control_portal

-- ============================================================
-- 1. SHOP / STAFF MASTER
-- ============================================================
CREATE TABLE IF NOT EXISTS cc_shop_staff (
    id              SERIAL PRIMARY KEY,
    emp_id          VARCHAR(20) NOT NULL UNIQUE,
    emp_name        VARCHAR(100) NOT NULL,
    shop_code       VARCHAR(20) NOT NULL,
    shop_name       VARCHAR(100) NOT NULL,
    department      VARCHAR(50),
    role            VARCHAR(20) NOT NULL DEFAULT 'staff',  -- 'staff' | 'supervisor' | 'manager'
    approver_emp_id VARCHAR(20),                           -- emp_id of supervisor/manager
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- 2. USER AUTH (custom — stores hashed password via Django)
--    Django auth_user table is used; this maps emp_id to Django user
-- ============================================================

-- ============================================================
-- 3. ITEM MASTER
-- ============================================================
CREATE TABLE IF NOT EXISTS cc_item_master (
    id              SERIAL PRIMARY KEY,
    item_code       VARCHAR(20) NOT NULL DEFAULT 'AAAAA',
    item_name       VARCHAR(200) NOT NULL,
    department      VARCHAR(50),
    grp             VARCHAR(100),
    sub_group       VARCHAR(100),
    uom             VARCHAR(20),
    category        VARCHAR(20) NOT NULL CHECK (category IN ('RAW','Finished','Packaging')),
    cost_price      NUMERIC(12,2) DEFAULT 0,
    selling_price   NUMERIC(12,2) DEFAULT 0,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uidx_item_code ON cc_item_master(item_code);

-- ============================================================
-- 4. WASTAGE INCIDENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS cc_incident (
    id              SERIAL PRIMARY KEY,
    incident_no     VARCHAR(20) NOT NULL UNIQUE,           -- e.g. WI-SPN-20260421-0001
    shop_code       VARCHAR(20) NOT NULL,
    shop_name       VARCHAR(100) NOT NULL,
    department      VARCHAR(50) NOT NULL,
    submitted_by    VARCHAR(20) NOT NULL,                  -- emp_id
    submitted_name  VARCHAR(100),
    item_code       VARCHAR(20) NOT NULL DEFAULT 'AAAAA',
    item_name       VARCHAR(200) NOT NULL,
    category        VARCHAR(20) NOT NULL,
    quantity        NUMERIC(12,3) NOT NULL,
    uom             VARCHAR(20),
    selling_price   NUMERIC(12,2) NOT NULL DEFAULT 0,
    total_value     NUMERIC(12,2) GENERATED ALWAYS AS (quantity * selling_price) STORED,
    reason          VARCHAR(30) NOT NULL CHECK (reason IN ('Expired','Over Production','Damage','Quality Issue','Handling Error','Others')),
    remarks         TEXT,
    status          VARCHAR(20) NOT NULL DEFAULT 'Pending' CHECK (status IN ('Pending','Approved','Rejected')),
    submit_date     TIMESTAMP NOT NULL DEFAULT NOW(),
    approved_by     VARCHAR(20),
    approved_name   VARCHAR(100),
    approved_date   TIMESTAMP,
    photo_path      TEXT,
    video_path      TEXT,
    photo_hash      VARCHAR(64),                           -- SHA256 for duplicate detection
    video_hash      VARCHAR(64),
    is_late_submission BOOLEAN DEFAULT FALSE,              -- TRUE if submitted after midnight +48h window
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_incident_shop_date  ON cc_incident(shop_code, submit_date);
CREATE INDEX IF NOT EXISTS idx_incident_status     ON cc_incident(status);
CREATE INDEX IF NOT EXISTS idx_incident_submitted  ON cc_incident(submitted_by);
CREATE INDEX IF NOT EXISTS idx_incident_approved   ON cc_incident(approved_by);

-- ============================================================
-- 5. MEDIA DEDUP REGISTRY
-- ============================================================
CREATE TABLE IF NOT EXISTS cc_media_registry (
    id              SERIAL PRIMARY KEY,
    incident_no     VARCHAR(20) NOT NULL,
    media_type      VARCHAR(10) NOT NULL CHECK (media_type IN ('photo','video')),
    file_hash       VARCHAR(64) NOT NULL,
    file_path       TEXT NOT NULL,
    uploaded_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_media_hash ON cc_media_registry(file_hash);

-- ============================================================
-- 6. SAMPLE DATA - Shop Staff
-- ============================================================
INSERT INTO cc_shop_staff (emp_id, emp_name, shop_code, shop_name, department, role, approver_emp_id) VALUES
('EMP001', 'Kofi Mensah',   'SPN', 'Spintex Branch',  'Kitchen',  'staff',      'SUP001'),
('EMP002', 'Ama Asante',    'SPN', 'Spintex Branch',  'Bakery',   'staff',      'SUP001'),
('EMP003', 'Kwame Boateng', 'MSS', 'Maamobi Branch',  'Kitchen',  'staff',      'SUP002'),
('SUP001', 'Eric Owusu',    'SPN', 'Spintex Branch',  NULL,       'supervisor', NULL),
('SUP002', 'Grace Amoah',   'MSS', 'Maamobi Branch',  NULL,       'supervisor', NULL)
ON CONFLICT (emp_id) DO NOTHING;

-- ============================================================
-- 7. SAMPLE DATA - Item Master
-- ============================================================
INSERT INTO cc_item_master (item_code, item_name, department, grp, sub_group, uom, category, cost_price, selling_price) VALUES
('BKR001', 'White Bread Loaf',    'Bakery',  'Bread',    'Sliced Bread',  'PCS', 'Finished',   3.50,  8.00),
('BKR002', 'Chocolate Cake',      'Bakery',  'Cakes',    'Layer Cakes',   'PCS', 'Finished',  12.00, 35.00),
('KIT001', 'Jollof Rice (Large)', 'Kitchen', 'Rice',     'Jollof',        'PCS', 'Finished',   8.00, 25.00),
('KIT002', 'Grilled Chicken',     'Kitchen', 'Protein',  'Chicken',       'PCS', 'Finished',  15.00, 45.00),
('RAW001', 'All Purpose Flour',   'Bakery',  'Raw Mat',  'Flour',         'KG',  'RAW',        2.50,   0.00),
('RAW002', 'Cooking Oil 5L',      'Kitchen', 'Raw Mat',  'Oils',          'BTL', 'RAW',       35.00,   0.00),
('PKG001', 'Cake Box Large',      'Bakery',  'Packaging','Boxes',         'PCS', 'Packaging',  1.20,   0.00),
('AAAAA',  'Miscellaneous Item',  NULL,      NULL,       NULL,            'PCS', 'Finished',   0.00,   0.00)
ON CONFLICT (item_code) DO NOTHING;

-- ============================================================
-- DONE
-- ============================================================
