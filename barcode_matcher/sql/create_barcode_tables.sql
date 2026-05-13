-- ============================================================
-- BARCODE MATCHING SYSTEM - DATABASE ARCHITECTURE
-- Created: December 13, 2025
-- Daily Upload: barcodedata.xlsx from sample_data folder
-- ============================================================

-- 1. Master Barcode-Item Mapping Table
-- Columns match source file: VC_ITEM_BARCODE, VC_ITEM_CODE
-- Daily updates from: D:\Dashboard Code\NO_WH\DS\barcode_matcher\sample_data\barcodedata
DROP TABLE IF EXISTS barcode_item_master CASCADE;
CREATE TABLE barcode_item_master (
    id SERIAL PRIMARY KEY,
    vc_item_barcode VARCHAR(50) NOT NULL,
    vc_item_code VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Ensure one barcode maps to one item only
    CONSTRAINT uq_barcode UNIQUE (vc_item_barcode)
);

-- Index for fast barcode lookup (critical for performance)
CREATE INDEX idx_barcode_lookup ON barcode_item_master(vc_item_barcode) WHERE is_active = TRUE;
CREATE INDEX idx_item_code_lookup ON barcode_item_master(vc_item_code) WHERE is_active = TRUE;

-- 2. Upload History Table (audit trail)
DROP TABLE IF EXISTS barcode_upload_history CASCADE;
CREATE TABLE barcode_upload_history (
    upload_id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    uploaded_by VARCHAR(100),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_rows INTEGER,
    matched_rows INTEGER,
    unmatched_rows INTEGER,
    status VARCHAR(20) DEFAULT 'completed', -- completed, failed, processing
    error_message TEXT
);

-- 3. Upload Details Table (stores each row from user upload)
-- User file has only ONE column: barcode
DROP TABLE IF EXISTS barcode_upload_details CASCADE;
CREATE TABLE barcode_upload_details (
    detail_id SERIAL PRIMARY KEY,
    upload_id INTEGER REFERENCES barcode_upload_history(upload_id) ON DELETE CASCADE,
    row_number INTEGER,
    barcode VARCHAR(50),
    item_code VARCHAR(50), -- Filled if matched, NULL if not
    match_status VARCHAR(20), -- 'matched', 'not_found'
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Index for querying by upload
    CONSTRAINT fk_upload FOREIGN KEY (upload_id) REFERENCES barcode_upload_history(upload_id)
);

CREATE INDEX idx_upload_details_upload_id ON barcode_upload_details(upload_id);
CREATE INDEX idx_upload_details_match_status ON barcode_upload_details(match_status);

-- 4. Unmatched Barcodes Queue (for investigation)
DROP TABLE IF EXISTS barcode_unmatched_queue CASCADE;
CREATE TABLE barcode_unmatched_queue (
    queue_id SERIAL PRIMARY KEY,
    barcode VARCHAR(50) NOT NULL,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    occurrence_count INTEGER DEFAULT 1,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_item_code VARCHAR(20),
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(100),
    notes TEXT,
    
    CONSTRAINT uq_unmatched_barcode UNIQUE (barcode)
);

CREATE INDEX idx_unmatched_barcode ON barcode_unmatched_queue(barcode) WHERE resolved = FALSE;

-- ============================================================
-- SEED DATA - Initial load (will be updated daily)
-- ============================================================

-- Note: Data will be loaded from Python script (load_barcode_master.py)
-- Source: D:\Dashboard Code\NO_WH\DS\barcode_matcher\sample_data\barcodedata.xlsx or .csv
-- Run: python batch\load_barcode_master.py (daily automated task)

-- ============================================================
-- HELPER VIEWS
-- ============================================================

-- View: Active barcode mappings
CREATE OR REPLACE VIEW v_active_barcodes AS
SELECT 
    barcode,
    item_code,
    item_name,
    created_at,
    updated_at
FROM barcode_item_master
WHERE is_active = TRUE
ORDER BY barcode;

-- View: Recent upload summary
CREATE OR REPLACE VIEW v_recent_uploads AS
SELECT 
    h.upload_id,
-- View: Active barcode mappings
CREATE OR REPLACE VIEW v_active_barcodes AS
SELECT 
    vc_item_barcode,
    vc_item_code,
    created_at,
    updated_at
FROM barcode_item_master
WHERE is_active = TRUE
ORDER BY vc_item_barcode;
-- View: Unmatched barcodes needing attention
CREATE OR REPLACE VIEW v_unmatched_barcodes_pending AS
SELECT 
    barcode,
    occurrence_count,
    first_seen,
    last_seen,
    EXTRACT(DAY FROM CURRENT_TIMESTAMP - first_seen) AS days_pending
FROM barcode_unmatched_queue
WHERE resolved = FALSE
ORDER BY occurrence_count DESC, last_seen DESC;

-- ============================================================
-- STORED FUNCTION: Process Uploaded Barcodes
-- ============================================================

CREATE OR REPLACE FUNCTION process_barcode_upload(
    p_upload_id INTEGER,
    p_barcodes TEXT[] -- Array of barcodes from uploaded file
)
RETURNS TABLE(
    row_num INTEGER,
    barcode VARCHAR(50),
    item_code VARCHAR(20),
    match_status VARCHAR(20)
) AS $$
BEGIN
    RETURN QUERY
    WITH uploaded_data AS (
        -- Unnest array with row numbers
        SELECT 
            ROW_NUMBER() OVER () AS row_num,
            TRIM(UPPER(barcode_val)) AS barcode
        FROM UNNEST(p_barcodes) AS barcode_val
    ),
    matched_data AS (
        -- Left join to get item codes (NULL if not found)
        SELECT 
            ud.row_num,
            ud.barcode,
            bim.item_code,
            CASE 
                WHEN bim.item_code IS NOT NULL THEN 'matched'
                ELSE 'not_found'
            END AS match_status
        FROM uploaded_data ud
        LEFT JOIN barcode_item_master bim 
            ON ud.barcode = bim.barcode 
            AND bim.is_active = TRUE
    )
    SELECT 
        md.row_num,
        md.barcode,
        md.item_code,
        md.match_status
    FROM matched_data md
    ORDER BY md.row_num;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- MAINTENANCE FUNCTIONS
-- ============================================================

-- Function: Add new barcode mapping
CREATE OR REPLACE FUNCTION add_barcode_mapping(
    p_barcode VARCHAR(50),
    p_item_code VARCHAR(20),
    p_item_name VARCHAR(255) DEFAULT NULL
)
RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO barcode_item_master (barcode, item_code, item_name)
    VALUES (TRIM(UPPER(p_barcode)), TRIM(UPPER(p_item_code)), p_item_name)
    ON CONFLICT (barcode) DO UPDATE
    SET item_code = EXCLUDED.item_code,
        item_name = EXCLUDED.item_name,
        updated_at = CURRENT_TIMESTAMP;
    
    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
-- Function: Add new barcode mapping (used by daily load script)
CREATE OR REPLACE FUNCTION add_barcode_mapping(
    p_barcode VARCHAR(50),
    p_item_code VARCHAR(50)
)
RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO barcode_item_master (vc_item_barcode, vc_item_code)
    VALUES (TRIM(p_barcode), TRIM(p_item_code))
    ON CONFLICT (vc_item_barcode) DO UPDATE
    SET vc_item_code = EXCLUDED.vc_item_code,
        updated_at = CURRENT_TIMESTAMP;
    
    RETURN TRUE;
EXCEPTION WHEN OTHERS THEN
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql; needed)
-- ============================================================

GRANT SELECT, INSERT, UPDATE ON barcode_item_master TO postgres;
GRANT SELECT, INSERT ON barcode_upload_history TO postgres;
GRANT SELECT, INSERT ON barcode_upload_details TO postgres;
GRANT SELECT, INSERT, UPDATE ON barcode_unmatched_queue TO postgres;
GRANT SELECT ON v_active_barcodes TO postgres;
GRANT SELECT ON v_recent_uploads TO postgres;
GRANT SELECT ON v_unmatched_barcodes_pending TO postgres;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Check master table count
SELECT COUNT(*) AS total_barcodes FROM barcode_item_master WHERE is_active = TRUE;

-- Sample data
SELECT * FROM barcode_item_master LIMIT 10;

-- Check for duplicate barcodes
SELECT barcode, COUNT(*) 
FROM barcode_item_master 
WHERE is_active = TRUE
GROUP BY barcode 
HAVING COUNT(*) > 1;
