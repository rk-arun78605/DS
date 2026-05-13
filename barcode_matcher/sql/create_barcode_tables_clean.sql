-- ============================================================
-- BARCODE MATCHING SYSTEM - DATABASE ARCHITECTURE
-- Created: December 13, 2025
-- Daily Upload: barcodedata.xlsx from sample_data folder
-- ============================================================

-- 1. Master Barcode-Item Mapping Table
-- Columns match source file: VC_ITEM_BARCODE, VC_ITEM_CODE
DROP TABLE IF EXISTS barcode_item_master CASCADE;
CREATE TABLE barcode_item_master (
    id SERIAL PRIMARY KEY,
    vc_item_barcode VARCHAR(50) NOT NULL,
    vc_item_code VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT uq_barcode UNIQUE (vc_item_barcode)
);

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
    status VARCHAR(20) DEFAULT 'completed',
    error_message TEXT
);

-- 3. Upload Details Table (stores each row from user upload)
DROP TABLE IF EXISTS barcode_upload_details CASCADE;
CREATE TABLE barcode_upload_details (
    detail_id SERIAL PRIMARY KEY,
    upload_id INTEGER REFERENCES barcode_upload_history(upload_id) ON DELETE CASCADE,
    row_number INTEGER,
    barcode VARCHAR(50),
    item_code VARCHAR(50),
    match_status VARCHAR(20),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
-- VIEWS
-- ============================================================

CREATE OR REPLACE VIEW v_active_barcodes AS
SELECT 
    vc_item_barcode,
    vc_item_code,
    created_at,
    updated_at
FROM barcode_item_master
WHERE is_active = TRUE
ORDER BY vc_item_barcode;

CREATE OR REPLACE VIEW v_recent_uploads AS
SELECT 
    h.upload_id,
    h.filename,
    h.uploaded_by,
    h.uploaded_at,
    h.total_rows,
    h.matched_rows,
    h.unmatched_rows,
    ROUND(100.0 * h.matched_rows / NULLIF(h.total_rows, 0), 2) AS match_rate_pct,
    h.status
FROM barcode_upload_history h
ORDER BY h.uploaded_at DESC;

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
-- FUNCTIONS
-- ============================================================

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
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION track_unmatched_barcode(
    p_barcode VARCHAR(50)
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO barcode_unmatched_queue (barcode, occurrence_count)
    VALUES (TRIM(p_barcode), 1)
    ON CONFLICT (barcode) DO UPDATE
    SET occurrence_count = barcode_unmatched_queue.occurrence_count + 1,
        last_seen = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- GRANTS
-- ============================================================

GRANT SELECT, INSERT, UPDATE ON barcode_item_master TO postgres;
GRANT SELECT, INSERT ON barcode_upload_history TO postgres;
GRANT SELECT, INSERT ON barcode_upload_details TO postgres;
GRANT SELECT, INSERT, UPDATE ON barcode_unmatched_queue TO postgres;
GRANT SELECT ON v_active_barcodes TO postgres;
GRANT SELECT ON v_recent_uploads TO postgres;
GRANT SELECT ON v_unmatched_barcodes_pending TO postgres;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO postgres;
