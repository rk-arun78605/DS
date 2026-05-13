-- Create whstock table in century_penetration database
-- This table stores warehouse stock data for Century items

CREATE TABLE IF NOT EXISTS whstock (
    id SERIAL PRIMARY KEY,
    vc_item_code VARCHAR(50) NOT NULL,
    wh_code VARCHAR(20) NOT NULL,
    wh_name VARCHAR(100),
    balance_qty NUMERIC(15, 2) DEFAULT 0,
    upload_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_whstock_item_code ON whstock(vc_item_code);
CREATE INDEX IF NOT EXISTS idx_whstock_wh_code ON whstock(wh_code);
CREATE INDEX IF NOT EXISTS idx_whstock_upload_date ON whstock(upload_date);
CREATE INDEX IF NOT EXISTS idx_whstock_item_wh ON whstock(vc_item_code, wh_code);
CREATE INDEX IF NOT EXISTS idx_whstock_item_upload ON whstock(vc_item_code, upload_date DESC);

-- Add comment
COMMENT ON TABLE whstock IS 'Warehouse stock data for Century items - synced from salesdata.whstock';
COMMENT ON COLUMN whstock.vc_item_code IS 'Item code (VC product code)';
COMMENT ON COLUMN whstock.wh_code IS 'Warehouse code';
COMMENT ON COLUMN whstock.wh_name IS 'Warehouse name';
COMMENT ON COLUMN whstock.balance_qty IS 'Current warehouse stock quantity';
COMMENT ON COLUMN whstock.upload_date IS 'Date when stock data was uploaded';
