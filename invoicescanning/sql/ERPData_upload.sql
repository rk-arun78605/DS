ALTER TABLE public.erpdata DISABLE TRIGGER ALL;

COPY ERPdata (
    invno, 
    store_code, 
    amt, 
    invdate, 
    entry_time, 
    tillno, 
    cashier
)
FROM 'C:\MIS-ARUN\shopbillcount_20260509.csv'
WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

-- 3. Re-enable the trigger
ALTER TABLE public.erpdata ENABLE TRIGGER ALL;
