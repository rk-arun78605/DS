-- For non-null/non-empty fullqrcode:
-- min(id) per fullqrcode → duplicate=1 (original)
-- all others              → duplicate=0 (duplicate)
-- NULL/empty fullqrcode stays duplicate=1 (unchanged, matches trigger)

WITH ranked AS (
  SELECT
    id,
    CASE
      WHEN fullqrcode IS NULL OR TRIM(fullqrcode) = '' THEN 1
      WHEN ROW_NUMBER() OVER (PARTITION BY fullqrcode ORDER BY id) = 1 THEN 1
      ELSE 0
    END AS correct_duplicate
  FROM public.invoices
)
UPDATE public.invoices i
SET duplicate = r.correct_duplicate
FROM ranked r
WHERE i.id = r.id
  AND i.duplicate IS DISTINCT FROM r.correct_duplicate;