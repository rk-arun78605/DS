import uuid
from django.db import models


class ProductScan(models.Model):
    scan_id      = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    scanned_at   = models.DateTimeField(auto_now_add=True)

    # ── Core identification ────────────────────────────────────────────────
    product_name = models.TextField(blank=True, null=True)
    brand        = models.TextField(blank=True, null=True)
    category     = models.CharField(max_length=120, blank=True, null=True)
    product_type = models.TextField(blank=True, null=True)
    description  = models.TextField(blank=True, null=True)
    is_fnv       = models.BooleanField(null=True, blank=True)   # True = fresh produce/vegetable/herb

    # ── Quantity & packaging ───────────────────────────────────────────────
    quantity             = models.DecimalField(max_digits=14, decimal_places=3, null=True, blank=True)
    unit_of_measurement  = models.CharField(max_length=60, blank=True, null=True)
    units_per_case       = models.IntegerField(null=True, blank=True)
    num_cases            = models.IntegerField(null=True, blank=True)
    num_pieces           = models.IntegerField(null=True, blank=True)
    total_units          = models.DecimalField(max_digits=14, decimal_places=3, null=True, blank=True)
    grammage             = models.CharField(max_length=80, blank=True, null=True)  # e.g. "500g", "250ml", "6x330ml"

    # ── Weight ────────────────────────────────────────────────────────────
    weight_per_unit = models.DecimalField(max_digits=12, decimal_places=3, null=True, blank=True)
    weight_unit     = models.CharField(max_length=20, blank=True, null=True)
    total_weight    = models.DecimalField(max_digits=14, decimal_places=3, null=True, blank=True)
    scale_reading   = models.CharField(max_length=60, blank=True, null=True)

    # ── Label details ─────────────────────────────────────────────────────
    has_expiry = models.BooleanField(null=True, blank=True)   # True if BB/expiry visible on pack

    # ── Alcohol specific ──────────────────────────────────────────────────
    is_alcohol          = models.BooleanField(default=False)
    alcohol_type        = models.CharField(max_length=100, blank=True, null=True)
    alcohol_subtype     = models.CharField(max_length=200, blank=True, null=True)
    bottle_size_ml      = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    fill_level_pct      = models.DecimalField(max_digits=5,  decimal_places=1, null=True, blank=True)
    ml_remaining        = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    ml_consumed         = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    alcohol_percentage  = models.DecimalField(max_digits=5,  decimal_places=2, null=True, blank=True)

    # ── Label details ─────────────────────────────────────────────────────
    barcode          = models.CharField(max_length=100, blank=True, null=True)
    expiry_date      = models.CharField(max_length=60,  blank=True, null=True)
    country_of_origin= models.CharField(max_length=100, blank=True, null=True)

    # ── AI metadata ───────────────────────────────────────────────────────
    confidence_score  = models.DecimalField(max_digits=4, decimal_places=3, null=True, blank=True)
    ai_notes          = models.TextField(blank=True, null=True)
    raw_ai_response   = models.JSONField(null=True, blank=True)
    model_used        = models.CharField(max_length=80, blank=True, null=True)

    # ── Session ───────────────────────────────────────────────────────────
    session_key  = models.CharField(max_length=100, blank=True, null=True)
    device_info  = models.TextField(blank=True, null=True)

    class Meta:
        db_table = "product_scans"
        ordering = ["-scanned_at"]

    def __str__(self):
        return f"{self.product_name or 'Unknown'} — {self.scanned_at:%d %b %Y %H:%M}"

    def to_dict(self):
        return {
            "id":              self.id,
            "scan_id":         str(self.scan_id),
            "scanned_at":      self.scanned_at.strftime("%d %b %Y %H:%M:%S"),
            "product_name":    self.product_name,
            "brand":           self.brand,
            "category":        self.category,
            "product_type":    self.product_type,
            "description":     self.description,
            "is_fnv":          self.is_fnv,
            "quantity":        float(self.quantity) if self.quantity is not None else None,
            "unit_of_measurement": self.unit_of_measurement,
            "units_per_case":  self.units_per_case,
            "num_cases":       self.num_cases,
            "num_pieces":      self.num_pieces,
            "total_units":     float(self.total_units) if self.total_units is not None else None,
            "grammage":        self.grammage,
            "weight_per_unit": float(self.weight_per_unit) if self.weight_per_unit is not None else None,
            "weight_unit":     self.weight_unit,
            "total_weight":    float(self.total_weight) if self.total_weight is not None else None,
            "scale_reading":   self.scale_reading,
            "has_expiry":      self.has_expiry,
            "expiry_date":     self.expiry_date,
            "barcode":         self.barcode,
            "country_of_origin": self.country_of_origin,
            "is_alcohol":      self.is_alcohol,
            "alcohol_type":    self.alcohol_type,
            "alcohol_subtype": self.alcohol_subtype,
            "bottle_size_ml":  float(self.bottle_size_ml) if self.bottle_size_ml is not None else None,
            "fill_level_pct":  float(self.fill_level_pct) if self.fill_level_pct is not None else None,
            "ml_remaining":    float(self.ml_remaining) if self.ml_remaining is not None else None,
            "ml_consumed":     float(self.ml_consumed) if self.ml_consumed is not None else None,
            "alcohol_percentage": float(self.alcohol_percentage) if self.alcohol_percentage is not None else None,
            "confidence_score": float(self.confidence_score) if self.confidence_score is not None else None,
            "ai_notes":        self.ai_notes,
            "model_used":      self.model_used,
        }
