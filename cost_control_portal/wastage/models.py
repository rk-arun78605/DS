from django.db import models
from django.contrib.auth.models import User


class ShopStaff(models.Model):
    ROLE_CHOICES = [
        ('staff',      'Staff'),
        ('supervisor', 'Supervisor / Manager'),
        ('management', 'Management'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    emp_id = models.CharField(max_length=20, unique=True)
    emp_name = models.CharField(max_length=100)
    shop_code = models.CharField(max_length=20)
    shop_name = models.CharField(max_length=100)
    department = models.CharField(max_length=50, blank=True, null=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='staff')
    approver = models.ForeignKey(
        'self', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='subordinates', to_field='emp_id'
    )
    # Comma-separated list of ALL shop codes this supervisor oversees, e.g. "SPN,MSS,KAS"
    managed_shops = models.CharField(max_length=500, blank=True, null=True)
    approver_name = models.CharField(max_length=200, blank=True, null=True)
    managed_shop_name = models.CharField(max_length=200, blank=True, null=True)
    shop_start_time = models.CharField(max_length=20, blank=True, null=True)
    shop_close_time = models.CharField(max_length=20, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'cc_shop_staff'
        verbose_name = 'Shop Staff'
        verbose_name_plural = 'Shop Staff'

    def __str__(self):
        return f"{self.emp_name} ({self.emp_id}) - {self.shop_name}"

    @property
    def is_approver(self):
        return self.role in ('supervisor', 'manager')


class ItemMaster(models.Model):
    CATEGORY_CHOICES = [
        ('RAW', 'RAW'),
        ('Finished', 'Finished'),
        ('Packaging', 'Packaging'),
        ('WIP', 'WIP / Work In Progress'),
        ('Semifinished', 'Semifinished'),
    ]
    item_code = models.CharField(max_length=200, unique=True, default='AAAAA')
    item_name = models.CharField(max_length=200)
    department = models.CharField(max_length=50, blank=True, null=True)
    grp = models.CharField(max_length=100, blank=True, null=True)
    sub_group = models.CharField(max_length=100, blank=True, null=True)
    uom = models.CharField(max_length=20, blank=True, null=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default='Finished')
    cost_price = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    selling_price = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    is_active = models.BooleanField(default=True)
    store_name = models.CharField(max_length=200, blank=True, null=True)
    net_selling_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'cc_item_master'
        verbose_name = 'Item Master'
        verbose_name_plural = 'Item Master'

    def __str__(self):
        return f"{self.item_code} - {self.item_name}"


class Incident(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Approved', 'Approved'),
        ('Rejected', 'Rejected'),
        ('Draft', 'Draft'),
    ]
    REASON_CHOICES = [
        ('Expired', 'Expired'),
        ('Over Production', 'Over Production'),
        ('Damage', 'Damage'),
        ('Quality Issue', 'Quality Issue'),
        ('Handling Error', 'Handling Error'),
        ('Received wrong', 'Received wrong'),
        ('No Wastage', 'No Wastage'),
        ('Others', 'Others'),
    ]
    DEPARTMENT_CHOICES = [
        ('Bakery', 'Bakery'),
        ('Kitchen', 'Kitchen'),
        ('RM', 'RM'),
    ]

    incident_no = models.CharField(max_length=30, db_index=True)  # Non-unique: multiple items per incident
    shop_code = models.CharField(max_length=20)
    shop_name = models.CharField(max_length=100)
    department = models.CharField(max_length=50, choices=DEPARTMENT_CHOICES)
    submitted_by = models.CharField(max_length=20)
    submitted_name = models.CharField(max_length=100, blank=True)
    item_code = models.CharField(max_length=20, default='AAAAA')
    item_name = models.CharField(max_length=200)
    category = models.CharField(max_length=20)
    quantity = models.DecimalField(max_digits=12, decimal_places=3)
    uom = models.CharField(max_length=20, blank=True, null=True)
    selling_price = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total_value = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    reason = models.CharField(max_length=30, choices=REASON_CHOICES)
    remarks = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    submit_date = models.DateTimeField(auto_now_add=True)
    approved_by = models.CharField(max_length=20, blank=True, null=True)
    approved_name = models.CharField(max_length=100, blank=True, null=True)
    approved_date = models.DateTimeField(null=True, blank=True)
    photo_path = models.TextField(blank=True, null=True)
    video_path = models.TextField(blank=True, null=True)
    photo_hash = models.CharField(max_length=64, blank=True, null=True)
    video_hash = models.CharField(max_length=64, blank=True, null=True)
    photo_exif_sig = models.CharField(max_length=64, blank=True, null=True)
    photo_phash = models.CharField(max_length=64, blank=True, null=True)
    video_meta_sig = models.CharField(max_length=64, blank=True, null=True)
    is_late_submission = models.BooleanField(default=False)
    is_system_generated = models.BooleanField(default=False)  # True for auto-created 3AM incidents
    # Change 3: approved quantity (supervisor can adjust)
    approved_quantity = models.DecimalField(max_digits=12, decimal_places=3, null=True, blank=True)
    # Change 4: expiry date + expiry photo for Expired reason
    expiry_date = models.DateField(null=True, blank=True)
    expiry_photo_path = models.TextField(null=True, blank=True)
    expiry_photo_hash = models.CharField(max_length=64, null=True, blank=True)
    # Change 7: per-item supervisor remark
    supervisor_remark = models.TextField(null=True, blank=True)
    # Per-item evidence photo (one per item row)
    item_photo_path  = models.TextField(null=True, blank=True)
    item_photo_hash  = models.CharField(max_length=64, null=True, blank=True)
    item_photo_phash = models.CharField(max_length=64, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'cc_incident'
        verbose_name = 'Wastage Incident'
        verbose_name_plural = 'Wastage Incidents'
        ordering = ['-submit_date']
        indexes = [
            models.Index(fields=['incident_no'], name='idx_incident_no'),
            models.Index(fields=['shop_code', 'submit_date'], name='idx_incident_shop_date'),
            models.Index(fields=['submitted_by', 'submit_date'], name='idx_incident_staff_date'),
            models.Index(fields=['status'], name='idx_incident_status'),
        ]

    def __str__(self):
        return f"{self.incident_no} - {self.shop_code} ({self.status})"

    def save(self, *args, **kwargs):
        self.total_value = self.quantity * self.selling_price
        super().save(*args, **kwargs)


class ShopLocation(models.Model):
    """Change 10: Geofencing — store location for each shop."""
    shop_code = models.CharField(max_length=20, unique=True)
    shop_name = models.CharField(max_length=100)
    latitude = models.DecimalField(max_digits=10, decimal_places=7)
    longitude = models.DecimalField(max_digits=10, decimal_places=7)
    radius_meters = models.IntegerField(default=200)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = 'cc_shop_location'

    def __str__(self):
        return f"{self.shop_code} — {self.shop_name}"


class MediaRegistry(models.Model):
    MEDIA_TYPE_CHOICES = [('photo', 'Photo'), ('video', 'Video')]
    incident_no = models.CharField(max_length=30)
    media_type = models.CharField(max_length=10, choices=MEDIA_TYPE_CHOICES)
    # Layer 1: exact byte-for-byte match (SHA-256)
    file_hash = models.CharField(max_length=64)
    # Layer 2: EXIF/metadata fingerprint — hash of DateTimeOriginal+GPS+camera model
    # Same photo re-saved/compressed will still match if EXIF is preserved
    exif_signature = models.CharField(max_length=64, blank=True, null=True)
    # Layer 3: perceptual hash (pHash) — visual similarity, 64-char hex
    # Catches resized, cropped, brightness-adjusted versions of the same image
    perceptual_hash = models.CharField(max_length=64, blank=True, null=True)
    file_path = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'cc_media_registry'
        verbose_name = 'Media Registry'

    def __str__(self):
        return f"{self.incident_no} - {self.media_type}"

