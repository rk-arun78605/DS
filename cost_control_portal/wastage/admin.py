from django.contrib import admin
from .models import ShopStaff, ItemMaster, Incident, MediaRegistry


@admin.register(ShopStaff)
class ShopStaffAdmin(admin.ModelAdmin):
    list_display = ('emp_id', 'emp_name', 'shop_code', 'shop_name', 'role', 'is_active')
    list_filter = ('role', 'shop_code', 'is_active')
    search_fields = ('emp_id', 'emp_name', 'shop_code')


@admin.register(ItemMaster)
class ItemMasterAdmin(admin.ModelAdmin):
    list_display = ('item_code', 'item_name', 'category', 'department', 'selling_price', 'is_active')
    list_filter = ('category', 'department', 'is_active')
    search_fields = ('item_code', 'item_name')


@admin.register(Incident)
class IncidentAdmin(admin.ModelAdmin):
    list_display = ('incident_no', 'shop_code', 'department', 'submitted_by', 'item_name', 'total_value', 'status', 'submit_date')
    list_filter = ('status', 'shop_code', 'department', 'reason')
    search_fields = ('incident_no', 'submitted_by', 'item_name')
    readonly_fields = ('incident_no', 'submit_date', 'total_value', 'created_at', 'updated_at')


@admin.register(MediaRegistry)
class MediaRegistryAdmin(admin.ModelAdmin):
    list_display = ('incident_no', 'media_type', 'file_hash', 'uploaded_at')
    list_filter = ('media_type',)
    search_fields = ('incident_no', 'file_hash')

