"""
Migration 0006: Cost Control Feature Changes
  - Incident: add approved_quantity, expiry_date, expiry_photo_path, expiry_photo_hash, supervisor_remark
  - Incident: update STATUS_CHOICES (add Draft), REASON_CHOICES (add 'Received wrong')
  - ItemMaster: update CATEGORY_CHOICES (add WIP, Semifinished) — choices only, no DB change
  - New model: ShopLocation (for geofencing, Change 10)
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('wastage', '0005_itemmaster_net_selling_price_itemmaster_store_name_and_more'),
    ]

    operations = [
        # ── Incident new fields ───────────────────────────────────────────────
        migrations.AddField(
            model_name='incident',
            name='approved_quantity',
            field=models.DecimalField(
                blank=True, decimal_places=3, max_digits=12, null=True,
                verbose_name='Approved Quantity',
            ),
        ),
        migrations.AddField(
            model_name='incident',
            name='expiry_date',
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='incident',
            name='expiry_photo_path',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='incident',
            name='expiry_photo_hash',
            field=models.CharField(blank=True, max_length=64, null=True),
        ),
        migrations.AddField(
            model_name='incident',
            name='supervisor_remark',
            field=models.TextField(blank=True, null=True),
        ),
        # ── Choices-only AlterField (no DDL change, just metadata) ────────────
        migrations.AlterField(
            model_name='incident',
            name='status',
            field=models.CharField(
                choices=[
                    ('Pending', 'Pending'),
                    ('Approved', 'Approved'),
                    ('Rejected', 'Rejected'),
                    ('Draft', 'Draft'),
                ],
                default='Pending',
                max_length=20,
            ),
        ),
        migrations.AlterField(
            model_name='incident',
            name='reason',
            field=models.CharField(
                choices=[
                    ('Expired', 'Expired'),
                    ('Over Production', 'Over Production'),
                    ('Damage', 'Damage'),
                    ('Quality Issue', 'Quality Issue'),
                    ('Handling Error', 'Handling Error'),
                    ('Received wrong', 'Received wrong'),
                    ('No Wastage', 'No Wastage'),
                    ('Others', 'Others'),
                ],
                max_length=30,
            ),
        ),
        migrations.AlterField(
            model_name='itemmaster',
            name='category',
            field=models.CharField(
                choices=[
                    ('RAW', 'RAW'),
                    ('Finished', 'Finished'),
                    ('Packaging', 'Packaging'),
                    ('WIP', 'WIP / Work In Progress'),
                    ('Semifinished', 'Semifinished'),
                ],
                default='Finished',
                max_length=20,
            ),
        ),
        # ── New model: ShopLocation ───────────────────────────────────────────
        migrations.CreateModel(
            name='ShopLocation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('shop_code', models.CharField(max_length=20, unique=True)),
                ('shop_name', models.CharField(max_length=100)),
                ('latitude', models.DecimalField(decimal_places=7, max_digits=10)),
                ('longitude', models.DecimalField(decimal_places=7, max_digits=10)),
                ('radius_meters', models.IntegerField(default=200)),
                ('is_active', models.BooleanField(default=True)),
            ],
            options={
                'db_table': 'cc_shop_location',
            },
        ),
    ]
