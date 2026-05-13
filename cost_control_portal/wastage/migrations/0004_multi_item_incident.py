from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('wastage', '0003_add_media_fingerprint_fields'),
    ]

    operations = [
        # Remove unique constraint — one incident_no can now span multiple item rows
        migrations.AlterField(
            model_name='incident',
            name='incident_no',
            field=models.CharField(db_index=True, max_length=30),
        ),
        # Add composite performance indexes
        migrations.AddIndex(
            model_name='incident',
            index=models.Index(fields=['incident_no'], name='idx_incident_no'),
        ),
        migrations.AddIndex(
            model_name='incident',
            index=models.Index(fields=['shop_code', 'submit_date'], name='idx_incident_shop_date'),
        ),
        migrations.AddIndex(
            model_name='incident',
            index=models.Index(fields=['submitted_by', 'submit_date'], name='idx_incident_staff_date'),
        ),
        migrations.AddIndex(
            model_name='incident',
            index=models.Index(fields=['status'], name='idx_incident_status'),
        ),
    ]
