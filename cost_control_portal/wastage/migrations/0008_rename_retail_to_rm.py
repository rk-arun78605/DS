from django.db import migrations, models


def rename_retail_to_rm(apps, schema_editor):
    Incident = apps.get_model('wastage', 'Incident')
    Incident.objects.filter(department='Retail').update(department='RM')

    ItemMaster = apps.get_model('wastage', 'ItemMaster')
    ItemMaster.objects.filter(department='Retail').update(department='RM')

    ShopStaff = apps.get_model('wastage', 'ShopStaff')
    ShopStaff.objects.filter(department='Retail').update(department='RM')


class Migration(migrations.Migration):

    dependencies = [
        ('wastage', '0007_incident_item_photo_hash_incident_item_photo_path_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='incident',
            name='department',
            field=models.CharField(
                choices=[('Bakery', 'Bakery'), ('Kitchen', 'Kitchen'), ('RM', 'RM')],
                max_length=50,
            ),
        ),
        migrations.RunPython(rename_retail_to_rm, migrations.RunPython.noop),
    ]
