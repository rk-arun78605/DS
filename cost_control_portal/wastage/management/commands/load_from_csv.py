"""
Management command: load_from_csv
===================================
Loads master data from CSV files into the database.

Usage:
    python manage.py load_from_csv --staff path/to/cc_shop_staff.csv
    python manage.py load_from_csv --items path/to/cc_item_master.csv
    python manage.py load_from_csv --staff cc_shop_staff.csv --items cc_item_master.csv
    python manage.py load_from_csv  # auto-finds files in data_templates/

Options:
    --clear    Clear existing records before loading (USE WITH CAUTION)
"""

import csv
import os
from pathlib import Path

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.db import transaction

from wastage.models import ShopStaff, ItemMaster


class Command(BaseCommand):
    help = 'Load staff and item master data from CSV files.'

    def add_arguments(self, parser):
        parser.add_argument('--staff', type=str, default=None, help='Path to cc_shop_staff.csv')
        parser.add_argument('--items', type=str, default=None, help='Path to cc_item_master.csv')
        parser.add_argument('--clear', action='store_true', help='Clear existing data before loading')

    def handle(self, *args, **options):
        base = Path(__file__).resolve().parents[4] / 'data_templates'

        staff_file = options['staff'] or str(base / 'cc_shop_staff.csv')
        items_file = options['items'] or str(base / 'cc_item_master.csv')

        if options['clear']:
            confirm = input('⚠️  This will DELETE all existing staff and items. Type YES to confirm: ')
            if confirm.strip() != 'YES':
                self.stdout.write('Aborted.')
                return
            ShopStaff.objects.all().delete()
            ItemMaster.objects.all().delete()
            self.stdout.write('✓ Cleared existing data.')

        if os.path.exists(staff_file):
            self._load_staff(staff_file)
        else:
            self.stderr.write(f'Staff file not found: {staff_file}')

        if os.path.exists(items_file):
            self._load_items(items_file)
        else:
            self.stderr.write(f'Items file not found: {items_file}')

    def _load_staff(self, filepath):
        self.stdout.write(f'\n📋 Loading staff from: {filepath}')
        created = updated = skipped = 0

        rows = []
        with open(filepath, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strip whitespace from all values
                rows.append({k.strip(): v.strip() for k, v in row.items()})

        # First pass: create all staff records (without approver links)
        with transaction.atomic():
            for row in rows:
                emp_id = row.get('emp_id', '').strip()
                if not emp_id:
                    self.stderr.write(f'  ⚠ Skipping row with empty emp_id')
                    skipped += 1
                    continue

                emp_name = row.get('emp_name', '').strip()
                shop_code = row.get('shop_code', '').strip().upper()
                shop_name = row.get('shop_name', '').strip()
                department = row.get('department', '').strip() or None
                role = row.get('role', 'staff').strip().lower()
                managed_shops_raw = row.get('managed_shops', '').strip()
                # Accept both comma and semicolon as separators
                managed_shops = ','.join(
                    s.strip().upper() for s in managed_shops_raw.replace(';', ',').split(',') if s.strip()
                ) or None
                password = row.get('password', 'melcom123').strip()
                is_active_raw = row.get('is_active', 'True').strip().lower()
                is_active = is_active_raw in ('true', '1', 'yes')

                if role not in ('staff', 'supervisor', 'manager'):
                    self.stderr.write(f'  ⚠ {emp_id}: invalid role "{role}", defaulting to "staff"')
                    role = 'staff'

                # Create or update Django User (username = lowercase emp_id)
                username = emp_id.lower()
                user, user_created = User.objects.get_or_create(username=username)
                if user_created or password:
                    user.set_password(password)
                    first, *rest = emp_name.split(' ', 1)
                    user.first_name = first
                    user.last_name = rest[0] if rest else ''
                    user.save()

                # Create or update ShopStaff
                staff_obj, s_created = ShopStaff.objects.update_or_create(
                    emp_id=emp_id,
                    defaults={
                        'emp_name': emp_name,
                        'shop_code': shop_code,
                        'shop_name': shop_name,
                        'department': department,
                        'role': role,
                        'managed_shops': managed_shops,
                        'is_active': is_active,
                        'user': user,
                    }
                )

                if s_created:
                    created += 1
                    status = '✓ Created'
                else:
                    updated += 1
                    status = '↻ Updated'

                self.stdout.write(
                    f'  {status}: {emp_name} ({emp_id}) — {role} @ {shop_code}'
                    + (f'  [manages: {managed_shops}]' if managed_shops else '')
                )

        # Second pass: link approvers (after all staff exist)
        with transaction.atomic():
            for row in rows:
                emp_id = row.get('emp_id', '').strip()
                approver_id = row.get('approver_emp_id', '').strip()
                if emp_id and approver_id:
                    try:
                        staff_obj = ShopStaff.objects.get(emp_id=emp_id)
                        approver_obj = ShopStaff.objects.get(emp_id=approver_id)
                        staff_obj.approver = approver_obj
                        staff_obj.save(update_fields=['approver'])
                    except ShopStaff.DoesNotExist:
                        self.stderr.write(
                            f'  ⚠ Approver {approver_id} not found for {emp_id} — skipped'
                        )

        self.stdout.write(
            f'\n✅ Staff: {created} created, {updated} updated, {skipped} skipped'
        )

    def _load_items(self, filepath):
        self.stdout.write(f'\n📦 Loading items from: {filepath}')
        created = updated = skipped = 0

        with open(filepath, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            with transaction.atomic():
                for row in reader:
                    row = {k.strip(): v.strip() for k, v in row.items()}
                    item_code = row.get('item_code', '').strip()
                    if not item_code:
                        self.stderr.write('  ⚠ Skipping row with empty item_code')
                        skipped += 1
                        continue

                    item_name = row.get('item_name', '').strip()
                    if not item_name:
                        self.stderr.write(f'  ⚠ {item_code}: empty item_name — skipped')
                        skipped += 1
                        continue

                    category = row.get('category', 'Finished').strip()
                    if category not in ('RAW', 'Finished', 'Packaging'):
                        self.stderr.write(
                            f'  ⚠ {item_code}: invalid category "{category}", defaulting to "Finished"'
                        )
                        category = 'Finished'

                    is_active_raw = row.get('is_active', 'True').strip().lower()
                    is_active = is_active_raw in ('true', '1', 'yes')

                    def safe_decimal(val, default='0'):
                        try:
                            return float(val) if val else float(default)
                        except ValueError:
                            return float(default)

                    _, item_created = ItemMaster.objects.update_or_create(
                        item_code=item_code,
                        defaults={
                            'item_name': item_name,
                            'department': row.get('department', '').strip() or None,
                            'grp': row.get('grp', '').strip() or None,
                            'sub_group': row.get('sub_group', '').strip() or None,
                            'uom': row.get('uom', '').strip() or None,
                            'category': category,
                            'cost_price': safe_decimal(row.get('cost_price')),
                            'selling_price': safe_decimal(row.get('selling_price')),
                            'is_active': is_active,
                        }
                    )

                    if item_created:
                        created += 1
                        self.stdout.write(f'  ✓ Created: [{item_code}] {item_name}')
                    else:
                        updated += 1
                        self.stdout.write(f'  ↻ Updated: [{item_code}] {item_name}')

        self.stdout.write(
            f'\n✅ Items: {created} created, {updated} updated, {skipped} skipped'
        )
