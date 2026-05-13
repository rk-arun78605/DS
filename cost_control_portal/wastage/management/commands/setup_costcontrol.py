"""
Management command to create the Cost Control Portal database, tables,
and initial sample users.

Usage:
    python manage.py setup_costcontrol
"""

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from wastage.models import ShopStaff, ItemMaster


class Command(BaseCommand):
    help = 'Set up initial data for Cost Control Portal'

    def handle(self, *args, **options):
        self.stdout.write('\n🔧 Setting up Cost Control Portal...\n')
        self._create_staff()
        self._create_items()
        self.stdout.write(self.style.SUCCESS('\n✅ Setup complete! Use the credentials above to log in.\n'))

    def _create_staff(self):
        self.stdout.write('👥 Creating staff accounts...')

        staff_data = [
            {
                'emp_id': 'EMP001', 'emp_name': 'Kofi Mensah',
                'shop_code': 'SPN', 'shop_name': 'Spintex Branch',
                'department': 'Kitchen', 'role': 'staff',
                'password': 'melcom123',
            },
            {
                'emp_id': 'EMP002', 'emp_name': 'Ama Asante',
                'shop_code': 'SPN', 'shop_name': 'Spintex Branch',
                'department': 'Bakery', 'role': 'staff',
                'password': 'melcom123',
            },
            {
                'emp_id': 'EMP003', 'emp_name': 'Kwame Boateng',
                'shop_code': 'MSS', 'shop_name': 'Maamobi Branch',
                'department': 'Kitchen', 'role': 'staff',
                'password': 'melcom123',
            },
            {
                'emp_id': 'SUP001', 'emp_name': 'Eric Owusu',
                'shop_code': 'SPN', 'shop_name': 'Spintex Branch',
                'department': None, 'role': 'supervisor',
                'password': 'melcom123',
            },
            {
                'emp_id': 'SUP002', 'emp_name': 'Grace Amoah',
                'shop_code': 'MSS', 'shop_name': 'Maamobi Branch',
                'department': None, 'role': 'supervisor',
                'password': 'melcom123',
            },
            {
                'emp_id': 'MGR001', 'emp_name': 'James Acheampong',
                'shop_code': 'SPN', 'shop_name': 'Spintex Branch',
                'department': None, 'role': 'manager',
                'password': 'melcom123',
            },
        ]

        approver_map = {
            'EMP001': 'SUP001',
            'EMP002': 'SUP001',
            'EMP003': 'SUP002',
        }

        created_staff = {}

        for data in staff_data:
            emp_id = data['emp_id']
            username = emp_id.lower()

            # Create or update Django user
            user, user_created = User.objects.get_or_create(username=username)
            user.set_password(data['password'])
            user.first_name = data['emp_name'].split()[0]
            user.last_name = ' '.join(data['emp_name'].split()[1:])
            user.save()

            # Create or update ShopStaff
            staff, staff_created = ShopStaff.objects.update_or_create(
                emp_id=emp_id,
                defaults={
                    'user': user,
                    'emp_name': data['emp_name'],
                    'shop_code': data['shop_code'],
                    'shop_name': data['shop_name'],
                    'department': data.get('department'),
                    'role': data['role'],
                    'is_active': True,
                }
            )
            created_staff[emp_id] = staff

            status = '✓ Created' if staff_created else '↺ Updated'
            self.stdout.write(
                f"  {status}: {data['emp_name']} ({emp_id}) — {data['role']} @ {data['shop_code']} "
                f"[login: {username} / {data['password']}]"
            )

        # Link approvers
        for emp_id, approver_id in approver_map.items():
            if emp_id in created_staff and approver_id in created_staff:
                created_staff[emp_id].approver = created_staff[approver_id]
                created_staff[emp_id].save()

    def _create_items(self):
        self.stdout.write('\n📦 Creating item master...')

        items = [
            ('BKR001', 'White Bread Loaf',       'Bakery',  'Bread',    'Sliced Bread',  'PCS', 'Finished',   3.50,  8.00),
            ('BKR002', 'Chocolate Cake (Whole)',  'Bakery',  'Cakes',    'Layer Cakes',   'PCS', 'Finished',  12.00, 35.00),
            ('BKR003', 'Croissant',               'Bakery',  'Pastry',   'Croissant',     'PCS', 'Finished',   2.00,  6.00),
            ('BKR004', 'Baguette',                'Bakery',  'Bread',    'French Bread',  'PCS', 'Finished',   3.00,  7.50),
            ('KIT001', 'Jollof Rice (Large)',     'Kitchen', 'Rice',     'Jollof',        'PCS', 'Finished',   8.00, 25.00),
            ('KIT002', 'Grilled Chicken Breast',  'Kitchen', 'Protein',  'Chicken',       'PCS', 'Finished',  15.00, 45.00),
            ('KIT003', 'Fried Plantain (Portion)','Kitchen', 'Sides',    'Plantain',      'PCS', 'Finished',   2.00,  8.00),
            ('KIT004', 'Chicken Soup (Pot)',      'Kitchen', 'Soups',    'Soups',         'POT', 'Finished',  40.00, 95.00),
            ('RTL001', 'Bottled Water 500ml',     'RM',      'Beverages','Water',         'PCS', 'Finished',   1.20,  3.00),
            ('RTL002', 'Orange Juice 1L',         'RM',      'Beverages','Juice',         'BTL', 'Finished',   5.00, 12.00),
            ('RAW001', 'All Purpose Flour (1KG)', 'Bakery',  'Raw Mat',  'Flour',         'KG',  'RAW',        2.50,  0.00),
            ('RAW002', 'Cooking Oil 5L',          'Kitchen', 'Raw Mat',  'Oils',          'BTL', 'RAW',       35.00,  0.00),
            ('RAW003', 'Sugar (1KG)',              'Bakery',  'Raw Mat',  'Sugar',         'KG',  'RAW',        3.00,  0.00),
            ('RAW004', 'Fresh Eggs (Tray)',        'Kitchen', 'Raw Mat',  'Eggs',          'TRY', 'RAW',       28.00,  0.00),
            ('PKG001', 'Cake Box Large',           'Bakery',  'Packaging','Boxes',         'PCS', 'Packaging',  1.20,  0.00),
            ('PKG002', 'Clear Food Wrap Roll',     'Kitchen', 'Packaging','Wrap',          'RLL', 'Packaging',  4.00,  0.00),
            ('AAAAA',  'Miscellaneous Item',       None,      None,       None,            'PCS', 'Finished',   0.00,  0.00),
        ]

        for row in items:
            item_code, item_name, dept, grp, sub, uom, cat, cost, sell = row
            obj, created = ItemMaster.objects.update_or_create(
                item_code=item_code,
                defaults={
                    'item_name': item_name,
                    'department': dept,
                    'grp': grp,
                    'sub_group': sub,
                    'uom': uom,
                    'category': cat,
                    'cost_price': cost,
                    'selling_price': sell,
                    'is_active': True,
                }
            )
            status = '✓ Created' if created else '↺ Updated'
            self.stdout.write(f"  {status}: [{item_code}] {item_name}")
