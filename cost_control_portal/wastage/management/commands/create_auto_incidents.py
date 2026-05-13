"""
Management command: create_auto_incidents
=========================================
Scheduled to run at 3:00 AM daily.

For every active staff member (role=staff) in each shop:
  - If they submitted NO incident on YESTERDAY's date → create a
    system-generated "No Wastage" incident so supervisors can see and approve it.

Auto-incident identifiers:
  - submitted_by  : 'AUTO001'
  - submitted_name: 'System Generated'
  - incident_no   : SYS-{SHOP}-{YYYYMMDD}-{EMPID}

Usage:
    python manage.py create_auto_incidents          # process yesterday
    python manage.py create_auto_incidents --date 2026-04-20   # specific date
"""

from datetime import date, timedelta, datetime, time as dtime

from django.core.management.base import BaseCommand
from django.utils import timezone

from wastage.models import Incident, ShopStaff


class Command(BaseCommand):
    help = 'Auto-create No Wastage incidents for staff who did not submit yesterday.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--date',
            type=str,
            default=None,
            help='Date to check (YYYY-MM-DD). Defaults to yesterday.',
        )

    def handle(self, *args, **options):
        date_str = options.get('date')
        if date_str:
            try:
                check_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                self.stderr.write(f'Invalid date: {date_str}. Use YYYY-MM-DD.')
                return
        else:
            check_date = date.today() - timedelta(days=1)

        self.stdout.write(f'[AutoIncident] Checking for date: {check_date}')

        # Get all active non-approver staff
        all_staff = ShopStaff.objects.filter(
            role='staff',
            is_active=True,
        ).select_related('user')

        created_count = 0
        skipped_count = 0

        for staff in all_staff:
            # Check if this staff submitted any incident on check_date
            submitted = Incident.objects.filter(
                submitted_by=staff.emp_id,
                submit_date__date=check_date,
            ).exclude(is_system_generated=True).exists()

            if submitted:
                skipped_count += 1
                continue  # Already submitted — nothing to do

            # Check if we already created a system incident for this staff + date
            already_exists = Incident.objects.filter(
                submitted_by='AUTO001',
                is_system_generated=True,
                shop_code=staff.shop_code,
                submit_date__date=check_date,
                remarks__icontains=staff.emp_id,
            ).exists()

            if already_exists:
                skipped_count += 1
                continue  # Don't double-create

            # Build incident number: SYS-{SHOP}-{YYYYMMDD}-{EMPID}
            incident_no = f"SYS-{staff.shop_code.upper()}-{check_date.strftime('%Y%m%d')}-{staff.emp_id}"
            # Ensure uniqueness
            suffix = 1
            base_no = incident_no
            while Incident.objects.filter(incident_no=incident_no).exists():
                incident_no = f"{base_no}-{suffix}"
                suffix += 1

            incident = Incident.objects.create(
                incident_no=incident_no,
                shop_code=staff.shop_code,
                shop_name=staff.shop_name,
                department=staff.department or 'Kitchen',
                submitted_by='AUTO001',
                submitted_name='System Generated',
                item_code='AAAAA',
                item_name='No Wastage Recorded',
                category='Finished',
                quantity=0,
                uom='',
                selling_price=0,
                total_value=0,
                reason='No Wastage',
                remarks=(
                    f'Auto-generated: No wastage incident submitted by '
                    f'{staff.emp_name} ({staff.emp_id}) for {check_date}.'
                ),
                status='Pending',
                is_system_generated=True,
            )

            # Backdate submit_date to the end of check_date (23:59:59)
            backdated = datetime.combine(check_date, dtime(23, 59, 59))
            Incident.objects.filter(pk=incident.pk).update(submit_date=backdated)

            created_count += 1
            self.stdout.write(
                f'  ✓ Created SYS incident {incident_no} for '
                f'{staff.emp_name} ({staff.emp_id}) @ {staff.shop_code}'
            )

        self.stdout.write(
            f'[AutoIncident] Done — Created: {created_count}, Skipped: {skipped_count}'
        )
