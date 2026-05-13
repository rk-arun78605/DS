from django.apps import AppConfig


class WastageConfig(AppConfig):
    name = 'wastage'

    def ready(self):
        """Start background scheduler for auto-incident creation at 3 AM."""
        import os
        # Only run in the main process (not the reloader subprocess)
        if os.environ.get('RUN_MAIN') != 'true':
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            scheduler = BackgroundScheduler()
            scheduler.add_job(
                _auto_incidents_job,
                CronTrigger(hour=3, minute=0),
                id='auto_incidents_3am',
                replace_existing=True,
            )
            scheduler.start()
            import logging
            logging.getLogger(__name__).info(
                '✅ Auto-incident scheduler started (runs daily at 03:00)'
            )
        except ImportError:
            pass  # apscheduler not installed — use Windows Task Scheduler instead
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f'Scheduler failed to start: {e}')


def _auto_incidents_job():
    """Job called by APScheduler at 3 AM — creates No Wastage incidents."""
    from django.core.management import call_command
    call_command('create_auto_incidents')
