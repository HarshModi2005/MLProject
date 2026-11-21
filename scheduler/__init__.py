"""
scheduler/

Autonomous execution scheduler, reply monitor and automated follow-ups. Import the submodules directly
(scheduler.autonomous_scheduler, scheduler.reply_monitor, scheduler.followup_engine). This package used to import
`start_scheduler`, `stop_scheduler`, `check_replies_job` and `send_followups_job` from modules that do not define them
(the first two are methods of AutonomousScheduler, the last two live in autonomous_scheduler), so no part of it could be
imported at all.
"""
