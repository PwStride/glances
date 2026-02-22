#
# This file is part of Glances.
#
# SPDX-FileCopyrightText: 2026 Nicolas Hennion <nicolas@nicolargo.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""GPU Performance Notifier.

This module handles desktop notifications for GPU performance warnings.
"""

import platform
import subprocess
from collections import defaultdict

from glances.logger import logger


class GpuNotifier:
    """Handles desktop notifications for GPU performance alerts."""

    def __init__(self, cooldown_period=300):
        """Initialize the GPU notifier.

        Args:
            cooldown_period: Minimum seconds between notifications for same GPU
        """
        self.cooldown_period = cooldown_period
        self.notification_history = defaultdict(
            lambda: {
                'last_notification': 0,
                'last_status': 'OK',
            }
        )

    def should_notify(self, gpu_id, status, current_time):
        """Check if a notification should be sent.

        Args:
            gpu_id: GPU identifier
            status: Current status (OK, WARNING, CRITICAL, ERROR)
            current_time: Current timestamp in seconds

        Returns:
            bool: True if notification should be sent
        """
        history = self.notification_history[gpu_id]

        # Check cooldown period
        if current_time - history['last_notification'] < self.cooldown_period:
            return False

        # Only send notification if status changed or it's critical
        if status == history['last_status'] and status != 'CRITICAL':
            return False

        return True

    def send_notification(self, gpu_id, status, message, current_time):
        """Send desktop notification for GPU alert.

        Args:
            gpu_id: GPU identifier
            status: Alert status (WARNING, CRITICAL, ERROR)
            message: Notification message
            current_time: Current timestamp in seconds

        Returns:
            bool: True if notification was sent successfully
        """
        if not self.should_notify(gpu_id, status, current_time):
            return False

        # Update notification history
        history = self.notification_history[gpu_id]
        history['last_notification'] = current_time
        history['last_status'] = status

        # Prepare notification content
        title = f"GPU Performance Alert: {gpu_id}"
        if status == 'CRITICAL':
            title = f"⚠️ CRITICAL GPU ALERT: {gpu_id}"
        elif status == 'ERROR':
            title = f"❌ GPU MONITORING ERROR: {gpu_id}"

        # Send platform-specific notification
        try:
            self._send_desktop_notification(title, message, status)
            return True
        except Exception as e:
            logger.debug(f"Failed to send GPU notification for {gpu_id}: {e}")
            return False

    def _send_desktop_notification(self, title, message, urgency):
        """Send desktop notification using platform-specific tools.

        Args:
            title: Notification title
            message: Notification message
            urgency: Alert urgency (WARNING, CRITICAL, ERROR)

        Raises:
            Exception: If notification fails to send
        """
        system = platform.system()

        try:
            if system == 'Linux':
                # Use notify-send on Linux
                urgency_level = 'critical' if urgency in ('CRITICAL', 'ERROR') else 'normal'
                subprocess.run(
                    ['notify-send', '-u', urgency_level, '-i', 'dialog-warning', title, message],
                    check=False,
                    timeout=5,
                )
            elif system == 'Darwin':
                # Use osascript on macOS
                # Escape quotes in message
                safe_message = message.replace('"', '\\"')
                safe_title = title.replace('"', '\\"')
                script = f'display notification "{safe_message}" with title "{safe_title}"'
                subprocess.run(['osascript', '-e', script], check=False, timeout=5)
            elif system == 'Windows':
                # Use PowerShell on Windows
                # Escape quotes in message
                safe_message = message.replace('"', '""')
                safe_title = title.replace('"', '""')
                ps_script = (
                    '[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, '
                    'ContentType = WindowsRuntime] > $null; '
                    '$Template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent('
                    '[Windows.UI.Notifications.ToastTemplateType]::ToastText02); '
                    f'$Template.SelectSingleNode("//text[@id=\'1\']").InnerText = "{safe_title}"; '
                    f'$Template.SelectSingleNode("//text[@id=\'2\']").InnerText = "{safe_message}"; '
                    '[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Glances").Show($Template)'
                )
                subprocess.run(['powershell', '-Command', ps_script], check=False, timeout=5)
            else:
                raise NotImplementedError(f"Notifications not supported on {system}")
        except FileNotFoundError as e:
            logger.debug(f"Notification tool not found on {system}: {e}")
            raise
        except subprocess.TimeoutExpired:
            logger.debug(f"Notification command timed out on {system}")
            raise
        except Exception as e:
            logger.debug(f"Error sending notification on {system}: {e}")
            raise

    def reset_history(self, gpu_id=None):
        """Reset notification history for a specific GPU or all GPUs.

        Args:
            gpu_id: GPU ID to reset, or None to reset all
        """
        if gpu_id is None:
            self.notification_history.clear()
        elif gpu_id in self.notification_history:
            del self.notification_history[gpu_id]
