from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from backend.app.config import settings

logger = logging.getLogger("medscanassist.email")


def _build_verification_email(to_email: str, full_name: str, verify_url: str) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Verify your MedScanAssist account"
    msg["From"] = f"{settings.smtp_from_name} <{settings.smtp_user}>"
    msg["To"] = to_email

    text_body = (
        f"Hi {full_name},\n\n"
        f"Welcome to MedScanAssist! Please verify your email address by visiting:\n\n"
        f"{verify_url}\n\n"
        f"This link expires in {settings.verification_token_expire_hours} hours.\n\n"
        f"If you did not create an account, you can safely ignore this email.\n\n"
        f"— The MedScanAssist Team"
    )

    html_body = f"""\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f0f4f8;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="padding:40px 20px;">
    <tr><td align="center">
      <table width="480" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.08);">
        <tr>
          <td style="background:linear-gradient(135deg,#1a56db,#2563eb);padding:32px 40px;text-align:center;">
            <h1 style="margin:0;color:#ffffff;font-size:24px;font-weight:700;letter-spacing:-0.5px;">MedScanAssist</h1>
            <p style="margin:8px 0 0;color:rgba(255,255,255,0.85);font-size:14px;">Clinical Analysis Platform</p>
          </td>
        </tr>
        <tr>
          <td style="padding:36px 40px 20px;">
            <p style="margin:0 0 16px;color:#1e293b;font-size:16px;">Hi <strong>{full_name}</strong>,</p>
            <p style="margin:0 0 24px;color:#475569;font-size:15px;line-height:1.6;">
              Welcome to MedScanAssist! To activate your account, please verify your email address by clicking the button below.
            </p>
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr><td align="center" style="padding:8px 0 28px;">
                <a href="{verify_url}"
                   style="display:inline-block;background:#2563eb;color:#ffffff;text-decoration:none;
                          padding:14px 36px;border-radius:8px;font-size:15px;font-weight:600;
                          letter-spacing:0.3px;">
                  Verify Email Address
                </a>
              </td></tr>
            </table>
            <p style="margin:0 0 12px;color:#64748b;font-size:13px;line-height:1.5;">
              Or copy and paste this link into your browser:
            </p>
            <p style="margin:0 0 24px;color:#2563eb;font-size:13px;word-break:break-all;">
              {verify_url}
            </p>
            <hr style="border:none;border-top:1px solid #e2e8f0;margin:24px 0;">
            <p style="margin:0;color:#94a3b8;font-size:12px;line-height:1.5;">
              This link expires in {settings.verification_token_expire_hours} hours.
              If you did not create an account, you can safely ignore this email.
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    return msg


def send_verification_email(to_email: str, full_name: str, token: str) -> bool:
    """Send a verification email. Returns True on success, False on failure."""
    verify_url = f"{settings.frontend_url}/verify?token={token}"

    if not settings.smtp_password:
        logger.warning(
            "SMTP_PASSWORD not configured — skipping email send. "
            "Verification URL: %s", verify_url,
        )
        return False

    msg = _build_verification_email(to_email, full_name, verify_url)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_user, to_email, msg.as_string())
        logger.info("Verification email sent to %s", to_email)
        return True
    except Exception:
        logger.exception("Failed to send verification email to %s", to_email)
        return False
