"""Promote an existing user account to the admin role.

Usage (from repo root):
    python -m backend.scripts.promote_admin you@example.com

The target user must already exist (sign up via the normal flow first).
This script writes directly to the same SQLite database the API uses.
"""

from __future__ import annotations

import argparse
import sys

from backend.app.services.user_service import normalize_email, user_service


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a user to the 'admin' role.")
    parser.add_argument("email", help="Email address of an existing user")
    parser.add_argument(
        "--demote",
        action="store_true",
        help="Set role back to 'clinician' instead of promoting.",
    )
    args = parser.parse_args()

    email = normalize_email(args.email)
    existing = user_service.get_by_email(email)
    if existing is None:
        print(f"No user found for {email}. Have them sign up first.", file=sys.stderr)
        return 1

    if args.demote:
        with user_service._connect() as conn:  # noqa: SLF001
            conn.execute("UPDATE users SET role = 'clinician' WHERE email = ?", (email,))
            conn.commit()
        updated = user_service.get_by_email(email)
        print(f"OK: {email} role -> {updated['role']}")
        return 0

    updated = user_service.promote_to_admin(email)
    if updated is None:
        print(f"Failed to promote {email}.", file=sys.stderr)
        return 1
    print(f"OK: {email} role -> {updated['role']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
