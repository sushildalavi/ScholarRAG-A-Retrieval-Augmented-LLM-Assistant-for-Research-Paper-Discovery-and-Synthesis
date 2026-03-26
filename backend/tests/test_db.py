import os
import unittest
from unittest.mock import patch

from psycopg2 import OperationalError

from backend.services.db import get_connection


class DatabaseConnectionTests(unittest.TestCase):
    def test_supabase_pooler_error_has_local_dev_hint(self):
        database_url = "postgresql://postgres.projectref:pw@aws-1-us-east-1.pooler.supabase.com:5432/postgres"
        old = os.environ.get("DATABASE_URL")
        try:
            os.environ["DATABASE_URL"] = database_url
            with patch("backend.services.db._load_dotenv_if_available", return_value=None):
                with patch(
                    "backend.services.db.psycopg2.connect",
                    side_effect=OperationalError("FATAL:  Tenant or user not found"),
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        get_connection()
            self.assertIn("docker compose up -d db", str(ctx.exception))
            self.assertIn("Supabase Dashboard > Connect", str(ctx.exception))
        finally:
            if old is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = old

    def test_generic_connection_error_is_preserved(self):
        database_url = "postgresql://user:pw@127.0.0.1:5432/app"
        old = os.environ.get("DATABASE_URL")
        try:
            os.environ["DATABASE_URL"] = database_url
            with patch("backend.services.db._load_dotenv_if_available", return_value=None):
                with patch(
                    "backend.services.db.psycopg2.connect",
                    side_effect=OperationalError("connection refused"),
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        get_connection()
            self.assertIn("DATABASE_URL connection failed", str(ctx.exception))
            self.assertIn("connection refused", str(ctx.exception))
        finally:
            if old is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = old


if __name__ == "__main__":
    unittest.main()
