import builtins
import importlib
import sys
import unittest
from unittest.mock import patch


class OptionalDependencyImportTests(unittest.TestCase):
    def test_web_app_imports_without_pandas_and_pillow(self):
        sys.modules.pop("tcontext.web_app", None)
        sys.modules.pop("pandas", None)
        sys.modules.pop("PIL", None)

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pandas" or name.startswith("PIL"):
                raise ImportError("simulated missing dependency")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            module = importlib.import_module("tcontext.web_app")

        self.assertIsNone(module.pd)
        self.assertIsNone(module.Image)


if __name__ == "__main__":
    unittest.main()
