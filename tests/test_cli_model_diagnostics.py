import unittest
from unittest.mock import patch

from harness_agent import cli


class DummyRuntime:
    model_name = "active-model"
    history = ["history"]


class ModelDiagnosticsCliTests(unittest.TestCase):
    def test_model_status_all_routes_to_status_report(self) -> None:
        with (
            patch("harness_agent.model_diagnostics.model_status_report", return_value='{"ok": true}') as report,
            patch("harness_agent.model_diagnostics.format_model_status_table", return_value="TABLE") as formatter,
        ):
            result = cli.handle_model_status_cli("/model-status all", DummyRuntime())

        self.assertEqual(result, "TABLE")
        report.assert_called_once_with("active-model", ["history"], all_models=True)
        formatter.assert_called_once_with('{"ok": true}')

    def test_model_status_specific_model_uses_current_history(self) -> None:
        with (
            patch("harness_agent.model_diagnostics.model_status_report", return_value='{"ok": true}') as report,
            patch("harness_agent.model_diagnostics.format_model_status_table", return_value="TABLE") as formatter,
        ):
            result = cli.handle_model_status_cli("/model-status other-model", DummyRuntime())

        self.assertEqual(result, "TABLE")
        report.assert_called_once_with("other-model", ["history"], all_models=False)
        formatter.assert_called_once_with('{"ok": true}')

    def test_model_status_json_flag_preserves_raw_json(self) -> None:
        with (
            patch("harness_agent.model_diagnostics.model_status_report", return_value='{"ok": true}') as report,
            patch("harness_agent.model_diagnostics.format_model_status_table") as formatter,
        ):
            result = cli.handle_model_status_cli("/model-status all --json", DummyRuntime())

        self.assertEqual(result, '{"ok": true}')
        report.assert_called_once_with("active-model", ["history"], all_models=True)
        formatter.assert_not_called()

    def test_model_test_all_routes_to_test_report(self) -> None:
        with patch("harness_agent.model_diagnostics.model_test_report", return_value="TEST") as report:
            result = cli.handle_model_test_cli("/model-test all", DummyRuntime())

        self.assertEqual(result, "TEST")
        report.assert_called_once_with("active-model", all_models=True)

    def test_model_test_specific_model_routes_to_test_report(self) -> None:
        with patch("harness_agent.model_diagnostics.model_test_report", return_value="TEST") as report:
            result = cli.handle_model_test_cli("/model-test other-model", DummyRuntime())

        self.assertEqual(result, "TEST")
        report.assert_called_once_with("other-model", all_models=False)


if __name__ == "__main__":
    unittest.main()
