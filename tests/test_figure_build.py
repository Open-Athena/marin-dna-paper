"""Offline build contracts and render coverage for all external figure recipes."""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from figure_fixtures import EXTERNAL_FIGURES, metric_frame, prediction_frame

import local_data
import plot

ROOT = Path(__file__).resolve().parents[1]


class FigureBuildTests(unittest.TestCase):
    def setUp(self):
        local_data._read_parquet.cache_clear()
        plot._read_checkpoint.cache_clear()
        self.runner = CliRunner()

    def tearDown(self):
        local_data._read_parquet.cache_clear()
        plot._read_checkpoint.cache_clear()

    def test_remote_missing_is_explicit_and_offline(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(local_data, "ROOT", Path(directory)),
            patch.dict("os.environ", {}, clear=True),
        ):
            with patch.object(
                local_data, "urlopen", side_effect=AssertionError("network")
            ):
                with self.assertRaisesRegex(
                    local_data.MissingData,
                    "Original source: s3://bucket/metrics.parquet",
                ):
                    local_data.local_path("s3://bucket/metrics.parquet")

    def test_cached_parquet_is_read_locally(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(local_data, "ROOT", Path(directory)),
        ):
            path = Path(directory) / "remote/s3/bucket/metrics.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame({"value": [0.25], "se": [0.01]}).to_parquet(path)
            self.assertEqual(
                local_data.read_polars("s3://bucket/metrics.parquet")["value"][0], 0.25
            )

    def test_explicit_fetch_downloads_then_uses_cache(self):
        from io import BytesIO

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(local_data, "ROOT", Path(directory)),
            patch.object(local_data, "FETCH", True),
        ):
            with patch.object(
                local_data, "urlopen", return_value=BytesIO(b"snapshot")
            ) as download:
                path = local_data.local_path("https://example.org/input.parquet")
                self.assertEqual(path.read_bytes(), b"snapshot")
                self.assertEqual(
                    local_data.local_path("https://example.org/input.parquet"), path
                )
                download.assert_called_once()
                self.assertEqual(list(path.parent.glob("*.partial")), [])

    def test_missing_input_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vep_by_model_size.pdf"
            output.write_bytes(b"original publication figure")
            with patch.object(
                local_data,
                "read_pandas",
                side_effect=local_data.MissingData("missing predictions"),
            ):
                result = self.runner.invoke(
                    plot.cli, ["--output-dir", directory, "vep_by_model_size"]
                )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("missing predictions", result.output)
            self.assertEqual(output.read_bytes(), b"original publication figure")
            self.assertEqual(list(Path(directory).glob(".render-*")), [])

    def test_second_format_failure_preserves_both_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            for ext in ("pdf", "svg"):
                (output_dir / f"test.{ext}").write_bytes(b"original")
            original_save = plot.Figure.savefig

            def fail_svg(fig, filename, **kwargs):
                if Path(filename).suffix == ".svg":
                    raise RuntimeError("SVG failure")
                return original_save(fig, filename, **kwargs)

            with patch.object(plot.Figure, "savefig", fail_svg):
                with self.assertRaisesRegex(RuntimeError, "SVG failure"):
                    plot.render(output_dir, "test", lambda _: plt.figure())
            for ext in ("pdf", "svg"):
                self.assertEqual((output_dir / f"test.{ext}").read_bytes(), b"original")

    def test_commands_and_make_targets_cover_exactly_the_paper(self):
        stems = set(
            re.findall(
                r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\.pdf\}",
                (ROOT / "paper.tex").read_text(),
            )
        )
        self.assertEqual(set(plot.cli.commands), stems)
        makefile = (ROOT / "Makefile").read_text()
        for stem in stems:
            self.assertIn(f"figure-{stem}:\n", makefile)
            self.assertIn(f"fetch-figure-{stem}:\n", makefile)
            self.assertIn(f"figures/{stem}.pdf", makefile)
        plan = subprocess.run(
            ["make", "-n", "figures-all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(plan.count("src/plot.py "), len(stems))

    def test_artwork_and_fit_report(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("variant_scoring_workflow", "sequence_loss_scaling"):
                result = self.runner.invoke(plot.cli, ["--output-dir", directory, name])
                self.assertEqual(
                    result.exit_code, 0, result.output + repr(result.exception)
                )
                for ext in ("pdf", "svg"):
                    self.assertGreater(
                        (Path(directory) / f"{name}.{ext}").stat().st_size, 100
                    )
            self.assertEqual(
                (Path(directory) / "variant_scoring_workflow.svg").read_bytes(),
                (ROOT / "plot_data/artwork/variant_scoring_workflow.svg").read_bytes(),
            )
            report = (Path(directory) / "sequence_loss_scaling_fit.txt").read_text()
            self.assertIn("data points : 8", report)

    def test_external_recipes_with_synthetic_inputs(self):
        # Exercises the real loaders, filtering, metrics, and plotting end to end;
        # only remote Parquet reads are replaced. Nothing touches figures/.
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(local_data, "read_pandas", prediction_frame),
            patch.object(local_data, "read_polars", metric_frame),
        ):
            for name in EXTERNAL_FIGURES:
                with self.subTest(figure=name):
                    result = self.runner.invoke(
                        plot.cli, ["--output-dir", directory, name]
                    )
                    self.assertEqual(
                        result.exit_code, 0, result.output + repr(result.exception)
                    )
                    for ext in ("pdf", "svg"):
                        self.assertGreater(
                            (Path(directory) / f"{name}.{ext}").stat().st_size, 100
                        )

    def test_style_and_fetch_options_do_not_leak_between_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            previous_style = plt.rcParams.copy()
            for name in ("next_base_prediction", "learning_rate_transfer"):
                result = self.runner.invoke(
                    plot.cli, ["--output-dir", directory, "--fetch", name]
                )
                self.assertEqual(
                    result.exit_code, 0, result.output + repr(result.exception)
                )
                self.assertFalse(local_data.FETCH)
                self.assertEqual(dict(plt.rcParams), dict(previous_style))


if __name__ == "__main__":
    unittest.main()
