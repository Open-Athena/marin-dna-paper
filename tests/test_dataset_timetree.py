"""Offline checks for chronology preservation in the cohort figure."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from plot_dataset_timetree import collapse, counts, leaves, parse_newick


class TimeTreeTests(unittest.TestCase):
    def test_collapse_preserves_dates_after_removing_unary_nodes(self):
        tree = parse_newick('(((a:2,b:2):3,c:5):5,d:10);')
        reduced = collapse(tree, {'ab': {'a', 'b'}, 'd': {'d'}}, {'ab': 'A+B', 'd': 'D'})
        self.assertEqual(reduced.age, 10)
        self.assertEqual({n.key for n in leaves(reduced)}, {'ab', 'd'})
        self.assertTrue(all(n.age == 0 for n in leaves(reduced)))

    def test_zoom_uses_subtree_crown_age(self):
        tree = parse_newick('(((a:2,b:2):3,c:5):5,d:10);')
        reduced = collapse(tree, {'ab': {'a', 'b'}, 'c': {'c'}}, {'ab': 'A+B', 'c': 'C'})
        self.assertEqual(reduced.age, 5)

    def test_rejects_non_clock_tree_and_negative_edges(self):
        for newick in ['(a:2,b:3);', '(a:-1,b:-1);']:
            with self.assertRaises(AssertionError):
                parse_newick(newick)

    def test_missing_or_nested_groups_are_not_silently_omitted(self):
        tree = parse_newick('((a:2,b:2):3,c:5);')
        for selection in [{'x': {'missing'}}, {'ab': {'a', 'b'}, 'a': {'a'}}]:
            with self.assertRaises(AssertionError):
                collapse(tree, selection, {k: k for k in selection})

    def test_quoted_taxa_and_rounding_tolerance(self):
        tree = parse_newick("('Family (no order)':2.0000001,b:2)'root';")
        self.assertAlmostEqual(tree.age, 2, places=6)
        self.assertEqual(leaves(tree)[0].name, 'Family (no order)')

    def test_saved_cohort_totals(self):
        animal, mammal = counts()
        self.assertEqual(tuple(map(sum, zip(*animal.values()))), (499, 108))
        self.assertEqual(tuple(map(sum, zip(*mammal.values()))), (81, 108))


if __name__ == '__main__':
    unittest.main()
