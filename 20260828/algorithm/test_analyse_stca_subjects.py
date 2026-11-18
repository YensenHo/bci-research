import unittest
from analyse_stca_subjects import describe


class TestSubjectSummary(unittest.TestCase):
    def test_direction_counts_and_subject_count(self):
        result = describe([.1, -.1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual((result['improved'], result['declined'], result['tied']), (1, 1, 7))
        self.assertAlmostEqual(result['mean_pp'], 0)
        with self.assertRaises(ValueError):
            describe([0] * 27)


if __name__ == '__main__':
    unittest.main()
