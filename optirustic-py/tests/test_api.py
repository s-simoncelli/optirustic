import unittest
from pathlib import Path

import matplotlib.pyplot as plt
from optirustic import NSGA3, ObjectiveDirection, NSGA2


class PythonTest(unittest.TestCase):
    root_folder = Path(__file__).parent.parent.parent

    def test_reader(self):
        file = self.root_folder / "examples" / "results" / "DTLZ1_3obj_NSGA3_gen400.json"
        data = NSGA3(file.as_posix())
        p = data.problem

        self.assertEqual(p.number_of_variables, 7)
        self.assertEqual(p.variables["x1"].min_value, 0)

        self.assertEqual(len(p.objectives), 3)
        self.assertEqual(p.objectives["f1"].direction, ObjectiveDirection.Minimise)

        self.assertTrue("g" in p.constraints.keys())
        self.assertEqual(data.algorithm, "NSGA3")
        self.assertEqual(data.generation, 400)
        self.assertEqual(data.took.seconds, 4)
        self.assertEqual(data.exported_on.day, 10)

        self.assertEqual(data.individuals[0].constraint_violation, 0)
        self.assertAlmostEqual(data.individuals[0].get_objective_value("f2"), 0.167, 3)
        self.assertTrue("x5" in data.individuals[0].variables.keys())
        self.assertTrue("reference_point_index" in data.individuals[0].data.keys())

        self.assertAlmostEqual(data.hyper_volume([100, 100, 100]), 999999.97, 2)

    def test_plot(self):
        self.assertTrue(isinstance(NSGA2(
            (self.root_folder / "examples" / "results" / "SCH_2obj_NSGA2_gen250.json").as_posix()
        ).plot(), plt.Figure))

        self.assertTrue(isinstance(NSGA3(
            (self.root_folder / "examples" / "results" / "DTLZ1_3obj_NSGA3_gen400.json").as_posix()
        ).plot(), plt.Figure))

        self.assertTrue(isinstance(NSGA3(
            (self.root_folder / "examples" / "results" / "DTLZ1_8obj_NSGA3_gen750.json").as_posix()
        ).plot(), plt.Figure))

        self.assertTrue(isinstance(NSGA2.plot_convergence(
            (self.root_folder / "examples" / "results" / "convergence").as_posix(), [10000, 10000]
        ), plt.Figure))


if __name__ == "__main__":
    unittest.main()
