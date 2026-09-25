"""
Automated tests for PhysicsLibrary's computational core.

Run them from the repository root (after `pip install -e .` or `pip install .`):

    python -m unittest discover -s tests -v

They use only the standard library's unittest plus numpy.testing, so nothing
extra needs installing; pytest will also collect and run them if you prefer it.

Every test checks a function against a known answer — an analytic result, a
hand-computed value, an independent reference implementation (statsmodels,
scipy), or synthetic data generated with a known ground truth — rather than
just checking that the code runs. Tests decorated with
`@unittest.expectedFailure` document a KNOWN, understood defect (each carries a
docstring explaining it); they keep the suite green today and will start
reporting "unexpected success" once the defect is fixed, at which point the
decorator should simply be removed.
"""
