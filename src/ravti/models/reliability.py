from __future__ import annotations


def reliability_lambdas(
    cmc: float,
    lambda_tax_base: float,
    lambda_ref_base: float,
) -> tuple[float, float]:
    """
    Map cross-modal consistency (CMC) in [0, 1] to (lambda_tax, lambda_ref).

    Interpretation (tunable): when CMC is high, increase reliance on retrieved
    visuals; when low, lean on taxonomic text. This is a minimal differentiable
    prior for sweeps; replace with learned gating as the project matures.
    """
    cmc = float(min(1.0, max(0.0, cmc)))
    lambda_tax = lambda_tax_base * (1.0 - cmc)
    lambda_ref = lambda_ref_base * cmc
    return lambda_tax, lambda_ref
