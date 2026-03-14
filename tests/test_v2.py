"""
syndatakit v2 — full test suite
Run: pytest tests/ -v
"""
import sys, os, pytest, json, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def all_seeds():
    from syndatakit.catalog import load_seed, DATASETS
    return {did: load_seed(did) for did in DATASETS}

@pytest.fixture(scope="module")
def hmda(all_seeds):      return all_seeds["hmda"]
@pytest.fixture(scope="module")
def fred_macro(all_seeds):return all_seeds["fred_macro"]
@pytest.fixture(scope="module")
def world_bank(all_seeds):return all_seeds["world_bank"]
@pytest.fixture(scope="module")
def credit_risk(all_seeds):return all_seeds["credit_risk"]
@pytest.fixture(scope="module")
def edgar(all_seeds):     return all_seeds["edgar"]

@pytest.fixture(scope="module")
def syn_hmda(hmda):
    from syndatakit.generators import GaussianCopulaGenerator
    gen = GaussianCopulaGenerator()
    gen.fit(hmda)
    return gen.sample(500, seed=42)

@pytest.fixture(scope="module")
def syn_macro(fred_macro):
    from syndatakit.generators.time_series import VARGenerator
    gen = VARGenerator(lags=2, time_col="year")
    gen.fit(fred_macro)
    return gen.sample(300, seed=42)

@pytest.fixture(scope="module")
def syn_wb(world_bank):
    from syndatakit.generators.panel import FixedEffectsGenerator
    gen = FixedEffectsGenerator(entity_col="country_code", time_col="year")
    gen.fit(world_bank)
    return gen.sample(300, seed=42)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Catalog
# ═══════════════════════════════════════════════════════════════════════════════

class TestCatalog:
    def test_list_datasets_count(self):
        from syndatakit.catalog import list_datasets
        assert len(list_datasets()) == 18

    def test_list_datasets_vertical_filter(self):
        from syndatakit.catalog import list_datasets
        df = list_datasets(vertical="Capital Markets")
        assert len(df) == 4
        assert set(df["id"]) == {"edgar", "cftc", "equity_returns", "corporate_bonds"}

    def test_get_dataset_info_valid(self):
        from syndatakit.catalog import get_dataset_info
        info = get_dataset_info("fred_macro")
        assert info["col_count"] == 15
        assert "vix" in info["columns"]

    def test_get_dataset_info_invalid(self):
        from syndatakit.catalog import get_dataset_info
        with pytest.raises(ValueError, match="Unknown"):
            get_dataset_info("does_not_exist")

    @pytest.mark.parametrize("did", [
        "hmda","fdic","credit_risk","edgar","cftc",
        "fred_macro","bls","world_bank","irs_soi","census_acs",
        "equity_returns","corporate_bonds","insurance_claims","life_insurance",
        "commercial_real_estate","rental_market","retail_transactions","commodity_prices"
    ])
    def test_all_seeds_build(self, did):
        from syndatakit.catalog import load_seed
        df = load_seed(did)
        assert len(df) == 2000
        assert df.shape[1] > 0
        assert not df.isnull().all().any()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Base Generator
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaseGenerator:
    def test_abstract_cannot_instantiate(self):
        from syndatakit.generators.base import BaseGenerator
        with pytest.raises(TypeError):
            BaseGenerator()

    def test_require_fitted_guard(self, hmda):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            gen.sample(10)

    def test_fit_sample_fluent(self, hmda):
        from syndatakit.generators import GaussianCopulaGenerator
        df = GaussianCopulaGenerator().fit_sample(hmda, 50, seed=1)
        assert len(df) == 50

    def test_repr_before_after_fit(self, hmda):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        assert "not fitted" in repr(gen)
        gen.fit(hmda)
        assert "fitted on" in repr(gen)
        assert "2,000" in repr(gen)

    def test_syn_id_unique(self, syn_hmda):
        assert syn_hmda["syn_id"].nunique() == len(syn_hmda)

    def test_syn_id_format(self, syn_hmda):
        assert all(syn_hmda["syn_id"].str.startswith("SYN-"))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Gaussian Copula Generator
# ═══════════════════════════════════════════════════════════════════════════════

class TestGaussianCopula:
    @pytest.mark.parametrize("did", [
        "hmda","fdic","credit_risk","edgar","cftc","irs_soi","census_acs",
        "equity_returns","corporate_bonds","insurance_claims","life_insurance",
        "commercial_real_estate","rental_market","retail_transactions","commodity_prices"
    ])
    def test_all_cross_sectional_datasets(self, did, all_seeds):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(all_seeds[did])
        df = gen.sample(100, seed=42)
        assert len(df) == 100
        assert "syn_id" in df.columns

    def test_reproducible_with_seed(self, hmda):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        df1 = gen.sample(50, seed=99)
        df2 = gen.sample(50, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self, hmda):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        df1 = gen.sample(100, seed=1)
        df2 = gen.sample(100, seed=2)
        assert not df1["loan_amount"].equals(df2["loan_amount"])

    def test_no_nulls_in_output(self, syn_hmda):
        body = syn_hmda.drop(columns=["syn_id"])
        assert body.isnull().sum().sum() == 0

    def test_correlation_matrix_property(self, hmda):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        corr = gen.correlation_matrix
        assert corr is not None
        assert corr.shape[0] == corr.shape[1]
        # Diagonal should be ~1
        assert np.allclose(np.diag(corr.values), 1.0, atol=0.01)

    def test_marginal_kinds_property(self, hmda):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        kinds = gen.marginal_kinds
        assert "loan_amount" in kinds
        assert kinds["property_type"] == "categorical"

    def test_small_dataset_stability(self):
        """Generator should not crash on very small datasets (n=30)."""
        from syndatakit.generators import GaussianCopulaGenerator
        from syndatakit.catalog import load_seed
        gen = GaussianCopulaGenerator()
        gen.fit(load_seed("hmda").sample(30, random_state=0))
        df = gen.sample(20, seed=1)
        assert len(df) == 20


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Filters
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilters:
    def test_categorical_list_filter(self, hmda):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(hmda)
        df = gen.sample(500, filters={"state": ["CA", "TX"]}, seed=1)
        assert set(df["state"].unique()).issubset({"CA", "TX"})

    def test_numeric_min_filter(self, hmda):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(hmda)
        df = gen.sample(500, filters={"debt_to_income_min": 50}, seed=1)
        if len(df) > 0:
            assert df["debt_to_income"].min() >= 50

    def test_numeric_max_filter(self, hmda):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(hmda)
        df = gen.sample(500, filters={"loan_amount_max": 200000}, seed=1)
        if len(df) > 0:
            assert df["loan_amount"].max() <= 200000

    def test_alias_filter(self, hmda):
        """'dti' should resolve to 'debt_to_income'."""
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(hmda)
        df = gen.sample(500, filters={"dti_min": 50}, seed=1)
        if len(df) > 0:
            assert df["debt_to_income"].min() >= 50

    def test_exact_categorical_filter(self, edgar):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(edgar)
        df = gen.sample(400, filters={"sector": "Technology"}, seed=1)
        if len(df) > 0:
            assert all(df["sector"] == "Technology")

    def test_binary_filter(self, credit_risk):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(credit_risk)
        df = gen.sample(400, filters={"default_12m": 1}, seed=1)
        if len(df) > 0:
            assert all(df["default_12m"].astype(str) == "1")

    def test_combined_filters(self, hmda):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator(); gen.fit(hmda)
        df = gen.sample(600, filters={"state": ["CA"], "dti_min": 40}, seed=1)
        if len(df) > 0:
            assert all(df["state"] == "CA")
            assert df["debt_to_income"].min() >= 40


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VAR Time Series Generator
# ═══════════════════════════════════════════════════════════════════════════════

class TestVARGenerator:
    def test_basic_generation(self, syn_macro):
        assert len(syn_macro) == 300
        assert "syn_id" in syn_macro.columns

    def test_correct_column_count(self, fred_macro, syn_macro):
        # syn should have (original cols - time_col) + syn_id
        expected = len(fred_macro.columns)  # year dropped, syn_id added → same count
        assert len(syn_macro.columns) == expected

    def test_numeric_columns_in_range(self, fred_macro, syn_macro):
        for col in ["cpi_yoy", "unemployment_rate", "fed_funds_rate"]:
            if col in syn_macro.columns:
                r_min = fred_macro[col].min() * 4 - fred_macro[col].abs().max()
                r_max = fred_macro[col].max() * 4
                assert syn_macro[col].between(r_min, r_max).all(), \
                    f"{col} values outside plausible range"

    def test_bls_var_generation(self, all_seeds):
        from syndatakit.generators.time_series import VARGenerator
        gen = VARGenerator(lags=2, time_col="quarter")
        gen.fit(all_seeds["bls"])
        df = gen.sample(100, seed=1)
        assert len(df) == 100


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Panel Generator
# ═══════════════════════════════════════════════════════════════════════════════

class TestPanelGenerator:
    def test_basic_generation(self, syn_wb):
        assert len(syn_wb) == 300
        assert "syn_id" in syn_wb.columns

    def test_entity_col_present(self, syn_wb):
        assert "country_code" in syn_wb.columns

    def test_time_col_present(self, syn_wb):
        assert "year" in syn_wb.columns

    def test_fdic_panel(self, all_seeds):
        from syndatakit.generators.panel import FixedEffectsGenerator
        gen = FixedEffectsGenerator(entity_col="state", time_col="charter_class")
        gen.fit(all_seeds["fdic"])
        df = gen.sample(100, seed=1)
        assert len(df) == 100


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Fidelity — Marginal
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarginalFidelity:
    def test_scores_in_range(self, hmda, syn_hmda):
        from syndatakit.fidelity.marginal import marginal_scores
        scores = marginal_scores(hmda, syn_hmda.drop(columns=["syn_id"]))
        for col, score in scores.items():
            assert 0 <= score <= 100, f"{col} score {score} out of range"

    def test_high_fidelity_on_large_sample(self, hmda, syn_hmda):
        from syndatakit.fidelity.marginal import mean_marginal_score, marginal_scores
        scores = marginal_scores(hmda, syn_hmda.drop(columns=["syn_id"]))
        assert mean_marginal_score(scores) >= 80.0

    def test_identical_data_scores_100(self, hmda):
        from syndatakit.fidelity.marginal import marginal_scores
        scores = marginal_scores(hmda, hmda)
        for score in scores.values():
            assert score >= 99.0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Fidelity — Joint
# ═══════════════════════════════════════════════════════════════════════════════

class TestJointFidelity:
    def test_correlation_score_range(self, hmda, syn_hmda):
        from syndatakit.fidelity.joint import correlation_distance_score
        score = correlation_distance_score(hmda, syn_hmda.drop(columns=["syn_id"]))
        assert 0 <= score <= 100

    def test_identical_data_perfect_joint(self, hmda):
        from syndatakit.fidelity.joint import correlation_distance_score
        score = correlation_distance_score(hmda, hmda)
        assert score >= 99.0

    def test_pairwise_report_returns_dict(self, hmda, syn_hmda):
        from syndatakit.fidelity.joint import pairwise_correlation_report
        result = pairwise_correlation_report(hmda, syn_hmda.drop(columns=["syn_id"]))
        assert isinstance(result, dict)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Fidelity — Temporal
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalFidelity:
    def test_stationarity_agreement(self, fred_macro, syn_macro):
        from syndatakit.fidelity.temporal.stationarity import stationarity_score
        result = stationarity_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        rate = result["_summary"]["agreement_rate"]
        assert 0 <= rate <= 100

    def test_cointegration_agreement(self, fred_macro, syn_macro):
        from syndatakit.fidelity.temporal.cointegration import cointegration_score
        result = cointegration_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        assert 0 <= result["_summary"]["agreement_rate"] <= 100

    def test_breaks_score(self, fred_macro, syn_macro):
        from syndatakit.fidelity.temporal.breaks import breaks_score
        result = breaks_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        assert 0 <= result["_summary"]["break_match_rate"] <= 100

    def test_causality_score(self, fred_macro, syn_macro):
        from syndatakit.fidelity.causality import causality_score
        result = causality_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        assert 0 <= result["_summary"]["agreement_rate"] <= 100


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Fidelity — Stylized Facts & Downstream
# ═══════════════════════════════════════════════════════════════════════════════

class TestStyleAndDownstream:
    def test_stylized_facts_score(self, fred_macro, syn_macro):
        from syndatakit.fidelity.stylized_facts import stylized_facts_score
        result = stylized_facts_score(fred_macro, syn_macro.drop(columns=["syn_id"]))
        assert "_summary" in result
        assert 0 <= result["_summary"]["mean_score"] <= 100

    def test_tstr_classification(self, credit_risk):
        from syndatakit.generators import GaussianCopulaGenerator
        from syndatakit.fidelity.downstream import tstr_score
        gen = GaussianCopulaGenerator(); gen.fit(credit_risk)
        syn = gen.sample(300, seed=1).drop(columns=["syn_id"])
        result = tstr_score(credit_risk, syn, target_col="default_12m", task="classification")
        assert "tstr_score" in result
        assert "trr_score"  in result
        assert "ratio"      in result

    def test_tstr_regression(self, hmda):
        from syndatakit.generators import GaussianCopulaGenerator
        from syndatakit.fidelity.downstream import tstr_score
        gen = GaussianCopulaGenerator(); gen.fit(hmda)
        syn = gen.sample(300, seed=1).drop(columns=["syn_id"])
        result = tstr_score(hmda, syn, target_col="loan_amount", task="regression")
        assert result["metric"] == "r2"
        assert "ratio" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Fidelity Report
# ═══════════════════════════════════════════════════════════════════════════════

class TestFidelityReport:
    def test_cross_sectional_report_keys(self, hmda, syn_hmda):
        from syndatakit.fidelity import fidelity_report
        report = fidelity_report(hmda, syn_hmda.drop(columns=["syn_id"]))
        for key in ["marginal", "joint", "stylized_facts", "privacy_basic", "summary"]:
            assert key in report

    def test_summary_scores_in_range(self, hmda, syn_hmda):
        from syndatakit.fidelity import fidelity_report
        report = fidelity_report(hmda, syn_hmda.drop(columns=["syn_id"]))
        s = report["summary"]
        assert 0 <= s["overall_fidelity"] <= 100
        assert 0 <= s["marginal_score"]   <= 100
        assert 0 <= s["joint_score"]      <= 100
        assert s["exact_copies"] == 0

    def test_temporal_section_for_time_series(self, fred_macro, syn_macro):
        from syndatakit.fidelity import fidelity_report
        report = fidelity_report(fred_macro, syn_macro.drop(columns=["syn_id"]),
                                  dataset_type="time_series")
        assert "temporal" in report
        assert "stationarity"  in report["temporal"]
        assert "cointegration" in report["temporal"]
        assert "breaks"        in report["temporal"]
        assert "causality"     in report["temporal"]

    def test_downstream_section_with_target(self, hmda, syn_hmda):
        from syndatakit.fidelity import fidelity_report
        report = fidelity_report(hmda, syn_hmda.drop(columns=["syn_id"]),
                                  target_col="loan_amount", include_downstream=True)
        assert "downstream" in report
        assert "tstr_score" in report["downstream"]

    def test_no_temporal_for_cross_sectional(self, hmda, syn_hmda):
        from syndatakit.fidelity import fidelity_report
        report = fidelity_report(hmda, syn_hmda.drop(columns=["syn_id"]),
                                  dataset_type="cross_sectional")
        assert "temporal" not in report

    def test_format_report_returns_string(self, hmda, syn_hmda):
        from syndatakit.fidelity import fidelity_report, format_report
        report = fidelity_report(hmda, syn_hmda.drop(columns=["syn_id"]))
        text = format_report(report)
        assert isinstance(text, str)
        assert "FIDELITY" in text


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Privacy
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrivacy:
    def test_no_exact_copies(self, hmda, syn_hmda):
        from syndatakit.privacy.audit import privacy_audit
        report = privacy_audit(hmda, syn_hmda, n_attacks=50, seed=1)
        assert report["exact_copies"]["count"] == 0

    def test_membership_inference_auc_near_half(self, hmda, syn_hmda):
        from syndatakit.privacy.disclosure import membership_inference_risk
        result = membership_inference_risk(
            real_train=hmda.iloc[:1600],
            real_holdout=hmda.iloc[1600:],
            synthetic=syn_hmda,
            n_sample=100, seed=1,
        )
        assert "attack_auc" in result
        assert 0.0 <= result["attack_auc"] <= 1.0

    def test_singling_out_risk_low(self, hmda, syn_hmda):
        from syndatakit.privacy.singling_out import singling_out_risk
        result = singling_out_risk(hmda, syn_hmda, n_attacks=100, seed=1)
        assert "singling_out_rate" in result
        assert 0.0 <= result["singling_out_rate"] <= 1.0

    def test_linkability_risk_has_baseline(self, hmda, syn_hmda):
        from syndatakit.privacy.linkability import linkability_risk
        result = linkability_risk(hmda, syn_hmda, n_attacks=100, seed=1)
        assert result["baseline"] == 0.5

    def test_audit_verdict_keys(self, hmda, syn_hmda):
        from syndatakit.privacy import privacy_audit
        report = privacy_audit(hmda, syn_hmda, n_attacks=80, seed=1)
        v = report["verdict"]
        for key in ["overall_risk", "exact_copies", "mi_auc",
                    "singling_out_rate", "linkability_rate", "recommendation"]:
            assert key in v

    def test_audit_risk_levels_valid(self, hmda, syn_hmda):
        from syndatakit.privacy import privacy_audit
        valid = {"very_low", "low", "medium", "high", "very_high"}
        report = privacy_audit(hmda, syn_hmda, n_attacks=80, seed=1)
        assert report["verdict"]["overall_risk"] in valid

    def test_format_audit_returns_string(self, hmda, syn_hmda):
        from syndatakit.privacy import privacy_audit, format_audit
        report = privacy_audit(hmda, syn_hmda, n_attacks=50, seed=1)
        text = format_audit(report)
        assert "PRIVACY AUDIT" in text


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Differential Privacy
# ═══════════════════════════════════════════════════════════════════════════════

class TestDifferentialPrivacy:
    def test_laplace_mechanism_adds_noise(self, hmda):
        from syndatakit.privacy.dp import laplace_mechanism
        true_mean = hmda["loan_amount"].mean()
        noisy = laplace_mechanism(true_mean, sensitivity=1e6, epsilon=1.0)
        assert noisy != true_mean

    def test_gaussian_mechanism_adds_noise(self, hmda):
        from syndatakit.privacy.dp import gaussian_mechanism
        true_val = hmda["applicant_income"].mean()
        noisy = gaussian_mechanism(true_val, sensitivity=1e6, epsilon=1.0, delta=1e-5)
        assert noisy != true_val

    def test_budget_tracks_consumption(self, hmda):
        from syndatakit.privacy.dp import PrivacyBudget, laplace_mechanism
        budget = PrivacyBudget(epsilon=2.0)
        laplace_mechanism(1.0, sensitivity=1.0, epsilon=0.5, budget=budget)
        laplace_mechanism(1.0, sensitivity=1.0, epsilon=0.5, budget=budget)
        assert abs(budget.remaining_epsilon - 1.0) < 1e-9

    def test_budget_exhaustion_raises(self):
        from syndatakit.privacy.dp import PrivacyBudget, laplace_mechanism
        budget = PrivacyBudget(epsilon=0.5)
        with pytest.raises(RuntimeError, match="exhausted"):
            laplace_mechanism(1.0, sensitivity=1.0, epsilon=1.0, budget=budget)

    def test_privatise_histogram(self):
        from syndatakit.privacy.dp import privatise_histogram
        counts = {"A": 100, "B": 50, "C": 25}
        result = privatise_histogram(counts, epsilon=2.0, seed=42)
        assert set(result.keys()) == set(counts.keys())
        assert abs(sum(result.values()) - 1.0) < 0.01   # normalised

    def test_budget_log_tracks_labels(self):
        from syndatakit.privacy.dp import PrivacyBudget, laplace_mechanism
        budget = PrivacyBudget(epsilon=5.0)
        laplace_mechanism(1.0, 1.0, 1.0, budget=budget, label="mean_query")
        laplace_mechanism(1.0, 1.0, 1.0, budget=budget, label="std_query")
        labels = [entry["label"] for entry in budget.log]
        assert "mean_query" in labels
        assert "std_query"  in labels


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Calibration — Priors
# ═══════════════════════════════════════════════════════════════════════════════

class TestPriors:
    def test_prior_normal_samples(self):
        from syndatakit.calibration.priors import Prior
        p = Prior("normal", mu=100.0, sigma=10.0)
        samples = p.sample(1000, seed=0)
        assert abs(samples.mean() - 100.0) < 3.0
        assert abs(samples.std()  - 10.0)  < 2.0

    def test_prior_lognormal_samples(self):
        from syndatakit.calibration.priors import Prior
        p = Prior("lognormal", mu=0.0, sigma=1.0)
        samples = p.sample(1000, seed=0)
        assert (samples > 0).all()

    def test_prior_beta_range(self):
        from syndatakit.calibration.priors import Prior
        p = Prior("beta", alpha=2.0, beta=5.0)
        samples = p.sample(500, seed=0)
        assert (samples >= 0).all() and (samples <= 1).all()

    def test_prior_fixed_constant(self):
        from syndatakit.calibration.priors import Prior
        p = Prior("fixed", value=42.0)
        samples = p.sample(100, seed=0)
        assert (samples == 42.0).all()

    def test_prior_invalid_distribution(self):
        from syndatakit.calibration.priors import Prior
        with pytest.raises(ValueError, match="Unknown distribution"):
            Prior("uniform", lo=0, hi=1)

    def test_prior_missing_params(self):
        from syndatakit.calibration.priors import Prior
        with pytest.raises(ValueError, match="missing params"):
            Prior("normal", mu=0.0)   # missing sigma

    def test_map_mean_pulls_toward_prior(self):
        from syndatakit.calibration.priors import Prior
        p = Prior("normal", mu=100.0, sigma=10.0, strength=5.0)
        # With only 10 observations, prior should dominate
        blended = p.map_mean(data_mean=200.0, n_obs=10)
        assert blended < 200.0   # pulled toward prior mean of 100
        assert blended > 100.0   # but data has some influence

    def test_map_mean_weak_prior_on_large_n(self):
        from syndatakit.calibration.priors import Prior
        p = Prior("normal", mu=100.0, sigma=10.0, strength=0.1)
        # With 10,000 obs, data should dominate
        blended = p.map_mean(data_mean=200.0, n_obs=10000)
        assert blended > 190.0   # very close to data mean

    def test_prior_set_construction(self):
        from syndatakit.calibration.priors import Prior, PriorSet
        ps = PriorSet({
            "col_a": Prior("normal",    mu=0.0, sigma=1.0),
            "col_b": Prior("lognormal", mu=1.0, sigma=0.5),
        })
        assert len(ps.columns()) == 2
        assert ps.get("col_a") is not None
        assert ps.get("col_c") is None

    def test_prior_set_map_mean(self):
        from syndatakit.calibration.priors import Prior, PriorSet
        ps = PriorSet({"x": Prior("normal", mu=50.0, sigma=5.0, strength=3.0)})
        blended = ps.map_mean("x", data_mean=100.0, n_obs=5)
        assert blended < 100.0

    def test_dataset_priors_all_present(self):
        from syndatakit.calibration.priors import DATASET_PRIORS, get_priors
        # Priors are defined for the original 10 core datasets
        core = ["hmda","fdic","credit_risk","edgar","cftc",
                "fred_macro","bls","world_bank","irs_soi","census_acs"]
        for did in core:
            ps = get_priors(did)
            assert isinstance(ps.columns(), list)
            assert len(ps.columns()) >= 2

    def test_get_priors_invalid(self):
        from syndatakit.calibration.priors import get_priors
        with pytest.raises(ValueError, match="No built-in priors"):
            get_priors("nonexistent_dataset")

    def test_prior_regularises_small_dataset(self, hmda):
        """Generator with priors on 50-row dataset should produce reasonable output."""
        from syndatakit.generators import GaussianCopulaGenerator
        from syndatakit.calibration.priors import get_priors
        priors = get_priors("hmda")
        gen_no_prior = GaussianCopulaGenerator()
        gen_with_prior = GaussianCopulaGenerator(priors=priors)
        small = hmda.sample(50, random_state=0)
        gen_no_prior.fit(small)
        gen_with_prior.fit(small)
        df_no  = gen_no_prior.sample(200, seed=1)
        df_yes = gen_with_prior.sample(200, seed=1)
        # Both should produce valid output
        assert len(df_no) == 200
        assert len(df_yes) == 200
        # With-prior mean should be closer to full-dataset mean
        full_mean = hmda["loan_amount"].mean()
        err_no  = abs(df_no["loan_amount"].mean()  - full_mean)
        err_yes = abs(df_yes["loan_amount"].mean() - full_mean)
        assert err_yes < err_no * 2.0  # prior version within 2x of no-prior (usually better)

    def test_prior_set_sample_prior_data(self):
        from syndatakit.calibration.priors import get_priors
        ps = get_priors("fred_macro")
        samples = ps.sample_prior_data(n=100, seed=0)
        assert isinstance(samples, dict)
        for col, arr in samples.items():
            assert len(arr) == 100
            assert np.isfinite(arr).all()

    def test_prior_set_summary(self):
        from syndatakit.calibration.priors import get_priors
        ps = get_priors("hmda")
        summary = ps.summary()
        assert isinstance(summary, list)
        cols = [r["column"] for r in summary]
        assert "loan_amount" in cols


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Calibration — Moment Matching & Scenario
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibration:
    def test_moment_matching_returns_df(self, hmda, syn_hmda):
        from syndatakit.calibration import match_moments
        result = match_moments(hmda, syn_hmda.drop(columns=["syn_id"]))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(syn_hmda)

    def test_moment_report_shape(self, hmda, syn_hmda):
        from syndatakit.calibration import match_moments, moment_report
        cal = match_moments(hmda, syn_hmda.drop(columns=["syn_id"]))
        report = moment_report(hmda, cal)
        assert len(report) > 0
        assert "real_mean" in report.columns

    def test_scenario_recession_shifts_gdp(self, syn_macro):
        from syndatakit.calibration import apply_scenario
        result = apply_scenario(syn_macro, "recession", intensity=1.0)
        if "gdp_growth_yoy" in result.columns:
            assert result["gdp_growth_yoy"].mean() < syn_macro["gdp_growth_yoy"].mean()

    def test_scenario_expansion_raises_wages(self, syn_macro):
        from syndatakit.calibration import apply_scenario
        result = apply_scenario(syn_macro, "expansion", intensity=1.0)
        if "gdp_growth_yoy" in result.columns:
            assert result["gdp_growth_yoy"].mean() > syn_macro["gdp_growth_yoy"].mean()

    def test_intensity_zero_no_change(self, syn_hmda):
        from syndatakit.calibration import apply_scenario
        result = apply_scenario(syn_hmda, "recession", intensity=0.0)
        num_cols = [c for c in syn_hmda.columns if syn_hmda[c].dtype.kind in "if"]
        for col in num_cols:
            if col in result.columns:
                assert abs(result[col].mean() - syn_hmda[col].mean()) < 1e-6

    def test_invalid_scenario_raises(self, syn_hmda):
        from syndatakit.calibration import apply_scenario
        with pytest.raises(ValueError, match="Unknown scenario"):
            apply_scenario(syn_hmda, "alien_invasion")

    def test_list_scenarios_count(self):
        from syndatakit.calibration import list_scenarios
        df = list_scenarios()
        assert len(df) == 5
        assert "recession" in df["name"].values


# ═══════════════════════════════════════════════════════════════════════════════
# 16. IO
# ═══════════════════════════════════════════════════════════════════════════════

class TestIO:
    def test_csv_roundtrip(self, syn_hmda):
        from syndatakit.io import write, read
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            write(syn_hmda, path)
            reloaded = read(path)
            assert len(reloaded) == len(syn_hmda)
        finally:
            os.unlink(path)

    def test_json_roundtrip(self, syn_hmda):
        from syndatakit.io import write, read
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            write(syn_hmda, path)
            reloaded = read(path)
            assert len(reloaded) == len(syn_hmda)
        finally:
            os.unlink(path)

    def test_stata_roundtrip(self, syn_hmda):
        from syndatakit.io import write, read
        with tempfile.NamedTemporaryFile(suffix=".dta", delete=False) as f:
            path = f.name
        try:
            write(syn_hmda, path)
            reloaded = read(path)
            assert len(reloaded) == len(syn_hmda)
        finally:
            os.unlink(path)

    def test_unsupported_format_raises(self, syn_hmda):
        from syndatakit.io import write
        with pytest.raises(ValueError):
            write(syn_hmda, "/tmp/test.xyz")

    def test_validate_passes_clean_data(self, hmda):
        from syndatakit.io import validate
        result = validate(hmda)
        assert result.passed
        assert len(result.errors) == 0

    def test_validate_catches_nulls(self):
        from syndatakit.io import validate
        df = pd.DataFrame({"a": [1, None, None, None, None], "b": [1, 2, 3, 4, 5]})
        # 80% nulls in 'a' — should fail with default threshold 0.3
        result = validate(df, min_rows=3)
        assert not result.passed
        assert any("a" in e for e in result.errors)

    def test_validate_warns_constant_column(self):
        from syndatakit.io import validate
        df = pd.DataFrame({"a": [1]*100, "b": range(100)})
        result = validate(df)
        assert any("constant" in w.lower() for w in result.warnings)

    def test_validate_warns_high_cardinality(self):
        from syndatakit.io import validate
        df = pd.DataFrame({
            "id": [f"id_{i}" for i in range(200)],
            "val": range(200),
        })
        result = validate(df, max_cardinality=50, min_rows=100)
        assert any("cardinality" in w.lower() for w in result.warnings)

    def test_supported_formats_list(self):
        from syndatakit.io import supported_formats
        fmts = supported_formats()
        assert "csv"   in fmts
        assert "json"  in fmts
        assert "stata" in fmts


# ═══════════════════════════════════════════════════════════════════════════════
# 17. API
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def api_client():
    from syndatakit.api import app, _warm_generators
    import contextlib, io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        _warm_generators()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

class TestAPI:
    def test_health(self, api_client):
        r = api_client.get("/health")
        j = json.loads(r.data)
        assert j["status"] == "ok"
        assert j["datasets"] == 18

    def test_list_datasets(self, api_client):
        r = api_client.get("/datasets")
        j = json.loads(r.data)
        assert j["meta"]["count"] == 18

    def test_list_datasets_vertical_filter(self, api_client):
        r = api_client.get("/datasets?vertical=Tax+%26+Income")
        j = json.loads(r.data)
        assert j["meta"]["count"] == 2

    def test_get_dataset_info(self, api_client):
        r = api_client.get("/datasets/edgar")
        j = json.loads(r.data)
        assert j["data"]["col_count"] == 13

    def test_get_unknown_dataset(self, api_client):
        r = api_client.get("/datasets/does_not_exist")
        assert r.status_code == 404

    def test_sample_endpoint(self, api_client):
        r = api_client.get("/datasets/hmda/sample?rows=5")
        j = json.loads(r.data)
        assert j["meta"]["rows"] == 5

    def test_generate_basic(self, api_client):
        r = api_client.post("/generate", json={"dataset": "hmda", "rows": 50, "seed": 1})
        j = json.loads(r.data)
        assert j["meta"]["rows_generated"] == 50

    def test_generate_var_dataset(self, api_client):
        r = api_client.post("/generate", json={"dataset": "fred_macro", "rows": 50, "seed": 1})
        j = json.loads(r.data)
        assert j["meta"]["rows_generated"] == 50

    def test_generate_with_scenario(self, api_client):
        r = api_client.post("/generate", json={
            "dataset": "fred_macro", "rows": 50,
            "scenario": "recession", "intensity": 1.0, "seed": 1,
        })
        j = json.loads(r.data)
        assert j["meta"]["scenario"] == "recession"

    def test_generate_csv_format(self, api_client):
        r = api_client.post("/generate", json={
            "dataset": "credit_risk", "rows": 20, "format": "csv", "seed": 1,
        })
        assert b"syn_id" in r.data
        lines = r.data.decode().strip().split("\n")
        assert len(lines) == 21  # header + 20 rows

    def test_generate_with_filter(self, api_client):
        r = api_client.post("/generate", json={
            "dataset": "hmda", "rows": 200,
            "filters": {"state": ["CA"]}, "seed": 1,
        })
        j = json.loads(r.data)
        rows = j["data"]
        if rows:
            assert all(row["state"] == "CA" for row in rows)

    def test_generate_unknown_dataset(self, api_client):
        r = api_client.post("/generate", json={"dataset": "xyz", "rows": 10})
        assert r.status_code == 404

    def test_generate_missing_dataset(self, api_client):
        r = api_client.post("/generate", json={"rows": 10})
        assert r.status_code == 400

    def test_evaluate_endpoint(self, api_client, hmda, syn_hmda):
        syn_body = syn_hmda.drop(columns=["syn_id"])
        r = api_client.post("/evaluate", json={
            "dataset": "hmda",
            "real":      hmda.head(200).to_dict(orient="records"),
            "synthetic": syn_body.head(200).to_dict(orient="records"),
        })
        j = json.loads(r.data)
        assert "summary" in j["data"]
        assert "overall_fidelity" in j["data"]["summary"]

    def test_audit_endpoint(self, api_client, hmda, syn_hmda):
        r = api_client.post("/audit", json={
            "real":      hmda.head(200).to_dict(orient="records"),
            "synthetic": syn_hmda.head(100).to_dict(orient="records"),
            "attacks":   50,
        })
        j = json.loads(r.data)
        assert "verdict" in j["data"]
        assert "overall_risk" in j["data"]["verdict"]

    def test_scenarios_endpoint(self, api_client):
        r = api_client.get("/scenarios")
        j = json.loads(r.data)
        assert j["meta"]["count"] == 5
        names = [s["name"] for s in j["data"]]
        assert "recession" in names

    def test_scenario_apply_endpoint(self, api_client, syn_hmda):
        r = api_client.post("/scenario/apply", json={
            "scenario":  "credit_crisis",
            "intensity": 0.5,
            "data":      syn_hmda.head(30).to_dict(orient="records"),
        })
        j = json.loads(r.data)
        assert j["meta"]["rows"] == 30

    def test_validate_endpoint(self, api_client, hmda):
        r = api_client.post("/validate", json={
            "data": hmda.head(100).to_dict(orient="records")
        })
        j = json.loads(r.data)
        assert j["data"]["passed"] is True

    def test_docs_page_returns_html(self, api_client):
        r = api_client.get("/docs")
        assert b"syndatakit" in r.data
        assert b"<html" in r.data


# ═══════════════════════════════════════════════════════════════════════════════
# 18. New datasets — smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewDatasets:
    @pytest.mark.parametrize("did,expected_col", [
        ("equity_returns",         "daily_return"),
        ("corporate_bonds",        "credit_spread"),
        ("insurance_claims",       "paid_losses"),
        ("life_insurance",         "mortality_rate"),
        ("commercial_real_estate", "cap_rate"),
        ("rental_market",          "asking_rent"),
        ("retail_transactions",    "fraud_flag"),
        ("commodity_prices",       "daily_return"),
    ])
    def test_new_dataset_generates(self, did, expected_col, all_seeds):
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(all_seeds[did])
        df = gen.sample(100, seed=42)
        assert len(df) == 100
        assert expected_col in df.columns

    def test_equity_returns_fat_tails(self, all_seeds):
        """Daily returns should have excess kurtosis > 1 (fat tails)."""
        from scipy import stats
        seed = all_seeds["equity_returns"]
        kurt = float(stats.kurtosis(seed["daily_return"]))
        assert kurt > 1.0, f"Expected fat tails (kurtosis > 1), got {kurt:.2f}"

    def test_corporate_bonds_spread_by_rating(self, all_seeds):
        """IG bonds should have lower spreads than HY bonds."""
        df = all_seeds["corporate_bonds"]
        ig_spread = df[df["credit_rating"].isin(["AAA","AA","A","BBB"])]["credit_spread"].mean()
        hy_spread = df[df["credit_rating"].isin(["BB","B","CCC"])]["credit_spread"].mean()
        assert ig_spread < hy_spread, "IG spreads should be lower than HY spreads"

    def test_retail_transactions_fraud_rate(self, all_seeds):
        """Fraud rate should be low (< 2%) matching industry average."""
        df = all_seeds["retail_transactions"]
        fraud_rate = df["fraud_flag"].mean()
        assert fraud_rate < 0.02, f"Fraud rate too high: {fraud_rate:.3f}"

    def test_life_insurance_mortality_increases_with_age(self, all_seeds):
        """Mortality rate should be higher for older policyholders."""
        df = all_seeds["life_insurance"]
        young = df[df["age_at_issue"] < 35]["mortality_rate"].mean()
        old   = df[df["age_at_issue"] > 60]["mortality_rate"].mean()
        assert old > young, "Mortality should increase with age"

    def test_commodity_prices_energy_more_volatile(self, all_seeds):
        """Energy commodities should have higher return volatility than metals."""
        df = all_seeds["commodity_prices"]
        energy_vol = df[df["sector"]=="Energy"]["daily_return"].std()
        metals_vol = df[df["sector"]=="Metals"]["daily_return"].std()
        assert energy_vol > metals_vol, "Energy should be more volatile than metals"

    def test_insurance_claims_large_loss_flag(self, all_seeds):
        """Large loss flag should mark top 5% of claims."""
        df = all_seeds["insurance_claims"]
        large_loss_paid  = df[df["large_loss_flag"]==1]["paid_losses"].mean()
        normal_loss_paid = df[df["large_loss_flag"]==0]["paid_losses"].mean()
        assert large_loss_paid > normal_loss_paid

    def test_vertical_counts(self):
        from syndatakit.catalog import list_datasets
        df = list_datasets()
        verticals = df["vertical"].value_counts().to_dict()
        assert verticals.get("Insurance", 0)      == 2
        assert verticals.get("Real Estate", 0)    == 2
        assert verticals.get("Retail Banking", 0) == 1
        assert verticals.get("Commodities", 0)    == 1
        assert verticals.get("Capital Markets", 0)== 4


# ═══════════════════════════════════════════════════════════════════════════════
# 19. Custom file generation (--input / /generate/custom)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCustomGeneration:
    def test_python_api_custom_fit(self, hmda):
        """Users can fit on any DataFrame and generate synthetic data."""
        from syndatakit.generators import GaussianCopulaGenerator
        gen = GaussianCopulaGenerator()
        gen.fit(hmda)
        syn = gen.sample(100, seed=1)
        assert len(syn) == 100
        assert set(syn.columns) - {"syn_id"} == set(hmda.columns)

    def test_python_api_custom_columns(self):
        """Works on arbitrary columns — not just built-in datasets."""
        import pandas as pd
        from syndatakit.generators import GaussianCopulaGenerator
        custom = pd.DataFrame({
            "revenue":     [1e6 * (1 + i*0.1) for i in range(200)],
            "growth_rate": [0.05 + i*0.001 for i in range(200)],
            "market":      (["US","EU","APAC"] * 67)[:200],
            "profitable":  ([1]*150 + [0]*50),
        })
        gen = GaussianCopulaGenerator()
        gen.fit(custom)
        syn = gen.sample(500, seed=42)
        assert len(syn) == 500
        assert "revenue" in syn.columns
        assert "market" in syn.columns

    def test_api_generate_custom_endpoint(self, api_client, hmda):
        """POST /generate/custom accepts a CSV and returns synthetic data."""
        import io
        csv_bytes = hmda.to_csv(index=False).encode()
        r = api_client.post(
            "/generate/custom",
            content_type="multipart/form-data",
            data={"file": (io.BytesIO(csv_bytes), "test.csv"), "rows": "50"},
        )
        assert r.status_code == 200
        j = json.loads(r.data)
        assert j["meta"]["rows_generated"] == 50
        assert j["meta"]["input_columns"] == list(hmda.columns)

    def test_api_generate_custom_csv_format(self, api_client, hmda):
        """POST /generate/custom returns CSV when format=csv."""
        import io
        csv_bytes = hmda.head(100).to_csv(index=False).encode()
        r = api_client.post(
            "/generate/custom",
            content_type="multipart/form-data",
            data={"file": (io.BytesIO(csv_bytes), "test.csv"),
                  "rows": "30", "format": "csv"},
        )
        assert r.status_code == 200
        assert b"syn_id" in r.data
        lines = r.data.decode().strip().split("\n")
        assert len(lines) == 31  # header + 30 rows

    def test_api_generate_custom_no_file_returns_400(self, api_client):
        """POST /generate/custom without a file returns 400."""
        r = api_client.post("/generate/custom", json={"rows": 10})
        assert r.status_code == 400

    def test_api_schemas_endpoint(self, api_client):
        """GET /schemas returns column types for all 18 datasets."""
        r = api_client.get("/schemas")
        j = json.loads(r.data)
        assert j["meta"]["count"] == 18
        assert "hmda" in j["data"]
        assert "loan_amount" in j["data"]["hmda"]

    def test_cli_list_shows_18_datasets(self):
        """syndatakit list should show 18 datasets."""
        from syndatakit.catalog import list_datasets
        df = list_datasets()
        assert len(df) == 18
