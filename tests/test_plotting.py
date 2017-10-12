import pytest

import matplotlib
matplotlib.use('AGG') # use a non-interactive backend
from matplotlib import pyplot as plt


from lifetimes import plotting
from lifetimes import BetaGeoFitter, ParetoNBDFitter, ModifiedBetaGeoFitter
from lifetimes.datasets import (load_cdnow_summary, load_transaction_data,
                                load_dataset)
from lifetimes import utils

TOLERANCE_VALUE = 20

bgf = BetaGeoFitter()
cd_data = load_cdnow_summary()
bgf.fit(cd_data['frequency'], cd_data['recency'], cd_data['T'], iterative_fitting=1)


@pytest.mark.plottest
class TestPlotting():

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_period_transactions(self):
        plt.figure()
        plotting.plot_period_transactions(bgf)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_period_transactions_parento(self):
        pnbd = ParetoNBDFitter()
        pnbd.fit(cd_data['frequency'], cd_data['recency'], cd_data['T'], iterative_fitting=1)

        plt.figure()
        plotting.plot_period_transactions(pnbd)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_period_transactions_mbgf(self):
        mbgf = ModifiedBetaGeoFitter()
        mbgf.fit(cd_data['frequency'], cd_data['recency'], cd_data['T'], iterative_fitting=1)

        plt.figure()
        plotting.plot_period_transactions(mbgf)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_period_transactions_max_frequency(self):
        plt.figure()
        plotting.plot_period_transactions(bgf, max_frequency=12)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_period_transactions_labels(self):
        plt.figure()
        plotting.plot_period_transactions(bgf, label=['A', 'B'])
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_frequency_recency_matrix(self):
        plt.figure()
        plotting.plot_frequency_recency_matrix(bgf)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_frequency_recency_matrix_max_recency(self):
        plt.figure()
        plotting.plot_frequency_recency_matrix(bgf, max_recency=100)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_frequency_recency_matrix_max_frequency(self):
        plt.figure()
        plotting.plot_frequency_recency_matrix(bgf, max_frequency=100)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_frequency_recency_matrix_max_frequency_max_recency(self):
        plt.figure()
        plotting.plot_frequency_recency_matrix(bgf, max_frequency=100, max_recency=100)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_probability_alive_matrix(self):
        plt.figure()
        plotting.plot_probability_alive_matrix(bgf)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_probability_alive_matrix_max_frequency(self):
        plt.figure()
        plotting.plot_probability_alive_matrix(bgf, max_frequency=100)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_probability_alive_matrix_max_recency(self):
        plt.figure()
        plotting.plot_probability_alive_matrix(bgf, max_recency=100)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_probability_alive_matrix_max_frequency_max_recency(self):
        plt.figure()
        plotting.plot_probability_alive_matrix(bgf, max_frequency=100, max_recency=100)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_expected_repeat_purchases(self):
        plt.figure()
        plotting.plot_expected_repeat_purchases(bgf)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_expected_repeat_purchases_with_label(self):
        plt.figure()
        plotting.plot_expected_repeat_purchases(bgf, label='test label')
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_transaction_rate_heterogeneity(self):
        """Test transactions rate heterogeneity."""
        plt.figure()
        plotting.plot_transaction_rate_heterogeneity(bgf)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_dropout_rate_heterogeneity(self):
        """Test dropout rate heterogeneity."""
        plt.figure()
        plotting.plot_dropout_rate_heterogeneity(bgf)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_customer_alive_history(self):
        plt.figure()
        transaction_data = load_transaction_data()
        # yes I know this is using the wrong data, but I'm testing plotting here.
        id_user = 35
        days_since_birth = 200
        sp_trans = transaction_data.loc[transaction_data['id'] == id_user]
        plotting.plot_history_alive(bgf, days_since_birth, sp_trans, 'date')
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_calibration_purchases_vs_holdout_purchases(self):
        transaction_data = load_transaction_data()
        summary = utils.calibration_and_holdout_data(transaction_data, 'id', 'date', '2014-09-01', '2014-12-31')
        bgf.fit(summary['frequency_cal'], summary['recency_cal'], summary['T_cal'])

        plt.figure()
        plotting.plot_calibration_purchases_vs_holdout_purchases(bgf, summary)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_calibration_purchases_vs_holdout_purchases_time_since_last_purchase(self):
        transaction_data = load_transaction_data()
        summary = utils.calibration_and_holdout_data(transaction_data, 'id', 'date', '2014-09-01', '2014-12-31')
        bgf.fit(summary['frequency_cal'], summary['recency_cal'], summary['T_cal'])

        plt.figure()
        plotting.plot_calibration_purchases_vs_holdout_purchases(bgf, summary, kind='time_since_last_purchase')
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_cumulative_transactions(self):
        """Test plotting cumultative transactions with CDNOW example."""
        transactions = load_dataset('CDNOW_sample.txt', header=None, sep='\s+')
        transactions.columns = ['id_total', 'id_sample', 'date', 'num_cd_purc',
                                'total_value']
        t = 39
        freq = 'W'

        transactions_summary = utils.summary_data_from_transaction_data(
            transactions, 'id_sample', 'date', datetime_format='%Y%m%d',
            observation_period_end='19970930', freq=freq)

        bgf = BetaGeoFitter(penalizer_coef=0.01)
        bgf.fit(transactions_summary['frequency'],
                transactions_summary['recency'], transactions_summary['T'])

        plt.figure()
        plotting.plot_cumulative_transactions(
            bgf, transactions, 'date', 'id_sample', 2 * t, t, freq=freq,
            xlabel='week', datetime_format='%Y%m%d')
        return plt.gcf()

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE_VALUE, style='default')
    def test_plot_incremental_transactions(self):
        """Test plotting incremental transactions with CDNOW example."""
        transactions = load_dataset('CDNOW_sample.txt', header=None, sep='\s+')
        transactions.columns = ['id_total', 'id_sample', 'date', 'num_cd_purc',
                                'total_value']
        t = 39
        freq = 'W'

        transactions_summary = utils.summary_data_from_transaction_data(
            transactions, 'id_sample', 'date', datetime_format='%Y%m%d',
            observation_period_end='19970930', freq=freq)

        bgf = BetaGeoFitter(penalizer_coef=0.01)
        bgf.fit(transactions_summary['frequency'],
                transactions_summary['recency'], transactions_summary['T'])

        plt.figure()
        plotting.plot_incremental_transactions(
            bgf, transactions, 'date', 'id_sample', 2 * t, t, freq=freq,
            xlabel='week', datetime_format='%Y%m%d')
        return plt.gcf()
