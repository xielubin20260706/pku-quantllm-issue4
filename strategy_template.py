import numpy as np


class Strategy:
    name = "5DR 五日反转"
    lookback = 5
    stratification_field = "five_dr"
    stratification_ascending = True

    def __init__(self):
        self.last_rebalance_month = None

    def target_weights(self, ctx):
        month = (ctx.date.year, ctx.date.month)
        if month == self.last_rebalance_month:
            return None

        score = ctx.current("five_dr")
        candidates = np.flatnonzero(np.isfinite(score) & ctx.tradable())
        if not len(candidates):
            return None

        top_n = int(ctx.params["top_n"])
        chosen = candidates[np.argsort(score[candidates])[:top_n]]
        weights = np.zeros(len(ctx.symbols))
        weights[chosen] = 1.0 / len(chosen)
        self.last_rebalance_month = month
        return weights
