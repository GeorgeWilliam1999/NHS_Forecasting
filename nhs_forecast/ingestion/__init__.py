"""Ingestion layer.

Every source exposes a single ``load(settings) -> pandas.DataFrame`` function
returning data in a *canonical curated schema*. Loaders attempt a live fetch and
fall back to the synthetic generator when ``settings.use_synthetic`` is True or a
live fetch fails. This keeps the end-to-end pipeline runnable anywhere while the
real fetch code documents exactly how production access works.
"""
